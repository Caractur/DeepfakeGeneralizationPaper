"""Step 3 - Generate paste-back diffusion fakes for the frames of one shard (Kandinsky 2.2 = K22). Resumable.

The generation itself is identical to the main project's 11_generate.py:
  1. Shared face mask (masks.py) and face size fs.
     alpha   = GaussianBlur(mask, sigma = 4% fs); alpha < 1/255 -> 0   (feathered blend weight)
     inpaint = mask dilated by 3 sigma (covers every pixel where alpha > 0)
  2. Square work crop: side = max(3.0 fs, 1.1 x inpaint-mask extent), centred on the face, clamped.
  3. Resize the crop (Lanczos) to the generator's resolution, inpaint with the configured prompt and the
     per-frame seed 1000*s + k, resize back (Lanczos). Quality check (qc.py); a failed face is regenerated
     with seed + 1,000,003 x attempt, up to 4 attempts.
  4. out = alpha * generated + (1 - alpha) * real.

What is different for helpers: only the CHANGED region is saved, as an RGBA PNG
  work/out/<GEN>/<frame>.png   RGB = the composited pixels, A = 255 where alpha > 0 (0 elsewhere)
cut to the bounding box of alpha > 0; its position (x0, y0) is in work/out/<GEN>/_gen_log.csv.
The main author pastes these pixels onto their own copy of the real frame, so the background stays
byte-identical to the real frame no matter which machine generated the face.

Usage:  python code/generate.py --gen K22 --shard shards/K22_shard1.txt   [--limit 2 for a quick test]
"""
import os, sys, time, csv, json, argparse, math
import numpy as np, pandas as pd, cv2, torch
from PIL import Image
from common import SRC_REAL_FRAMES, MASKS, OUT, GEN_ROOT, load_config, log_to, machine_info, read_shard

CONF = load_config()


def prepare(f, fs, scale=None):
    """Real frame, alpha, inpaint mask and work-crop box for one frame."""
    mc = CONF["mask"]
    rgb = cv2.cvtColor(cv2.imread(os.path.join(SRC_REAL_FRAMES, f)), cv2.COLOR_BGR2RGB)
    hard = cv2.imread(os.path.join(MASKS, f[:-4] + ".png"), 0).astype(np.float32) / 255.0
    sigma = mc["feather_sigma_frac"] * fs
    alpha = cv2.GaussianBlur(hard, (0, 0), sigma)
    alpha[alpha < mc["alpha_floor"]] = 0.0
    r = int(math.ceil(mc["inpaint_extra_sigmas"] * sigma))
    inpaint = cv2.dilate((hard > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    assert not ((alpha > 0) & (inpaint == 0)).any(), "feather exceeds inpaint mask"
    H, W = hard.shape
    ys, xs = np.nonzero(inpaint)
    ext = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    pts = np.load(os.path.join(MASKS, f[:-4] + "_pts.npy"))
    cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
    side = int(round(max((scale or CONF["work_crop"]["scale"]) * fs, 1.1 * ext)))
    side = min(side, H, W)
    x1 = int(round(min(max(cx - side / 2, 0), W - side)))
    y1 = int(round(min(max(cy - side / 2, 0), H - side)))
    box = (x1, y1, x1 + side, y1 + side)
    outside = np.ones_like(alpha, bool); outside[y1:y1 + side, x1:x1 + side] = False
    lost = float(alpha[outside].sum() / max(1e-6, alpha.sum()))
    alpha[outside] = 0.0
    return rgb, alpha, inpaint, box, lost


class Gen:
    def __init__(self, name):
        self.name = name
        self.c = CONF["generators"][name]
        ps = CONF["prompts"][self.c.get("prompt_set") or CONF["prompt_set"]]
        self.P, self.N = ps["prompt"], ps["negative"]
        self.res = self.c["resolution"]
        self.pipe = self.prior = None
        if name == "SD15":
            from diffusers import StableDiffusionInpaintPipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.c["model"], revision=self.c["revision"], torch_dtype=torch.float16,
                safety_checker=None, requires_safety_checker=False).to("cuda")
        elif name == "SDXL":
            from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
            vae = AutoencoderKL.from_pretrained(self.c["vae"], revision=self.c["vae_revision"], torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.c["model"], revision=self.c["revision"], vae=vae, torch_dtype=torch.float16, variant="fp16")
            self.pipe.enable_model_cpu_offload()
        elif name == "K22":
            from diffusers import KandinskyV22PriorPipeline, KandinskyV22InpaintPipeline
            self.prior = KandinskyV22PriorPipeline.from_pretrained(self.c["prior"], revision=self.c["prior_revision"], torch_dtype=torch.float16)
            self.pipe = KandinskyV22InpaintPipeline.from_pretrained(self.c["model"], revision=self.c["revision"], torch_dtype=torch.float16)
            self.prior.enable_model_cpu_offload(); self.pipe.enable_model_cpu_offload()
        else:
            raise ValueError(name)
        self.pipe.set_progress_bar_config(disable=True)
        if self.prior is not None:
            self.prior.set_progress_bar_config(disable=True)

    def __call__(self, crop, mask, seed):
        R = self.res
        img = Image.fromarray(crop).resize((R, R), Image.LANCZOS)
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((R, R), Image.NEAREST)
        g = torch.Generator("cpu").manual_seed(seed)
        P, N = self.P, self.N
        if self.name == "K22":
            cond = self.c.get("conditioning", "text")
            if cond.startswith("interp"):
                ctx = crop.copy()
                if cond == "interp":
                    ctx[mask > 0] = 128
                elif cond == "interp_blur":
                    b = cv2.GaussianBlur(crop, (0, 0), crop.shape[0] / 25)
                    ctx[mask > 0] = b[mask > 0]
                w = self.c.get("context_weight", 0.5)
                emb, neg = self.prior.interpolate([Image.fromarray(ctx), P], [w, 1 - w], negative_prompt=N,
                                                  guidance_scale=self.c.get("prior_guidance", 4.0), generator=g,
                                                  num_inference_steps=self.c.get("prior_steps", 25)).to_tuple()
            else:
                emb, neg = self.prior(P, negative_prompt=N, guidance_scale=self.c.get("prior_guidance", 4.0),
                                      generator=g, num_inference_steps=self.c.get("prior_steps", 25)).to_tuple()
            out = self.pipe(image=img, mask_image=m, image_embeds=emb, negative_image_embeds=neg, height=R, width=R,
                            num_inference_steps=self.c["steps"], guidance_scale=self.c["guidance"], generator=g).images[0]
        else:
            out = self.pipe(prompt=P, negative_prompt=N, image=img, mask_image=m, height=R, width=R,
                            num_inference_steps=self.c["steps"], guidance_scale=self.c["guidance"],
                            strength=self.c["strength"], generator=g).images[0]
        return np.array(out.convert("RGB").resize(crop.shape[1::-1], Image.LANCZOS))


def make_one(gen, f, s, k, fs, box, qc):
    rgb, alpha, inpaint, (x1, y1, x2, y2), lost = prepare(f, fs)
    crop = rgb[y1:y2, x1:x2]
    a = alpha[y1:y2, x1:x2, None]
    attempts = CONF["qc"]["max_attempts"] if qc is not None else 1
    for att in range(attempts):
        out_crop = gen(crop, inpaint[y1:y2, x1:x2], 1000 * s + k + 1000003 * att)
        out = rgb.copy()
        out[y1:y2, x1:x2] = np.clip(np.round(a * out_crop.astype(np.float32) + (1 - a) * crop.astype(np.float32)), 0, 255).astype(np.uint8)
        q = qc.score(np.ascontiguousarray(out[box[1]:box[3], box[0]:box[2]])) if qc is not None else dict(det=-1, p_face=-1, ok=True)
        if q["ok"]:
            break
    z = alpha == 0
    assert (out[z] == rgb[z]).all(), "background changed"
    return out, alpha, x2 - x1, lost, att + 1, q


def save_region(out, alpha, path):
    """RGBA PNG of the bounding box of alpha > 0; returns (x0, y0, w, h)."""
    ys, xs = np.nonzero(alpha > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([out[y0:y1, x0:x1], np.where(alpha[y0:y1, x0:x1] > 0, 255, 0).astype(np.uint8)])
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="K22")
    ap.add_argument("--shard", required=True, help="shard file, e.g. shards/K22_shard1.txt")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (quick test)")
    a = ap.parse_args()
    shard_path = a.shard if os.path.isabs(a.shard) else os.path.join(GEN_ROOT, a.shard)
    frames = read_shard(shard_path)[: a.limit]
    out_dir = os.path.join(OUT, a.gen); os.makedirs(out_dir, exist_ok=True)
    log = log_to(f"gen_{a.gen}")
    with open(os.path.join(out_dir, "_machine.json"), "w") as fh:
        json.dump(dict(machine_info(), shard=os.path.basename(shard_path)), fh, indent=2)
    geo = pd.read_csv(os.path.join(MASKS, "_geometry.csv")).set_index("frame")
    missing_mask = [f for f in frames if geo.status.get(f) != "ok"]
    if missing_mask:
        log(f"WARNING: {len(missing_mask)} frames have no face mask on this machine and are skipped: {missing_mask[:5]}")
    todo = [f for f in frames if f not in missing_mask and not os.path.exists(os.path.join(out_dir, f[:-4] + ".png"))]
    log(f"=== {a.gen} {os.path.basename(shard_path)}: {len(frames)} frames, {len(todo)} to do")
    gen = Gen(a.gen)
    qc = None
    if CONF["qc"]["enabled"]:
        from qc import QC
        qc = QC()
    lf = os.path.join(out_dir, "_gen_log.csv"); new = not os.path.exists(lf)
    fh = open(lf, "a", newline=""); wr = csv.writer(fh)
    if new:
        wr.writerow(["frame", "seconds", "work_side", "alpha_lost_outside_crop", "alpha_area_frac", "attempts",
                     "qc_det", "qc_p_face", "qc_ok", "x0", "y0", "w", "h"])
    t_all = time.time()
    for i, f in enumerate(todo):
        s, k = int(f[:3]), int(f[5])
        t0 = time.time()
        gg = geo.loc[f]; box = (int(gg.x1), int(gg.y1), int(gg.x2), int(gg.y2))
        out, alpha, side, lost, att, q = make_one(gen, f, s, k, float(gg.fsize), box, qc)
        x0, y0, w, h = save_region(out, alpha, os.path.join(out_dir, f[:-4] + ".png"))
        dt = time.time() - t0
        wr.writerow([f, round(dt, 2), side, round(lost, 5), round(float((alpha > 0).mean()), 5), att,
                     round(q["det"], 3), round(q["p_face"], 3), int(q["ok"]), x0, y0, w, h]); fh.flush()
        if i % 25 == 0 or i == len(todo) - 1:
            el = time.time() - t_all
            log(f"  {i + 1}/{len(todo)}  {dt:.1f}s/img (avg {el / (i + 1):.1f}s)  ETA {el / (i + 1) * (len(todo) - i - 1) / 3600:.1f} h")
    log(f"=== {a.gen} done")


if __name__ == "__main__":
    main()
