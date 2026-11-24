"""Step 3b - REFace face swaps for the frames of one shard (REF). Resumable. Runs in the "dfref" environment.

For each frame t_s_fk of the shard: the face of FaceForensics++ video s (real frame s_fk) is swapped into
the scene of video t (real frame t_fk), the same pairs FaceForensics++ Deepfakes used.
  1. REFace's own steps: FFHQ alignment (dlib), face parsing, DDIM 50 steps, guidance 3.5,
     seed 1000*s + k, checkpoint last.ckpt (Sanoojan/REFace), model in fp16 under autocast.
  2. Paste-back: the 512 px output and REFace's regenerated-region mask are warped back into the frame; the
     mask is feathered (sigma = 4% face size, alpha < 1/255 -> 0). Only the changed region is saved
     (RGBA PNG, A = 255 where alpha > 0), position in work/out/REF/_gen_log.csv.
Identical to the main project's 11_generate_ref.py apart from the shard list and the region-only output.

Needs setup_reface.py first.  Usage:  python code/generate_ref.py --shard shards/REF_shard1.txt [--limit 2]
"""
import os, sys, time, csv, json, argparse
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from common import SRC_REAL_FRAMES, MASKS, OUT, WORK, GEN_ROOT, load_config, real_name, log_to, machine_info, read_shard

REFACE = os.path.join(WORK, "REFace")
if not os.path.isdir(REFACE):
    sys.exit("REFace is not set up. Run:  python code/setup_reface.py")
os.chdir(REFACE)                      # REFace opens its configs and weights by relative path
sys.path.insert(0, REFACE)
import numpy as np, pandas as pd, cv2, torch
from PIL import Image
from omegaconf import OmegaConf
# REFace's transformers 4.19 cannot read the new HF cache / safetensors: load CLIP-L (pinned revision 32bd642)
# from the local folder made by setup_reface.py instead.
import transformers
LOCAL_CLIP = os.path.join(WORK, "clip-vit-large-patch14")
for _cls in (transformers.CLIPModel, transformers.CLIPVisionModel, transformers.CLIPTextModel, transformers.CLIPTokenizer):
    _orig = _cls.from_pretrained.__func__
    def _redirect(cls, name, *a, _o=_orig, **k):
        return _o(cls, LOCAL_CLIP if name == "openai/clip-vit-large-patch14" else name, *a, **k)
    _cls.from_pretrained = classmethod(_redirect)
import torchvision.transforms as T
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo

CONF = load_config()
CFG = "models/REFace/configs/project_ffhq.yaml"
CKPT = os.path.join(WORK, "REFace_weights", "last.ckpt")
STEPS, SCALE = 50, 3.5


def to_tensor(norm=True):
    t = [T.ToTensor()]
    if norm:
        t.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(t)


CLIPN = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def align(path):
    crops, _, quads = crop_faces(1024, [path], 1.0, center_sigma=0, xy_sigma=0, use_fa=False)
    quad = quads[0]
    inv = calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, 1024], [1024, 1024], [1024, 0]])
    return crops[0].convert("RGB"), quad, inv


class REFace:
    def __init__(self):
        conf = OmegaConf.load(CFG)
        sd = torch.load(CKPT, map_location="cpu")["state_dict"]
        self.model = instantiate_from_config(conf.model)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.half().cuda().eval()   # fp32 (~6 GB) overflows 8 GB GPUs
        self.sampler = DDIMSampler(self.model)
        self.parser = init_faceParsing_pretrained_model("default", "Other_dependencies/face_parsing/79999_iter.pth", "")
        p = conf.data.params.test.params
        self.remove_tar = list(p["remove_mask_tar_FFHQ"]); self.keep_src = list(p["preserve_mask_src_FFHQ"])

    def parse(self, pil1024):
        return faceParsing_demo(self.parser, pil1024.resize((1024, 1024), Image.BILINEAR), convert_to_seg12=True, model_name="default")

    @torch.no_grad()
    def swap(self, tgt_crop, src_crop, seed):
        m = self.model
        tmask = np.isin(self.parse(tgt_crop), self.remove_tar).astype(np.uint8) * 255
        smask = np.isin(self.parse(src_crop), self.keep_src).astype(np.uint8) * 255
        img = to_tensor()(tgt_crop.resize((512, 512)))
        keep = 1 - to_tensor(False)(Image.fromarray(tmask).resize((512, 512), Image.NEAREST))
        inpaint = img * keep
        ref = CLIPN(to_tensor(False)(src_crop.resize((224, 224), Image.BILINEAR))) * T.Resize((224, 224))(to_tensor(False)(Image.fromarray(smask)))
        x, inpaint, keep, ref = img[None].cuda(), inpaint[None].cuda(), keep[None].cuda(), ref[None].cuda()
        seed_everything(seed)
        with torch.autocast("cuda"), m.ema_scope():
            uc = m.learnable_vector.repeat(1, 1, 1)
            if m.stack_feat:
                uc = torch.cat([uc, m.other_learnable_vector.repeat(1, 1, 1)], dim=-1)
            lm = m.get_landmarks(x) if m.Landmark_cond else None
            c = m.conditioning_with_feat(ref.float(), landmarks=lm, tar=x.float()).float()
            if (m.land_mark_id_seperate_layers or m.sep_head_att):
                lmu = lm.unsqueeze(1) if len(lm.shape) != 3 else lm
                uc = torch.cat([uc, lmu], dim=-1)
            if c.shape[-1] == 1024:
                c = m.proj_out(c)
            if len(c.shape) == 2:
                c = c.unsqueeze(1)
            z = m.get_first_stage_encoding(m.encode_first_stage(inpaint)).detach()
            kw = {"inpaint_image": z, "inpaint_mask": T.Resize([z.shape[-1], z.shape[-1]])(keep)}
            s, _ = self.sampler.sample(S=STEPS, conditioning=c, batch_size=1, shape=[4, 64, 64], verbose=False,
                                       unconditional_guidance_scale=SCALE, unconditional_conditioning=uc, eta=0.0,
                                       x_T=None, test_model_kwargs=kw, src_im=ref.float(), tar=x)
            y = torch.clamp((m.decode_first_stage(s) + 1) / 2, 0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((y * 255).round().astype(np.uint8)), tmask


def paste(frame_rgb, out512, tmask1024, inv, fs):
    H, W = frame_rgb.shape[:2]
    gen = np.array(out512.resize((1024, 1024), Image.BILINEAR).transform((W, H), Image.PERSPECTIVE, inv, Image.BILINEAR))
    hard = np.array(Image.fromarray(tmask1024).transform((W, H), Image.PERSPECTIVE, inv, Image.NEAREST)) > 127
    mc = CONF["mask"]
    alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), mc["feather_sigma_frac"] * fs)
    alpha[alpha < mc["alpha_floor"]] = 0
    valid = np.array(Image.new("L", (1024, 1024), 255).transform((W, H), Image.PERSPECTIVE, inv, Image.NEAREST)) > 0
    alpha[~valid] = 0
    a = alpha[..., None]
    out = np.clip(np.round(a * gen.astype(np.float32) + (1 - a) * frame_rgb.astype(np.float32)), 0, 255).astype(np.uint8)
    z = alpha == 0
    assert (out[z] == frame_rgb[z]).all()
    return out, alpha


def save_region(out, alpha, path):
    ys, xs = np.nonzero(alpha > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([out[y0:y1, x0:x1], np.where(alpha[y0:y1, x0:x1] > 0, 255, 0).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(path, compress_level=6)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="shard file, e.g. shards/REF_shard1.txt")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (quick test)")
    a = ap.parse_args()
    shard_path = a.shard if os.path.isabs(a.shard) else os.path.join(GEN_ROOT, a.shard)
    frames = read_shard(shard_path)[: a.limit]
    out_dir = os.path.join(OUT, "REF"); os.makedirs(out_dir, exist_ok=True)
    log = log_to("gen_REF")
    with open(os.path.join(out_dir, "_machine.json"), "w") as fh:
        json.dump(dict(machine_info(), shard=os.path.basename(shard_path)), fh, indent=2)
    geo = pd.read_csv(os.path.join(MASKS, "_geometry.csv")).set_index("frame")
    lf = os.path.join(out_dir, "_gen_log.csv"); new = not os.path.exists(lf)
    logged = set() if new else set(pd.read_csv(lf).frame)      # includes frames REFace could not align
    todo = [n for n in frames if n not in logged and not os.path.exists(os.path.join(out_dir, n[:-4] + ".png"))]
    log(f"=== REF {os.path.basename(shard_path)}: {len(frames)} frames, {len(todo)} to do")
    R = REFace()
    fh = open(lf, "a", newline=""); wr = csv.writer(fh)
    if new:
        wr.writerow(["frame", "status", "seconds", "alpha_area_frac", "x0", "y0", "w", "h"])
    t_all = time.time()
    for i, name in enumerate(todo):
        t, s, k = int(name[:3]), int(name[4:7]), int(name[9])
        t0 = time.time()
        tgt_path, src_path = os.path.join(SRC_REAL_FRAMES, real_name(t, k)), os.path.join(SRC_REAL_FRAMES, real_name(s, k))
        try:
            tc, _, inv = align(tgt_path); sc, _, _ = align(src_path)
        except Exception:
            wr.writerow([name, "align_fail", 0, 0, "", "", "", ""]); fh.flush(); continue
        out512, tmask = R.swap(tc, sc, 1000 * s + k)
        frame = np.array(Image.open(tgt_path).convert("RGB"))
        g = geo.loc[real_name(t, k)] if geo.status.get(real_name(t, k)) == "ok" else None
        fs = float(g.fsize) if g is not None else 0.25 * min(frame.shape[:2])
        out, alpha = paste(frame, out512, tmask, inv, fs)
        if not (alpha > 0).any():
            wr.writerow([name, "empty_mask", round(time.time() - t0, 2), 0, "", "", "", ""]); fh.flush(); continue
        x0, y0, w, h = save_region(out, alpha, os.path.join(out_dir, name[:-4] + ".png"))
        dt = time.time() - t0
        wr.writerow([name, "ok", round(dt, 2), round(float((alpha > 0).mean()), 5), x0, y0, w, h]); fh.flush()
        if i % 25 == 0 or i == len(todo) - 1:
            el = time.time() - t_all
            log(f"  {i + 1}/{len(todo)}  {dt:.1f}s/img (avg {el / (i + 1):.1f}s)  ETA {el / (i + 1) * (len(todo) - i - 1) / 3600:.1f} h")
    log("=== REF done")


if __name__ == "__main__":
    main()
