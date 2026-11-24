"""Step 2 - Face masks and face boxes for the 5,000 real frames (same code as the main project's 10_masks.py).

For each frame:
  1. MediaPipe FaceDetection (full range, conf >= 0.5), highest-scoring face -> face box.
  2. MediaPipe FaceMesh (static, up to 4 faces); the mesh whose centre lies inside the detection box.
  3. Mask = convex hull of the 468 landmarks, dilated by 8% of the face size. Saved as a 0/255 PNG.
Output: work/masks/native/<frame>.png, <frame>_pts.npy and _geometry.csv. Resumable.

At the end the masks are compared with the main machine's (data/masks_reference.csv: a hash of each
mask's pixels, never the masks themselves). Small differences are possible on other hardware and are
handled later; a large mismatch should be reported to the main author.
"""
import os, csv, hashlib
import numpy as np, pandas as pd, cv2
from common import SRC_REAL_FRAMES, MASKS, DATA_DIR, load_config, patch_protobuf_for_mediapipe

patch_protobuf_for_mediapipe()
import mediapipe as mp

CONF = load_config()
BOX_SCALE, CROP_SCALE = CONF["crop"]["box_scale"], CONF["crop"]["crop_scale"]


def crop_box(det_box, w, h):
    xm, ym, bw, bh = det_box
    bw, bh = bw * w, bh * h
    cx, cy = (xm + det_box[2] / 2) * w, (ym + det_box[3] / 2) * h
    side = BOX_SCALE * max(bw, bh) * CROP_SCALE
    x1, y1 = int(round(max(0, cx - side / 2))), int(round(max(0, cy - side / 2)))
    x2, y2 = int(round(min(w, cx + side / 2))), int(round(min(h, cy + side / 2)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def detect_box(det, rgb):
    h, w = rgb.shape[:2]
    r = det.process(rgb)
    if not r.detections:
        return None, None
    b = max(r.detections, key=lambda d: d.score[0]).location_data.relative_bounding_box
    rel = (b.xmin, b.ymin, b.width, b.height)
    abs_box = (b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h)
    return crop_box(rel, w, h), abs_box


def geometry(det, mesh, rgb):
    h, w = rgb.shape[:2]
    box, dbox = detect_box(det, rgb)
    if box is None:
        return "no_face"
    r = mesh.process(rgb)
    if not r.multi_face_landmarks:
        return "no_mesh"
    best, best_d = None, 1e9
    dcx, dcy = (dbox[0] + dbox[2]) / 2, (dbox[1] + dbox[3]) / 2
    for fl in r.multi_face_landmarks:
        pts = np.array([(l.x * w, l.y * h) for l in fl.landmark], np.float32)
        cx, cy = pts.mean(0)
        if not (dbox[0] <= cx <= dbox[2] and dbox[1] <= cy <= dbox[3]):
            continue
        d = (cx - dcx) ** 2 + (cy - dcy) ** 2
        if d < best_d:
            best, best_d = pts, d
    if best is None:
        return "mesh_outside_box"
    fsize = float(max(np.ptp(best[:, 0]), np.ptp(best[:, 1])))
    return dict(crop=box, pts=best, fsize=fsize)


def hull_mask(shape, pts, dilate_px):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m


def compute():
    files = sorted(f for f in os.listdir(SRC_REAL_FRAMES) if f.endswith(".jpg"))
    frac = CONF["mask"]["dilate_frac"]
    os.makedirs(MASKS, exist_ok=True)
    geo_f = os.path.join(MASKS, "_geometry.csv")
    done = set()
    if os.path.exists(geo_f):
        with open(geo_f) as fh:
            done = {r["frame"] for r in csv.DictReader(fh)}
    new = not os.path.exists(geo_f)
    fh = open(geo_f, "a", newline=""); wr = csv.writer(fh)
    if new:
        wr.writerow(["frame", "status", "w", "h", "x1", "y1", "x2", "y2", "fsize", "dilate_px", "mask_frac"])
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det, \
         mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4, refine_landmarks=False,
                                         min_detection_confidence=0.5) as mesh:
        for i, f in enumerate(files):
            if f in done:
                continue
            rgb = cv2.cvtColor(cv2.imread(os.path.join(SRC_REAL_FRAMES, f)), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            g = geometry(det, mesh, rgb)
            if isinstance(g, str):
                wr.writerow([f, g, w, h] + [""] * 7)
                continue
            dpx = int(round(frac * g["fsize"]))
            m = hull_mask(rgb.shape, g["pts"], dpx)
            cv2.imwrite(os.path.join(MASKS, f[:-4] + ".png"), m)
            np.save(os.path.join(MASKS, f[:-4] + "_pts.npy"), g["pts"].astype(np.float32))
            wr.writerow([f, "ok", w, h, *g["crop"], round(g["fsize"], 2), dpx, round(float((m > 0).mean()), 5)])
            if i % 250 == 0:
                fh.flush(); print(f"  masks: {i}/{len(files)}", flush=True)
    fh.close()


def compare():
    mine = pd.read_csv(os.path.join(MASKS, "_geometry.csv")).set_index("frame")
    ref = pd.read_csv(os.path.join(DATA_DIR, "masks_reference.csv")).set_index("frame")
    same_status = (mine.status.reindex(ref.index) == ref.status).mean()
    ok = ref.index[(ref.status == "ok") & (mine.status.reindex(ref.index) == "ok")]
    px_same = np.mean([hashlib.sha256(cv2.imread(os.path.join(MASKS, f[:-4] + ".png"), 0).tobytes()).hexdigest()[:16]
                       == ref.mask_px_sha16[f] for f in ok])
    box_same = np.mean([(mine.loc[f, ["x1", "y1", "x2", "y2"]].astype(int).values == ref.loc[f, ["x1", "y1", "x2", "y2"]].astype(int).values).all() for f in ok])
    print(f"Comparison with the main machine: same face found/not found {100 * same_status:.2f}%, "
          f"identical face box {100 * box_same:.2f}%, identical mask pixels {100 * px_same:.2f}% (n={len(ok)})")
    if same_status < 0.98 or box_same < 0.95:
        print("WARNING: large difference from the main machine. Report these numbers to the main author before generating.")
    else:
        print("OK")


if __name__ == "__main__":
    compute()
    compare()
