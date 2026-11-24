"""Step 4 - Check a finished shard and pack it for sending back.

  python code/check_shard.py --gen K22 --shard shards/K22_shard1.txt
  python code/check_shard.py --gen REF --shard shards/REF_shard1.txt

Checks that every frame of the shard was processed (or has a logged reason why not), that every saved image
opens and matches its log entry, and summarises the quality check. Then writes
  work/return/<GEN>_<shard>.zip   (the images of this shard, their log rows, machine info, your mask comparison)
and prints its size and SHA-256. Send the zip AND the printed SHA-256 to the main author.
"""
import os, sys, json, zipfile, hashlib, argparse
import numpy as np, pandas as pd
from PIL import Image
from common import OUT, RETURN, MASKS, GEN_ROOT, read_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=["K22", "REF"])
    ap.add_argument("--shard", required=True)
    ap.add_argument("--allow_incomplete", action="store_true", help="pack even if frames are missing (partial delivery)")
    a = ap.parse_args()
    shard_path = a.shard if os.path.isabs(a.shard) else os.path.join(GEN_ROOT, a.shard)
    shard = os.path.splitext(os.path.basename(shard_path))[0]
    frames = read_shard(shard_path)
    out_dir = os.path.join(OUT, a.gen)
    logf = os.path.join(out_dir, "_gen_log.csv")
    if not os.path.exists(logf):
        sys.exit(f"No log found at {logf}: nothing generated yet.")
    log = pd.read_csv(logf).drop_duplicates("frame", keep="last").set_index("frame")
    problems, images = [], []
    for f in frames:
        png = os.path.join(out_dir, f[:-4] + ".png")
        if f not in log.index:
            problems.append(f"{f}: not processed" + (" (image exists but no log row: rerun generation)" if os.path.exists(png) else ""))
            continue
        r = log.loc[f]
        if a.gen == "REF" and r.status != "ok":
            continue                                   # REFace could not align this frame: a valid, logged outcome
        if not os.path.exists(png):
            problems.append(f"{f}: logged but image missing"); continue
        try:
            im = Image.open(png); im.load()
            if im.mode != "RGBA" or im.size != (int(r.w), int(r.h)):
                problems.append(f"{f}: image is {im.mode} {im.size}, log says RGBA {(int(r.w), int(r.h))}")
            elif not (np.asarray(im)[..., 3] > 0).any():
                problems.append(f"{f}: empty changed region")
        except Exception as e:
            problems.append(f"{f}: cannot open image ({e})"); continue
        images.append(png)
    rows = log.loc[[f for f in frames if f in log.index]]
    print(f"{a.gen} {shard}: {len(frames)} frames in shard, {len(rows)} processed, {len(images)} images")
    if a.gen == "K22":
        print(f"  quality check: passed {int(rows.qc_ok.sum())}, failed after all attempts {int((rows.qc_ok == 0).sum())}; "
              f"attempts {rows.attempts.value_counts().sort_index().to_dict()}")
    else:
        print(f"  status: {rows.status.value_counts().to_dict()}")
    if len(rows):
        print(f"  time: {rows.seconds.sum() / 3600:.1f} h, median {rows.seconds.median():.1f} s/frame")
    if problems:
        print(f"PROBLEMS ({len(problems)}):"); [print("  " + p) for p in problems[:20]]
        if not a.allow_incomplete:
            sys.exit("Not packed. Finish the shard (rerun the same generate command; it resumes) or use --allow_incomplete.")
    zpath = os.path.join(RETURN, f"{a.gen}_{shard}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in images:
            z.write(p, f"{a.gen}/{os.path.basename(p)}")
        z.writestr(f"{a.gen}/_gen_log.csv", rows.reset_index().to_csv(index=False))
        z.writestr(f"{a.gen}/_shard.txt", shard)
        if os.path.exists(os.path.join(out_dir, "_machine.json")):
            z.write(os.path.join(out_dir, "_machine.json"), f"{a.gen}/_machine.json")
        z.write(os.path.join(MASKS, "_geometry.csv"), "_geometry.csv")
    h = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    print(f"\nPacked: {zpath}\n  size {os.path.getsize(zpath) / 1e6:.1f} MB\n  SHA-256 {h}")
    print("Send this zip and the SHA-256 line to the main author (private link only).")


if __name__ == "__main__":
    main()
