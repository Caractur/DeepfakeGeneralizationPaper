"""Step 1 - Get the 5,000 real FaceForensics++ frames the study uses.

Source: Kaggle dataset fatimahirshad/faceforensics-extracted-dataset-c23 (CC BY-NC 4.0), the same frames the
paper used. Downloads the zip (~4.1 GB, resumable), extracts only FF++C32-Frames/Original/*.jpg into
work/real_frames/, and checks every file against data/real_frames.sha256 (byte-identical to the main machine).

If the automatic download fails (Kaggle sometimes requires signing in), download the zip in a browser from
  https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23
and run:  python code/fetch_data.py --zip <path to the downloaded zip>

The frames show real people from FaceForensics++ videos: keep them private and do not re-upload them anywhere.
"""
import os, sys, zipfile, hashlib, argparse
from common import WORK, SRC_REAL_FRAMES, DATA_DIR

URL = "https://www.kaggle.com/api/v1/datasets/download/fatimahirshad/faceforensics-extracted-dataset-c23"
PREFIX = "FF++C32-Frames/Original/"


def download(dst):
    import requests
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(URL, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
        if r.status_code == 416:            # already complete
            return
        if r.status_code not in (200, 206):
            sys.exit(f"Download failed (HTTP {r.status_code}). Download the zip in a browser (see the top of this file) "
                     f"and rerun with --zip <path>.")
        if r.status_code == 200 and have:   # server ignored the range: start over
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        done = have
        with open(dst, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                if total and (done // (1 << 20)) % 200 == 0:
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="path to an already downloaded Kaggle zip")
    ap.add_argument("--keep_zip", action="store_true", help="keep the 4 GB zip after extracting")
    a = ap.parse_args()
    expected = {}
    with open(os.path.join(DATA_DIR, "real_frames.sha256")) as fh:
        for line in fh:
            h, name = line.split()
            expected[name] = h
    os.makedirs(SRC_REAL_FRAMES, exist_ok=True)
    todo = [n for n in expected if not os.path.exists(os.path.join(SRC_REAL_FRAMES, n))]
    if todo:
        zpath = a.zip
        if zpath is None:
            os.makedirs(os.path.join(WORK, "downloads"), exist_ok=True)
            zpath = os.path.join(WORK, "downloads", "ffpp_c23.zip")
            print("Downloading the Kaggle dataset (~4.1 GB) ...", flush=True)
            download(zpath)
        print("Extracting the real frames ...", flush=True)
        with zipfile.ZipFile(zpath) as z:
            for n in todo:
                with z.open(PREFIX + n) as src, open(os.path.join(SRC_REAL_FRAMES, n), "wb") as out:
                    out.write(src.read())
        if a.zip is None and not a.keep_zip:
            os.remove(zpath)
    bad = [n for n, h in expected.items()
           if hashlib.sha256(open(os.path.join(SRC_REAL_FRAMES, n), "rb").read()).hexdigest() != h]
    if bad:
        sys.exit(f"CHECKSUM MISMATCH for {len(bad)} frames (e.g. {bad[:3]}). Do not continue; tell the main author.")
    print(f"OK: all {len(expected)} real frames present and byte-identical to the main machine.")


if __name__ == "__main__":
    main()
