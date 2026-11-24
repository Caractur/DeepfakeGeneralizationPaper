# Generating the fake images — instructions for co-authors

You are generating part of the deepfake images for the study. The work is split into **shards**: lists of
video frames. You run two generators on your own shard and send the results back to the main author, who merges
everything and runs all checks and training. Your part is about **12–14 hours of unattended GPU time** on a
laptop RTX 4070 (less on a faster card), plus about an hour of setup.

Your shards are listed in [ASSIGNMENTS.md](ASSIGNMENTS.md).

| Generator | What it does | Your shard | Time on an RTX 4070 laptop |
|---|---|---|---|
| **K22**: Kandinsky 2.2 inpainting | repaints the face in a real frame | ~1,351 frames | ~5.5 h (≈15 s/frame) |
| **REF**: REFace face swap | puts the face of video *s* into the scene of video *t* | ~1,414 frames | ~6.5 h (≈17 s/frame) |

Everything is pinned (model versions, prompts, seeds, settings in `config.yaml`), so do not change any setting.

---

## What you need

- **An NVIDIA GPU with at least 8 GB of memory** and a recent driver. Both generators were tested on 8 GB.
  - REFace needs PyTorch 1.13, which does **not** run on RTX 50-series (Blackwell) cards. If you have one, tell the
    main author: you will get only K22 work, with a newer PyTorch (see Troubleshooting).
- **About 50 GB of free disk** (models ~25 GB, two Python environments ~15 GB, data ~5 GB temporarily).
  If your system drive is small, set the environment variable `HF_HOME` to a folder on a bigger drive before
  starting; the models are downloaded there.
- **Miniconda** (https://docs.conda.io/en/latest/miniconda.html) and **git**.
- Windows or Linux. Keep the computer plugged in and set it to **never sleep** while generating.

## Rules

- The real frames come from FaceForensics++ and show real people. **Do not share or upload the frames, the masks
  or the generated images anywhere public.** Send your results only through a private link to the main author
  (see Step 5), and delete your copies once the main author confirms they arrived.
- Do not edit `config.yaml`, the shard files or the code. If something fails, see Troubleshooting or ask.

---

## Step 0 — Get the code and set up the environments

```bash
git clone https://github.com/Caractur/DeepfakeGeneralizationPaper.git
cd DeepfakeGeneralizationPaper/generation
```

Environment **dfgen** (masks, K22, quality check):

```bash
conda create -y -n dfgen python=3.10
conda activate dfgen
python env/install.py gen
```

The last line must print `OK  torch 2.6.0+cu124 | CUDA available: True | GPU: <your GPU>`.

Environment **dfref** (REFace), only if you have an REF shard:

```bash
conda create -y -n dfref -c conda-forge python=3.10 dlib=19.24
conda activate dfref
python env/install.py reface
python code/setup_reface.py
```

`install.py` must print `OK  torch 1.13.1+cu117 | CUDA available: True ...`; `setup_reface.py` downloads about
8 GB (broken connections are retried; if it stops anyway, run it again) and must end with `OK: REFace is set up.`

All commands below are run from the `generation` folder.

## Step 1 — Download the real frames (dfgen)

```bash
conda activate dfgen
python code/fetch_data.py
```

Downloads the FaceForensics++ frames from Kaggle (~4.1 GB, resumable), keeps the 5,000 real frames and checks
each one against the main author's copy. It must end with `OK: all 5000 real frames present and byte-identical`.
If the download fails, see Troubleshooting.

## Step 2 — Compute the face masks (dfgen)

```bash
python code/masks.py
```

About 5–10 minutes on the CPU. It ends with a comparison line like
`same face found/not found 100.00%, identical face box 100.00%, identical mask pixels 100.00%` and `OK`.
Copy that line into your message to the main author. If it prints a WARNING instead, stop and send the line.

## Step 2b — Download the models (dfgen)

```bash
python code/download_models.py
```

About 14 GB (Kandinsky 2.2 and CLIP, at the exact versions in `config.yaml`). Broken connections are retried
automatically; if it still stops, run it again and it continues. It must end with `OK: all models downloaded.`

## Step 3 — Generate K22 (dfgen)

First a 2-frame test:

```bash
python code/generate.py --gen K22 --shard shards/K22_shardN.txt --limit 2
```

(replace `N` with your shard number). If it ends with `=== K22 done`, start the full run:

```bash
python code/generate.py --gen K22 --shard shards/K22_shardN.txt
```

It prints progress and an ETA every 25 frames. **If it stops for any reason, run the same command again: it
continues where it stopped.** The 2 test frames are kept and not regenerated.

## Step 4 — Generate REF (dfref)

```bash
conda activate dfref
python code/generate_ref.py --shard shards/REF_shardN.txt --limit 2
python code/generate_ref.py --shard shards/REF_shardN.txt
```

Same behaviour: resumable, progress every 25 frames, ends with `=== REF done`. A few frames may be logged as
`align_fail` (REFace's face aligner found no face); that is expected and handled.

## Step 5 — Check, pack and send

```bash
conda activate dfgen
python code/check_shard.py --gen K22 --shard shards/K22_shardN.txt
python code/check_shard.py --gen REF --shard shards/REF_shardN.txt
```

Each command checks that every frame of the shard is done and every image is valid, prints a summary, and writes a
zip to `work/return/` (about 200–400 MB each). It prints a `SHA-256` line for each zip.

Send to the main author:
1. the two zips, through a **private** shared folder (OneDrive / Google Drive / Dropbox shared only with the main
   author, or your university's file-transfer service) — not email, not a public link;
2. the two `SHA-256` lines and the mask comparison line from Step 2.

When the main author confirms the merge worked, delete `work/` (or at least `work/real_frames`, `work/masks`,
`work/out`, `work/return`).

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `CUDA available: False` | Update the NVIDIA driver; check `nvidia-smi` works. |
| RTX 50-series card | In dfgen: `pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128` and tell the main author (it is recorded in your shard's machine info). REF is not possible on these cards. |
| Out of GPU memory | Close other programs that use the GPU (games, browsers with hardware acceleration) and rerun; it resumes. |
| Kaggle download fails | Sign in to kaggle.com, download the zip from https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23 in a browser, then `python code/fetch_data.py --zip <path to the zip>`. |
| `CHECKSUM MISMATCH` | Delete `work/real_frames` and rerun Step 1. If it happens again, tell the main author. |
| Mask comparison prints WARNING | Stop and send the printed line to the main author. |
| `check_shard` lists PROBLEMS | Rerun the generate command for that shard (it fills in what is missing), then check again. |
| Anything else | Send the last ~30 lines of the output and the file `work/logs/gen_K22.log` or `gen_REF.log`. |

## What gets sent back, exactly

For each frame only the region that was changed: an RGBA PNG where the alpha channel marks the changed pixels,
plus its position in `_gen_log.csv`. The main author pastes these pixels onto their own copy of the real frame,
so every image in the dataset has exactly the same untouched background as its real frame, whichever computer
made the face. Your zip also contains `_machine.json` (GPU, driver-side CUDA version, library versions), which
the paper reports, and your `_geometry.csv` (face boxes only) for the mask comparison.
