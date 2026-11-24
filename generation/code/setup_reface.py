"""Step 0b - Set up REFace (only for people with an REF shard). Run in the "dfref" environment.

  1. git clone https://github.com/Sanoojan/REFace into work/REFace, checked out at commit f73bd8d (unmodified)
  2. Download REFace's weights (Hugging Face Sanoojan/REFace, revision 514ebf4, ~6.5 GB) into work/REFace_weights
     and copy its "Other_dependencies" (face parsing, ArcFace, dlib landmarks, ...) into work/REFace
  3. Download CLIP ViT-L/14 (revision 32bd642) as pytorch_model.bin into work/clip-vit-large-patch14
     (REFace's old transformers version cannot read the newer safetensors file)
  4. pip install -e work/REFace --no-deps
Resumable: finished steps are skipped.
"""
import os, sys, shutil, subprocess
from common import WORK

REPO, COMMIT = "https://github.com/Sanoojan/REFace.git", "f73bd8d4ad5a2ca362d294982e6f381d83e58e62"
WEIGHTS_REV = "514ebf4560c1b8cc68eda6b58043e656ac7feb55"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
REFACE = os.path.join(WORK, "REFace")
WEIGHTS = os.path.join(WORK, "REFace_weights")
CLIP = os.path.join(WORK, "clip-vit-large-patch14")


def run(*cmd, cwd=None):
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(list(cmd), cwd=cwd)


def main():
    from huggingface_hub import snapshot_download
    if not os.path.isdir(os.path.join(REFACE, ".git")):
        run("git", "clone", REPO, REFACE)
    run("git", "checkout", "-q", COMMIT, cwd=REFACE)
    print("Downloading REFace weights (~6.5 GB) ...", flush=True)
    snapshot_download("Sanoojan/REFace", revision=WEIGHTS_REV, local_dir=WEIGHTS)
    src = os.path.join(WEIGHTS, "Other_dependencies")
    for sub in os.listdir(src):
        os.makedirs(os.path.join(REFACE, "Other_dependencies", sub), exist_ok=True)
        for f in os.listdir(os.path.join(src, sub)):
            dst = os.path.join(REFACE, "Other_dependencies", sub, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src, sub, f), dst)
    print("Downloading CLIP ViT-L/14 ...", flush=True)
    snapshot_download("openai/clip-vit-large-patch14", revision=CLIP_REV,
                      allow_patterns=["*.json", "*.txt", "pytorch_model.bin"], local_dir=CLIP)
    run(sys.executable, "-m", "pip", "install", "-q", "-e", REFACE, "--no-deps")
    for f in ["last.ckpt"]:
        assert os.path.exists(os.path.join(WEIGHTS, f)), f"missing {f}"
    assert os.path.exists(os.path.join(REFACE, "Other_dependencies", "face_parsing", "79999_iter.pth"))
    assert os.path.exists(os.path.join(REFACE, "Other_dependencies", "DLIB_landmark_det", "shape_predictor_68_face_landmarks.dat"))
    assert os.path.exists(os.path.join(CLIP, "pytorch_model.bin"))
    print("OK: REFace is set up.")


if __name__ == "__main__":
    main()
