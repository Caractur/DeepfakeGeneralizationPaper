"""Paths and helpers for the shared generation package.

Folder layout (all relative to this file):
  generation/config.yaml        generation settings (identical to the main project's)
  generation/data/              unit list and checksums (in git)
  generation/shards/            which frames each person generates (in git)
  generation/work/              everything downloaded or produced locally (NOT in git):
      real_frames/              the 5,000 FaceForensics++ real frames (fetch_data.py)
      masks/native/             face masks + face boxes (masks.py)
      out/<GEN>/                your generated face regions + _gen_log.csv
      return/                   the zip you send back (check_shard.py)
      REFace/, REFace_weights/, clip-vit-large-patch14/   (setup_reface.py)
"""
import os, json, platform
import yaml

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_ROOT = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(GEN_ROOT, "data")
SHARDS = os.path.join(GEN_ROOT, "shards")
WORK = os.path.join(GEN_ROOT, "work")
SRC_REAL_FRAMES = os.path.join(WORK, "real_frames")
MASKS = os.path.join(WORK, "masks", "native")
OUT = os.path.join(WORK, "out")
RETURN = os.path.join(WORK, "return")
LOGS = os.path.join(WORK, "logs")
for d in (WORK, OUT, RETURN, LOGS):
    os.makedirs(d, exist_ok=True)


def load_config():
    with open(os.path.join(GEN_ROOT, "config.yaml"), encoding="utf8") as f:
        return yaml.safe_load(f)


def real_name(s, k):
    return f"{s:03d}_f{k}.jpg"


def pair_name(t, s, k):
    """FaceForensics++ naming: <target scene>_<source face>_f<k>."""
    return f"{t:03d}_{s:03d}_f{k}.jpg"


def patch_protobuf_for_mediapipe():
    """mediapipe 0.10.9 calls SymbolDatabase.GetPrototype, removed in newer protobuf. Patch at runtime only."""
    from google.protobuf import symbol_database, message_factory
    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        symbol_database.SymbolDatabase.GetPrototype = lambda self, d: message_factory.GetMessageClass(d)


def log_to(name):
    fh = open(os.path.join(LOGS, f"{name}.log"), "a", encoding="utf8")

    def log(s):
        print(s, flush=True)
        fh.write(str(s) + "\n"); fh.flush()
    return log


def machine_info():
    """Recorded with every shard so the paper can state which hardware/software made which images."""
    info = dict(os=platform.platform(), python=platform.python_version())
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:
        info["torch"] = f"unavailable: {e}"
    for m in ("diffusers", "transformers", "mediapipe", "cv2", "numpy", "PIL"):
        try:
            info[m] = __import__(m).__version__
        except Exception:
            pass
    return info


def read_shard(path):
    """A shard file lists one frame name per line (comments start with #)."""
    with open(path, encoding="utf8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def hf_download(repo_id, revision, retries=30, **kw):
    """snapshot_download that survives broken connections: partial files are resumed on each retry."""
    import time
    from huggingface_hub import snapshot_download
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(repo_id, revision=revision, **kw)
        except Exception as e:
            print(f"  download interrupted ({type(e).__name__}); retry {attempt}/{retries} in 15 s ...", flush=True)
            time.sleep(15)
    raise RuntimeError(f"could not download {repo_id} after {retries} attempts; check the internet connection and rerun")
