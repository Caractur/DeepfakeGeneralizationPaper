"""Step 2b - Download the models for K22 generation and the quality check (~17 GB, dfgen environment).

Kandinsky 2.2 inpainting decoder + prior and CLIP ViT-L/14, at the exact revisions in config.yaml, into the
Hugging Face cache (set HF_HOME first to put it on another drive). Interrupted downloads resume automatically;
if the script stops anyway, just run it again.
"""
from common import load_config, hf_download

C = load_config()
k = C["generators"]["K22"]
FILES = ["*.json", "*.txt", "*.safetensors"]        # what diffusers/transformers load (not the duplicate .bin)
jobs = [(k["model"], k["revision"]), (k["prior"], k["prior_revision"]),
        (C["detectors"]["clip"]["model"], C["detectors"]["clip"]["revision"])]
for repo, rev in jobs:
    print(f"Downloading {repo} @ {rev[:7]} ...", flush=True)
    hf_download(repo, rev, allow_patterns=FILES)
print("OK: all models downloaded.")
