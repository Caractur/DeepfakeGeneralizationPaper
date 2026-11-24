# Deepfake Detector Generalization Under Generative Shift

Code for the study of how deepfake detectors trained on one family of generators (FaceForensics++ face swaps)
transfer to diffusion-based fakes, with every real/fake processing difference controlled.

**Co-authors generating images: start at [generation/README.md](generation/README.md)** and check
[generation/ASSIGNMENTS.md](generation/ASSIGNMENTS.md) for your shards.

What is in this repository so far:

| Folder | Contents |
|---|---|
| `generation/` | everything needed to generate the Kandinsky 2.2 and REFace fakes on another machine: code, pinned settings (`config.yaml`), the list of frame units, shard lists, environment files, checksums |

No images, face masks or FaceForensics++ data are stored here. The real frames are downloaded from their public
source by each user and verified against checksums; generated images are exchanged privately.
