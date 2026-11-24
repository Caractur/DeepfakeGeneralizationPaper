"""Automatic quality check for generated faces (used by 11_generate.py to retry failed generations).

A generated face passes when, on the face crop (shared box):
  1. MediaPipe FaceDetection finds a face with score >= DET_MIN, and
  2. CLIP ViT-L/14 zero-shot gives P("a photo of a person's face") >= CLIP_MIN against prompts describing
     the failure modes seen in the pilot (face covered by a mask / net / mesh pattern, text, back of head).
Failed outputs are regenerated with a new seed (seed + 1,000,003 x attempt), up to max_attempts.
"""
import numpy as np, torch, cv2
from common import load_config, patch_protobuf_for_mediapipe

patch_protobuf_for_mediapipe()
import mediapipe as mp

PROMPTS = ["a photo of a person's face",
           "a face covered by a face mask",
           "a face covered by a net or a mesh pattern",
           "an abstract pattern or texture",
           "text on a screen",
           "the back of a person's head"]
DET_MIN, CLIP_MIN, DET_STRONG = 0.5, 0.5, 0.7


class QC:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        c = load_config()["detectors"]["clip"]
        self.model = CLIPModel.from_pretrained(c["model"], revision=c["revision"], torch_dtype=torch.float16).cuda().eval()
        self.proc = CLIPProcessor.from_pretrained(c["model"], revision=c["revision"])
        with torch.no_grad():
            t = self.proc(text=PROMPTS, return_tensors="pt", padding=True).to("cuda")
            e = self.model.get_text_features(**t)
            self.text = e / e.norm(dim=-1, keepdim=True)
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)

    @torch.no_grad()
    def score(self, rgb_crop):
        r = self.det.process(rgb_crop)
        det = max((d.score[0] for d in r.detections), default=0.0) if r.detections else 0.0
        x = self.proc(images=rgb_crop, return_tensors="pt")["pixel_values"].to("cuda").half()
        e = self.model.get_image_features(pixel_values=x)
        e = e / e.norm(dim=-1, keepdim=True)
        p = (100 * e @ self.text.T).softmax(-1)[0].float().cpu().numpy()
        # reject: no confident face, or CLIP prefers a failure description AND the face score is only moderate
        return dict(det=float(det), p_face=float(p[0]), ok=bool(det >= DET_MIN and (p[0] >= CLIP_MIN or det >= DET_STRONG)))
