"""
Main inference module.

Usage:
    from inference import FaceAttributeExtractor

    extractor = FaceAttributeExtractor(checkpoint_path="checkpoints/best.pt")
    attrs = extractor.extract("photo.jpg")
    print(attrs)
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from models.backbone import FaceBackbone
from models.heads import build_heads, CLASS_LABELS
from utils.color_analysis import analyze_skin_tone, analyze_eye_color

try:
    from facenet_pytorch import MTCNN as _MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FaceAttributeExtractor:
    """
    End-to-end face attribute extractor.
    Combines CNN predictions with HSV/LAB color analysis.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = FaceBackbone().to(self.device)
        self.heads    = build_heads().to(self.device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.backbone.load_state_dict(state["backbone"])
            self.heads.load_state_dict(state["heads"])

        self.backbone.eval()
        self.heads.eval()

        self.detector = _MTCNN(keep_all=False, device=self.device) if _MTCNN_AVAILABLE else None

    # ── face detection ────────────────────────────────────────────────────────

    def _crop_face(self, img_pil: Image.Image) -> Image.Image:
        if self.detector is None:
            return img_pil
        boxes, _ = self.detector.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            return img_pil
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        mx = int((x2 - x1) * 0.10)
        my = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - mx);  y1 = max(0, y1 - my)
        x2 = min(img_pil.width, x2 + mx)
        y2 = min(img_pil.height, y2 + my)
        return img_pil.crop((x1, y1, x2, y2))

    # ── main inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, image_path: str) -> dict:
        """
        Run full attribute extraction on an image file.
        Returns a dict with all facial attributes.
        """
        img_pil  = Image.open(image_path).convert("RGB")
        face_pil = self._crop_face(img_pil)

        # CNN predictions
        tensor   = _PREPROCESS(face_pil).unsqueeze(0).to(self.device)
        features = self.backbone(tensor)

        result: dict = {}
        for attr_name, head in self.heads.items():
            pred_idx       = head(features).argmax(dim=1).item()
            result[attr_name] = CLASS_LABELS[attr_name][pred_idx]

        # Color analysis enrichment (eye color + skin tone cross-check)
        face_bgr = cv2.cvtColor(
            np.array(face_pil.resize((224, 224))), cv2.COLOR_RGB2BGR
        )
        result["color_ojos"]  = analyze_eye_color(face_bgr)
        result["tono_piel_lab"] = analyze_skin_tone(face_bgr)

        return result


def extract_face_attributes(image_path: str, checkpoint_path: str | None = None) -> dict:
    """Convenience wrapper — creates a new extractor on every call."""
    return FaceAttributeExtractor(checkpoint_path=checkpoint_path).extract(image_path)
