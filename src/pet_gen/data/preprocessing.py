"""Face detection and alignment for inference on arbitrary photos."""

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from pet_gen.data.transforms import get_eval_transform


class FacePreprocessor:
    """Detect and align faces using MTCNN, then apply eval transforms.

    Used at inference time when users submit their own photos
    (not needed for CelebA training since images are pre-aligned).
    """

    def __init__(self, image_size: int = 224, margin: int = 40, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.mtcnn = MTCNN(
            image_size=image_size + margin,
            margin=margin,
            keep_all=False,
            select_largest=True,
            device=self.device,
        )
        self.transform = get_eval_transform(image_size)
        self.image_size = image_size

    def detect_and_align(self, image: Image.Image | np.ndarray) -> torch.Tensor | None:
        """Detect face, crop, align, and return preprocessed tensor.

        Returns None if no face detected.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        face = self.mtcnn(image)
        if face is None:
            return None

        # MTCNN returns tensor in [-1, 1], convert to [0, 255] uint8 for albumentations
        face_np = ((face.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        transformed = self.transform(image=face_np)
        return transformed["image"]

    def preprocess_file(self, path: str) -> torch.Tensor | None:
        """Load image from path and preprocess."""
        image = Image.open(path).convert("RGB")
        return self.detect_and_align(image)
