"""Run inference: webcam capture or image file -> face embedding + attributes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from pet_gen.data.preprocessing import FacePreprocessor
from pet_gen.models.feature_model import FacialFeatureModel
from pet_gen.training.trainer import get_device

ATTRIBUTE_NAMES = [
    "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair",
    "Straight_Hair", "Wavy_Hair",
    "Arched_Eyebrows", "Bushy_Eyebrows",
    "Narrow_Eyes", "Big_Nose", "Pointy_Nose",
    "High_Cheekbones", "Oval_Face", "Pale_Skin", "Chubby",
]


def capture_from_camera() -> np.ndarray | None:
    """Open webcam, show preview, capture on SPACE, quit on Q."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return None

    print("Camera open. Press SPACE to capture, Q to quit.")
    captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture - SPACE to snap, Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            captured = frame.copy()
            print("Photo captured!")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def predict(model, preprocessor, image: np.ndarray | str, device: torch.device):
    """Run prediction on image (numpy array or file path)."""
    if isinstance(image, str):
        tensor = preprocessor.preprocess_file(image)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = preprocessor.detect_and_align(pil_image)

    if tensor is None:
        print("No face detected.")
        return None

    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(device))

    embedding = output["embedding"][0].cpu().numpy()
    logits = output["logits"][0].cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    return {"embedding": embedding, "probs": probs}


def print_results(results: dict):
    print(f"\nEmbedding shape: {results['embedding'].shape}")
    print(f"Embedding norm: {np.linalg.norm(results['embedding']):.4f}")
    print("\nPredicted attributes:")
    for name, prob in zip(ATTRIBUTE_NAMES, results["probs"]):
        marker = "+" if prob >= 0.5 else " "
        print(f"  [{marker}] {name}: {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Extract facial features")
    parser.add_argument("image_path", nargs="?", default=None,
                        help="Path to image (omit to use webcam)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--camera", action="store_true",
                        help="Force webcam capture")
    args = parser.parse_args()

    device = get_device()

    model = FacialFeatureModel(num_attributes=len(ATTRIBUTE_NAMES))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preprocessor = FacePreprocessor(device=device)

    if args.image_path and not args.camera:
        results = predict(model, preprocessor, args.image_path, device)
    else:
        frame = capture_from_camera()
        if frame is None:
            sys.exit(1)
        results = predict(model, preprocessor, frame, device)

    if results is None:
        sys.exit(1)

    print_results(results)


if __name__ == "__main__":
    main()
