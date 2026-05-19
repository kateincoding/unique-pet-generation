"""Run inference on a single photo: detect face, extract embedding + attributes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import torch

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


def main():
    parser = argparse.ArgumentParser(description="Extract facial features from a photo")
    parser.add_argument("image_path", help="Path to input photo")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Model checkpoint")
    args = parser.parse_args()

    device = get_device()

    # Load model
    model = FacialFeatureModel(num_attributes=len(ATTRIBUTE_NAMES))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess
    preprocessor = FacePreprocessor(device=device)
    tensor = preprocessor.preprocess_file(args.image_path)
    if tensor is None:
        print("No face detected in image.")
        sys.exit(1)

    # Inference
    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(device))

    import numpy as np

    embedding = output["embedding"][0].cpu().numpy()
    logits = output["logits"][0].cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print("\nPredicted attributes:")
    for name, prob in zip(ATTRIBUTE_NAMES, probs):
        marker = "+" if prob >= 0.5 else " "
        print(f"  [{marker}] {name}: {prob:.3f}")


if __name__ == "__main__":
    main()
