"""Backbone feature extractors."""

import torch.nn as nn
import torchvision.models as models


def create_backbone(name: str = "resnet18", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Create a backbone and return (backbone, feature_dim).

    The final classification layer is removed. Returns the trunk
    that outputs a flat feature vector.
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unknown backbone: {name}")
