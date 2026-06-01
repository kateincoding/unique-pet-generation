"""Tests for model architecture: shapes, forward pass, gradient flow."""

import torch

from pet_gen.models.backbone import create_backbone
from pet_gen.models.feature_model import FacialFeatureModel


def test_resnet18_backbone():
    backbone, dim = create_backbone("resnet18", pretrained=False)
    assert dim == 512
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, 512)


def test_mobilenet_backbone():
    backbone, dim = create_backbone("mobilenet_v2", pretrained=False)
    assert dim == 1280
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, 1280)


def test_facial_feature_model_forward():
    model = FacialFeatureModel(
        backbone_name="resnet18",
        pretrained=False,
        embedding_dim=256,
        num_attributes=15,
        dropout=0.0,
    )
    x = torch.randn(4, 3, 224, 224)
    output = model(x)

    assert "embedding" in output
    assert "logits" in output
    assert output["embedding"].shape == (4, 256)
    assert output["logits"].shape == (4, 15)


def test_gradient_flow():
    model = FacialFeatureModel(
        backbone_name="resnet18", pretrained=False, embedding_dim=128, num_attributes=5
    )
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    loss = output["logits"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_freeze_unfreeze():
    model = FacialFeatureModel(
        backbone_name="resnet18", pretrained=False, embedding_dim=128, num_attributes=5
    )

    model.freeze_backbone()
    for param in model.backbone.parameters():
        assert not param.requires_grad

    # Head params still trainable
    for param in model.embedding_head.parameters():
        assert param.requires_grad

    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad


def test_param_groups():
    model = FacialFeatureModel(
        backbone_name="resnet18", pretrained=False, embedding_dim=128, num_attributes=5
    )
    groups = model.get_param_groups(lr=1e-3, backbone_lr_scale=0.1)
    assert len(groups) == 3
    assert groups[0]["lr"] == 1e-4  # backbone
    assert groups[1]["lr"] == 1e-3  # embedding
    assert groups[2]["lr"] == 1e-3  # attribute head
