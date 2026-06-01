"""Full facial feature extraction model."""

import torch
import torch.nn as nn

from pet_gen.models.backbone import create_backbone
from pet_gen.models.heads import AttributeHead, EmbeddingHead


class FacialFeatureModel(nn.Module):
    """Backbone -> embedding bottleneck -> attribute prediction.

    Returns both the embedding (for pet generation conditioning)
    and attribute logits (for training supervision).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        num_attributes: int = 15,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone, feature_dim = create_backbone(backbone_name, pretrained)
        self.embedding_head = EmbeddingHead(feature_dim, embedding_dim, dropout)
        self.attribute_head = AttributeHead(embedding_dim, num_attributes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        logits = self.attribute_head(embedding)
        return {"embedding": embedding, "logits": logits}

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        """Return parameter groups with lower LR for backbone after unfreezing."""
        return [
            {"params": self.backbone.parameters(), "lr": lr * backbone_lr_scale},
            {"params": self.embedding_head.parameters(), "lr": lr},
            {"params": self.attribute_head.parameters(), "lr": lr},
        ]
