"""Loss functions for multi-label attribute prediction."""

import torch
import torch.nn as nn


def create_bce_loss(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """Create BCE loss with optional class-imbalance weighting."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
