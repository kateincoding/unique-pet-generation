import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FaceBackbone(nn.Module):
    """
    CNN feature extractor built from scratch.
    Input:  (B, 3, 224, 224)
    Output: (B, feature_dim)  — 512 by default
    """

    def __init__(self, feature_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            ConvBlock(3,   32),           # 224 → 112
            ConvBlock(32,  64),           # 112 →  56
            ConvBlock(64,  128),          #  56 →  28
            ConvBlock(128, 256),          #  28 →  14
            ConvBlock(256, feature_dim),  #  14 →   7
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.dropout(x)
