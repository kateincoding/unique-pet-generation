"""Prediction heads: embedding projection and attribute classifier."""

import torch.nn as nn


class EmbeddingHead(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class AttributeHead(nn.Module):
    def __init__(self, embedding_dim: int, num_attributes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_attributes)

    def forward(self, x):
        return self.fc(x)
