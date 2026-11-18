"""A small Vision Transformer trained from scratch (optional backbone)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTSmall(nn.Module):
    """Patch-8 ViT; tokens are reshaped to a [B, C, h, w] map so it plugs into ``Embedder``."""

    def __init__(self, img_size: int = 64, patch: int = 8, dim: int = 128, depth: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.grid = img_size // patch
        self.patch = nn.Conv2d(1, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.feat_channels = dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch(x).flatten(2).transpose(1, 2)
        if t.shape[1] != self.pos.shape[1]:
            raise ValueError("ViTSmall was built for a different input size")
        t = self.norm(self.encoder(t + self.pos))
        return t.transpose(1, 2).reshape(x.shape[0], -1, self.grid, self.grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
