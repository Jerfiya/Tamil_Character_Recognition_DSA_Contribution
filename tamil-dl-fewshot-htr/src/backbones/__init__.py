"""Backbone registry. Every backbone exposes ``features(x) -> [B, C, h, w]`` and ``feat_channels``;
``Embedder`` adds pooling and a linear projection to the embedding dimension."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

TIMM_NAMES = {"efficientnet_b0": "efficientnet_b0", "mobilenetv3_large": "mobilenetv3_large_100",
              "mobilenetv3_large_100": "mobilenetv3_large_100", "convnext_tiny": "convnext_tiny",
              "resnet50": "resnet50"}
SCRATCH = ("conv4", "resnet12", "resnet18", "vit_small")


class Embedder(nn.Module):
    """Backbone + (adaptive pool -> flatten -> linear) head producing a D-dimensional embedding.

    Spatial layout matters for characters (where a vowel sign sits), so small feature maps are
    pooled to ``pool`` x ``pool`` rather than to a single cell.
    """

    def __init__(self, body: nn.Module, emb_dim: int, pool: int = 2) -> None:
        super().__init__()
        self.body = body
        self.feat_channels = int(body.feat_channels)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.proj = nn.Linear(self.feat_channels * pool * pool, emb_dim)
        self.emb_dim = emb_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Last feature map [B, C, h, w]."""
        return self.body.features(x)

    def embed_from_features(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(fmap).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_from_features(self.features(x))


def build_backbone(mcfg: Mapping[str, Any]) -> Embedder:
    """Create an ``Embedder`` from the ``model`` section of the config."""
    name = mcfg["backbone"]
    emb = int(mcfg.get("emb_dim", 128))
    if name == "conv4":
        from .conv4 import Conv4
        return Embedder(Conv4(int(mcfg.get("hidden", 64))), emb, pool=3)
    if name == "resnet12":
        from .resnet12 import ResNet12
        return Embedder(ResNet12(int(mcfg.get("width", 16))), emb, pool=2)
    if name == "resnet18":
        from .resnet_scratch import ResNet18Gray
        return Embedder(ResNet18Gray(int(mcfg.get("width", 16))), emb, pool=2)
    if name == "vit_small":
        from .vit_small import ViTSmall
        return Embedder(ViTSmall(int(mcfg.get("input_size", 48))), emb, pool=2)
    if name in TIMM_NAMES:
        from .pretrained import TimmBackbone
        return Embedder(TimmBackbone(TIMM_NAMES[name], bool(mcfg.get("pretrained", True)),
                                     mcfg.get("gray_mode", "adapt")), emb, pool=1)
    raise ValueError(f"unknown backbone: {name}")
