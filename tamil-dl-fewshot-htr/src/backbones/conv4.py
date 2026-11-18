"""Conv-4: the classic few-shot backbone (4 x [conv3x3 - BN - ReLU - maxpool])."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class Conv4(nn.Module):
    """64x64 grey input -> [B, hidden, 4, 4]."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(conv_block(1, hidden), conv_block(hidden, hidden),
                                    conv_block(hidden, hidden), conv_block(hidden, hidden))
        self.feat_channels = hidden

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
