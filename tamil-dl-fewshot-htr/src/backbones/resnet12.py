"""ResNet-12 (four residual blocks of three 3x3 convs, max-pool after each block)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.body(x) + self.short(x)))


class ResNet12(nn.Module):
    """``width`` is the first block's channel count (64 in the literature; 32 here for CPU)."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        ch = [width, width * 2, width * 4, width * 8]
        self.layers = nn.Sequential(Block(1, ch[0]), Block(ch[0], ch[1]), Block(ch[1], ch[2]), Block(ch[2], ch[3]))
        self.feat_channels = ch[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
