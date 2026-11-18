"""ResNet-18 trained from scratch, adapted to 64x64 single-channel input.

Changes from the ImageNet layout: 3x3 stride-1 single-channel stem followed by one 2x
max-pool (instead of 7x7 stride-2 + max-pool), so a 64x64 letter keeps a 4x4 map at the end.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + idt)


class ResNet18Gray(nn.Module):
    """ResNet-18 ([2, 2, 2, 2] basic blocks) with configurable base ``width`` (64 = standard)."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        ch = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(1, ch[0], 3, 1, 1, bias=False), nn.BatchNorm2d(ch[0]),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        layers, cin = [], ch[0]
        for i, c in enumerate(ch):
            layers += [BasicBlock(cin, c, 1 if i == 0 else 2), BasicBlock(c, c, 1)]
            cin = c
        self.layers = nn.Sequential(*layers)
        self.feat_channels = ch[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
