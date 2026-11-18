"""ImageNet-pretrained backbones from timm (EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny, ResNet-50).

Grey input is handled either by adapting the stem to one channel (timm sums the RGB filters)
or by replicating the grey image to three channels (``gray_mode``).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, gray_mode: str = "adapt") -> None:
        super().__init__()
        import timm
        self.replicate = gray_mode == "replicate"
        kw = dict(num_classes=0, global_pool="", in_chans=3 if self.replicate else 1)
        try:
            self.net = timm.create_model(name, pretrained=pretrained, **kw)
            self.pretrained_loaded = bool(pretrained)
        except Exception as exc:  # offline or hub failure: keep going from scratch, loudly
            log.warning("could not load pretrained weights for %s (%s); using random init", name, exc)
            self.net = timm.create_model(name, pretrained=False, **kw)
            self.pretrained_loaded = False
        self.feat_channels = int(self.net.num_features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.replicate:
            x = x.expand(-1, 3, -1, -1)
        return self.net.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
