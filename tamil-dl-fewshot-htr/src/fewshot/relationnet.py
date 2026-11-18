"""Relation Network: a learned relation module scores concatenated support/query feature maps."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import Embedder

from . import FewShotMethod


class RelationNet(FewShotMethod):
    """Class feature map = mean of its support maps; relation(class map ++ query map) -> score.

    Scores are used as logits with cross-entropy (more stable than the original MSE and it
    keeps the loss comparable with the other methods); a learnable scale multiplies them.
    """
    name = "relationnet"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        c = backbone.feat_channels
        self.red = int(mcfg.get("relation_channels", 64))
        self.reduce = nn.Identity() if c == self.red else nn.Sequential(nn.Conv2d(c, self.red, 1, bias=False), nn.BatchNorm2d(self.red), nn.ReLU(inplace=True))
        self.relation = nn.Sequential(
            nn.Conv2d(2 * self.red, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.reduce(self.backbone.features(x))
        if f.shape[-1] > 4:
            f = F.adaptive_avg_pool2d(f, 4)
        return f

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        protos = sf.reshape(n_way, k_shot, *sf.shape[1:]).mean(1)            # [N, C, h, w]
        m = qf.shape[0]
        out = []
        chunk = max(1, 4096 // n_way)
        for i in range(0, m, chunk):
            q = qf[i:i + chunk]
            pair = torch.cat([protos.unsqueeze(0).expand(len(q), -1, -1, -1, -1),
                              q.unsqueeze(1).expand(-1, n_way, -1, -1, -1)], dim=2)
            out.append(self.relation(pair.flatten(0, 1)).view(len(q), n_way))
        return torch.cat(out) * self.log_scale.exp().clamp(max=100.0)

    def stats(self) -> Dict[str, float]:
        return {"temperature": float(self.log_scale.exp().detach())}
