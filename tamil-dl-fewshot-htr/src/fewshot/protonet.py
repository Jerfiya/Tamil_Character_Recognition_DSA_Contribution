"""Prototypical Network (main method)."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from backbones import Embedder
from losses import proto_logits

from . import FewShotMethod


class ProtoNet(FewShotMethod):
    """prototype = mean support embedding; logits = -squared Euclidean distance x temperature.

    The temperature is learnable (stored as its log). Distances are divided by the embedding
    dimension so the initial logits have a sensible scale for any ``emb_dim``. With
    ``distance: cosine`` the logits are cosine similarities x temperature.
    """
    name = "protonet"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.distance = mcfg.get("distance", "euclidean")
        t0 = float(mcfg.get("init_temperature", 1.0)) * (10.0 if self.distance == "cosine" else 1.0)
        self.log_temp = nn.Parameter(torch.tensor(math.log(t0)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(max=100.0)

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        raw = proto_logits(sf, qf, n_way, k_shot, self.distance)
        if self.distance != "cosine":
            raw = raw / sf.shape[1] * 16.0
        return raw * self.temperature

    def stats(self) -> Dict[str, float]:
        return {"temperature": float(self.temperature.detach())}
