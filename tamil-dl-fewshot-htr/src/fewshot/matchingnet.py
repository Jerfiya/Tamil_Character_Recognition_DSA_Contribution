"""Matching Network: cosine attention over the support set (optional full-context embeddings)."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import Embedder

from . import FewShotMethod


class MatchingNet(FewShotMethod):
    """p(y | q) = sum over support of softmax(scale * cos(q, s)) for supports of class y.

    ``fce: true`` adds a bidirectional LSTM over the support embeddings (context for g) with a
    skip connection; queries keep their plain embedding.
    """
    name = "matchingnet"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.log_scale = nn.Parameter(torch.tensor(2.5))
        self.fce = bool(mcfg.get("fce", False))
        if self.fce:
            d = backbone.emb_dim
            self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        if self.fce:
            sf = sf + self.lstm(sf.unsqueeze(0))[0].squeeze(0)
        sim = F.normalize(qf, dim=1) @ F.normalize(sf, dim=1).t()            # [M, N*K]
        att = F.softmax(sim * self.log_scale.exp().clamp(max=100.0), dim=1)
        probs = att.reshape(len(qf), n_way, k_shot).sum(2)
        return torch.log(probs + 1e-8)                                       # log-probs act as logits

    def stats(self) -> Dict[str, float]:
        return {"temperature": float(self.log_scale.exp().detach())}
