"""Plain supervised cross-entropy classifier: the non-few-shot reference (Protocol A only).

It sees every class during training and predicts with its linear head. Its embedding is also
scored in few-shot episodes (nearest prototype) so it can sit in the same tables.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import Embedder
from fewshot import EpisodeInfo, FewShotMethod
from losses import proto_logits, smooth_ce


class SupervisedClassifier(FewShotMethod):
    name = "supervised"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.head = nn.Linear(backbone.emb_dim, max(n_train_classes, 1))

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        return proto_logits(sf, qf, n_way, k_shot, "euclidean") / sf.shape[1] * 16.0

    def train_loss(self, feats: torch.Tensor, info: EpisodeInfo) -> Dict[str, Any]:
        logits = self.head(feats)
        y = info.all_global
        plain = F.cross_entropy(logits, y)
        loss = smooth_ce(logits, y, info.smoothing) if info.smoothing > 0 else plain
        return {"loss": loss, "plain": float(plain.detach()), "acc": float((logits.argmax(1) == y).float().mean())}

    def pool_val(self, feats: torch.Tensor, y_global: torch.Tensor) -> Optional[Dict[str, float]]:
        logits = self.head(feats)
        top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
        return {"loss": float(F.cross_entropy(logits, y_global)),
                "acc": float((logits.argmax(1) == y_global).float().mean()),
                "top3": float((top3 == y_global.unsqueeze(1)).any(1).float().mean())}
