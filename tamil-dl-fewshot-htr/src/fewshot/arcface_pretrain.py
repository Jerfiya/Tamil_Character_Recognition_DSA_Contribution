"""ArcFace / CosFace margin-head pretraining on the training classes.

The backbone learned here is then fine-tuned episodically (``method.init_from`` in a ProtoNet
run). Used directly for few-shot, a query is assigned to the nearest cosine prototype.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import Embedder
from losses import margin_logits, proto_logits

from . import EpisodeInfo, FewShotMethod


class ArcFaceModel(FewShotMethod):
    name = "arcface"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.s, self.m = float(mcfg.get("arc_s", 24.0)), float(mcfg.get("arc_m", 0.35))
        self.kind = mcfg.get("arc_type", "arcface")
        self.weight = nn.Parameter(torch.empty(max(n_train_classes, 1), backbone.emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        return proto_logits(sf, qf, n_way, k_shot, "cosine") * self.s

    def train_loss(self, feats: torch.Tensor, info: EpisodeInfo) -> Dict[str, Any]:
        cos = F.normalize(feats, dim=1) @ F.normalize(self.weight, dim=1).t()
        y = info.all_global
        loss = F.cross_entropy(margin_logits(cos, y, self.s, self.m, self.kind), y)
        with torch.no_grad():
            plain = F.cross_entropy(cos * self.s, y)
            onehot = F.one_hot(y, cos.shape[1]).bool()
            cos_t = cos[onehot]
            cos_n = cos.masked_fill(onehot, -2.0).max(1).values
        # margin statistics are reported in the pos/neg columns: target cosine vs hardest non-target cosine
        return {"loss": loss, "plain": float(plain), "acc": float((cos.argmax(1) == y).float().mean()),
                "loss_pos": float((1 - cos_t).mean()), "loss_neg": float(F.relu(cos_n).mean()),
                "dist_pos": float(cos_t.mean()), "dist_neg": float(cos_n.mean())}
