"""Siamese embedding trained with triplet (batch-hard) or contrastive loss; nearest-support decision."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from backbones import Embedder
from losses import batch_hard_triplet, contrastive_loss, sq_euclidean

from . import EpisodeInfo, FewShotMethod


class SiameseNet(FewShotMethod):
    """Embeddings are L2-normalised. An episode's images form the mining batch (N classes x
    (K+Q) images), so hard negatives are other letters in the same episode. At test time a
    query takes the label of its nearest support example."""
    name = "siamese"
    SCALE = 16.0   # fixed scale turning distances into logits for confidence/top-k

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.margin = float(mcfg.get("margin", 0.3))
        self.loss_kind = mcfg.get("siamese_loss", "triplet")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=1)

    def _loss(self, emb: torch.Tensor, y: torch.Tensor):
        if self.loss_kind == "contrastive":
            return contrastive_loss(emb, y, max(self.margin, 1.0))
        return batch_hard_triplet(emb, y, self.margin)

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        d = sq_euclidean(qf, sf).add(1e-12).sqrt().reshape(len(qf), n_way, k_shot)
        return -d.min(2).values * self.SCALE

    def train_loss(self, feats: torch.Tensor, info: EpisodeInfo) -> Dict[str, Any]:
        loss, comps = self._loss(feats, info.all_local)
        ns = info.n_way * info.k_shot
        with torch.no_grad():
            acc = float((self.episode_logits(feats[:ns], feats[ns:], info.n_way, info.k_shot).argmax(1) == info.query_y).float().mean())
        return {"loss": loss, "plain": float(loss.detach()), "acc": acc, **comps}

    def val_loss(self, sf, qf, query_y, n_way, k_shot, logits) -> float:
        y = torch.cat([torch.arange(n_way, device=sf.device).repeat_interleave(k_shot), query_y])
        return float(self._loss(torch.cat([sf, qf]), y)[0])
