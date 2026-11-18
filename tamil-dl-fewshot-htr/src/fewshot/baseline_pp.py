"""Baseline++: cosine classifier. Train a cosine head on the training classes; at test time
freeze the backbone and fit a NEW cosine head on the K support examples of each episode."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import Embedder
from losses import smooth_ce

from . import EpisodeInfo, FewShotMethod


def finetune_cosine_head(sf: torch.Tensor, n_way: int, k_shot: int, steps: int, lr: float, scale: float) -> torch.Tensor:
    """Fit cosine heads on frozen support features, batched over episodes.

    ``sf`` is [E, N*K, D] (class-major). Heads start at the normalised class means and are
    refined with Adam for ``steps`` steps. Returns normalised weights [E, N, D].
    """
    sf = F.normalize(sf.detach().float(), dim=-1)
    e = sf.shape[0]
    y = torch.arange(n_way, device=sf.device).repeat_interleave(k_shot).unsqueeze(0).expand(e, -1)
    with torch.enable_grad():
        w = sf.reshape(e, n_way, k_shot, -1).mean(2).clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(int(steps)):
            logits = scale * torch.bmm(sf, F.normalize(w, dim=-1).transpose(1, 2))     # [E, NK, N]
            loss = F.cross_entropy(logits.reshape(-1, n_way), y.reshape(-1), reduction="sum") / (n_way * k_shot)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return F.normalize(w.detach(), dim=-1)


class BaselinePP(FewShotMethod):
    name = "baselinepp"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.s = float(mcfg.get("arc_s", 24.0))
        self.steps, self.ft_lr = int(mcfg.get("bpp_steps", 60)), float(mcfg.get("bpp_lr", 0.05))
        self.weight = nn.Parameter(torch.empty(max(n_train_classes, 1), backbone.emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        w = finetune_cosine_head(sf.unsqueeze(0), n_way, k_shot, self.steps, self.ft_lr, self.s)[0]
        return self.s * F.normalize(qf, dim=1) @ w.t()

    def train_loss(self, feats: torch.Tensor, info: EpisodeInfo) -> Dict[str, Any]:
        cos = F.normalize(feats, dim=1) @ F.normalize(self.weight, dim=1).t()
        y = info.all_global
        logits = cos * self.s
        plain = F.cross_entropy(logits, y)
        loss = smooth_ce(logits, y, info.smoothing) if info.smoothing > 0 else plain
        return {"loss": loss, "plain": float(plain.detach()), "acc": float((cos.argmax(1) == y).float().mean())}
