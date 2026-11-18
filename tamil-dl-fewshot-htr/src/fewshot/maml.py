"""First-order MAML (optional method).

The inner loop adapts the backbone and a per-episode linear head on the support set; the
outer update uses the query-loss gradient at the adapted weights (first-order: inner
gradients are detached). The head starts at zero for every episode, so the model works for
any N-way without a fixed-size output layer.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from backbones import Embedder

from . import EpisodeInfo, FewShotMethod


class FOMAML(FewShotMethod):
    name = "maml"
    cacheable = False
    needs_images = True
    SCALE = 10.0

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__(backbone, mcfg, n_train_classes)
        self.inner_lr = float(mcfg.get("maml_inner_lr", 0.05))
        self.steps = int(mcfg.get("maml_inner_steps", 3))
        self.eval_steps = int(mcfg.get("maml_eval_steps", 5))
        # MAML normalises with the statistics of the current batch (support or query), as in Finn et al.:
        # running statistics accumulated under ever-changing fast weights made eval-mode BN collapse
        # validation to chance (11% on 10-way vs 37-42% with batch statistics; see decisions.md).
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = False
                m.register_buffer("running_mean", None)
                m.register_buffer("running_var", None)
                m.register_buffer("num_batches_tracked", None)

    def _embed(self, params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        return F.normalize(functional_call(self.backbone, params, (x,)), dim=1)

    def adapt_logits(self, xs: torch.Tensor, xq: torch.Tensor, n_way: int, k_shot: int, steps: int) -> torch.Tensor:
        """Adapt on the support images, return query logits [M, N]."""
        ys = torch.arange(n_way, device=xs.device).repeat_interleave(k_shot)
        with torch.enable_grad():
            # EMA copies (used for validation) have requires_grad=False: give them a differentiable leaf.
            fast = {n: (p if p.requires_grad else p.detach().requires_grad_(True))
                    for n, p in self.backbone.named_parameters()}
            head = torch.zeros(n_way, self.backbone.emb_dim, device=xs.device, requires_grad=True)
            for _ in range(steps):
                loss = F.cross_entropy(self.SCALE * self._embed(fast, xs) @ head.t(), ys)
                names = list(fast)
                grads = torch.autograd.grad(loss, [fast[n] for n in names] + [head], allow_unused=True)
                fast = {n: (fast[n] if g is None else fast[n] - self.inner_lr * g.detach())
                        for n, g in zip(names, grads[:-1])}
                head = head - 10.0 * self.inner_lr * grads[-1].detach()
            return self.SCALE * self._embed(fast, xq) @ head.t()

    def episode_logits_images(self, xs: torch.Tensor, xq: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        return self.adapt_logits(xs, xq, n_way, k_shot, self.eval_steps if not self.training else self.steps)

    def train_loss_images(self, x: torch.Tensor, info: EpisodeInfo) -> Dict[str, Any]:
        ns = info.n_way * info.k_shot
        logits = self.adapt_logits(x[:ns], x[ns:], info.n_way, info.k_shot, self.steps)
        loss = F.cross_entropy(logits, info.query_y)
        return {"loss": loss, "plain": float(loss.detach()),
                "acc": float((logits.argmax(1) == info.query_y).float().mean())}
