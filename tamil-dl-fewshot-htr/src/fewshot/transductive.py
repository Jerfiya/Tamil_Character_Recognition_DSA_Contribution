"""Transductive prototype refinement: a few soft k-means steps using the unlabeled queries."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from losses import prototypes, sq_euclidean


def _scores(q: torch.Tensor, protos: torch.Tensor, distance: str, scale: float) -> torch.Tensor:
    if distance == "cosine":
        return scale * F.normalize(q, dim=1) @ F.normalize(protos, dim=1).t()
    return -scale * sq_euclidean(q, protos)


@torch.no_grad()
def transductive_logits(sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int, steps: int = 5,
                        distance: str = "euclidean", scale: float = 1.0) -> torch.Tensor:
    """Refine prototypes with soft assignments of the query set, then score the queries.

    Each step: p = softmax(scores); prototype_c = (K * mean_support_c + sum_i p_ic q_i) / (K + sum_i p_ic).
    The labelled support keeps weight K, so a 1-shot prototype moves most.
    """
    sf, qf = sf.flatten(1).float(), qf.flatten(1).float()
    base = prototypes(sf, n_way, k_shot)
    protos = base.clone()
    for _ in range(int(steps)):
        p = F.softmax(_scores(qf, protos, distance, scale), dim=1)           # [M, N]
        protos = (k_shot * base + p.t() @ qf) / (k_shot + p.sum(0)).unsqueeze(1)
    return _scores(qf, protos, distance, scale)
