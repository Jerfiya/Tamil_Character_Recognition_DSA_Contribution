"""Loss functions and metric-space statistics shared by all methods."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def smooth_ce(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy with optional label smoothing."""
    return F.cross_entropy(logits, target, label_smoothing=float(smoothing))


def prototypes(support: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
    """Mean support embedding per class. ``support`` is [N*K, D], class-major."""
    return support.reshape(n_way, k_shot, -1).mean(1)


def sq_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances [len(a), len(b)]."""
    return (a.pow(2).sum(1, keepdim=True) - 2.0 * a @ b.t() + b.pow(2).sum(1).unsqueeze(0)).clamp_min(0.0)


def proto_logits(support: torch.Tensor, query: torch.Tensor, n_way: int, k_shot: int,
                 distance: str = "euclidean") -> torch.Tensor:
    """Un-scaled prototypical logits: -squared distance, or cosine similarity."""
    protos = prototypes(support, n_way, k_shot)
    if distance == "cosine":
        return F.normalize(query, dim=1) @ F.normalize(protos, dim=1).t()
    return -sq_euclidean(query, protos)


def _pairwise(emb: torch.Tensor) -> torch.Tensor:
    return sq_euclidean(emb, emb).add(1e-12).sqrt()


def batch_hard_triplet(emb: torch.Tensor, y: torch.Tensor, margin: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Triplet loss with hard-negative mining ("batch-all, active only").

    Every (anchor, positive, negative) triplet in the batch is formed; the loss is the mean over
    the triplets that still violate the margin, so easy triplets do not dilute the gradient.
    Pure batch-hard mining (hardest positive vs hardest negative per anchor) collapsed every
    embedding to one point on this data (loss stuck at the margin); see reports/decisions.md.
    """
    d = _pairwise(emb)
    same = y.unsqueeze(0) == y.unsqueeze(1)
    eye = torch.eye(len(y), dtype=torch.bool, device=emb.device)
    pos_mask, neg_mask = same & ~eye, ~same
    trip = d.unsqueeze(2) - d.unsqueeze(1) + margin                 # [a, p, n] = d(a,p) - d(a,n) + m
    valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
    viol = F.relu(trip) * valid
    active = (viol > 1e-12)
    n_active = active.sum()
    loss = viol.sum() / n_active.clamp_min(1)
    dd = d.detach()                                                 # components below are diagnostics only
    hard_pos = (dd * pos_mask).max(1).values
    hard_neg = (dd + 1e6 * (~neg_mask)).min(1).values
    comps = {"loss_pos": float(hard_pos.mean()), "loss_neg": float(F.relu(margin - hard_neg).mean()),
             "dist_pos": float(dd[pos_mask].mean()), "dist_neg": float(dd[neg_mask].mean()),
             "active_frac": float(n_active / valid.sum().clamp_min(1))}
    return loss, comps


def contrastive_loss(emb: torch.Tensor, y: torch.Tensor, margin: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pairwise contrastive loss over all pairs in the batch, positives and negatives balanced."""
    d = _pairwise(emb)
    same = y.unsqueeze(0) == y.unsqueeze(1)
    eye = torch.eye(len(y), dtype=torch.bool, device=emb.device)
    pos_mask, neg_mask = same & ~eye, ~same
    lp = d[pos_mask].pow(2).mean()
    neg_terms = F.relu(margin - d[neg_mask]).pow(2)
    k = min(len(neg_terms), max(1, int(pos_mask.sum())))          # hard-negative mining: top-k violators
    ln = neg_terms.topk(k).values.mean()
    comps = {"loss_pos": float(lp), "loss_neg": float(ln), "dist_pos": float(d[pos_mask].mean()),
             "dist_neg": float(d[neg_mask].mean()), "active_frac": float((neg_terms > 0).float().mean())}
    return lp + ln, comps


def margin_logits(cos: torch.Tensor, y: torch.Tensor, s: float, m: float, kind: str = "arcface") -> torch.Tensor:
    """ArcFace (additive angular) or CosFace (additive cosine) margin logits."""
    onehot = F.one_hot(y, cos.shape[1]).bool()
    if kind == "cosface":
        return s * torch.where(onehot, cos - m, cos)
    c = cos.clamp(-1 + 1e-7, 1 - 1e-7)
    phi = torch.cos(torch.acos(c) + m)
    phi = torch.where(c > math.cos(math.pi - m), phi, c - m * math.sin(math.pi - m))  # keep monotonic
    return s * torch.where(onehot, phi, cos)


@torch.no_grad()
def separation_stats(feats: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Inter-class prototype distance vs intra-class spread (Euclidean) and their ratio."""
    feats = feats.flatten(1).float()
    classes = labels.unique()
    cents, intra = [], []
    for c in classes:
        f = feats[labels == c]
        mu = f.mean(0)
        cents.append(mu)
        intra.append((f - mu).norm(dim=1).mean())
    cents_t = torch.stack(cents)
    d = _pairwise(cents_t)
    n = len(classes)
    inter = float(d.sum() / max(n * (n - 1), 1))
    intra_m = float(torch.stack(intra).mean())
    return {"inter_dist": inter, "intra_spread": intra_m, "sep_ratio": inter / max(intra_m, 1e-8),
            "emb_norm": float(feats.norm(dim=1).mean())}
