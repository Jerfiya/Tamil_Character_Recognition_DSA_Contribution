"""Shared helpers for every post-training script (calibrate, evaluate, explain, export).

``RunModel`` loads a run's config snapshot and a checkpoint (EMA weights when present) and
offers feature extraction plus method-agnostic logits for an arbitrary support/query split.
``fullset_split`` builds the closed-set "classify every query among all classes of a part"
problem used for per-class metrics, calibration and open-set evaluation.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from config import _wrap, get_device
from datasets import ProcessedData, load_processed
from fewshot import build_method
from tracking import load_checkpoint

log = logging.getLogger(__name__)


class RunModel:
    """A trained model of one run, ready for inference on the cached dataset."""

    def __init__(self, run_dir: Path, ckpt_name: str = "best_val_loss.pt", data: Optional[ProcessedData] = None,
                 threads: Optional[int] = None) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = _wrap(yaml.safe_load((self.run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")))
        torch.set_num_threads(int(threads or self.cfg.get("threads", 2)))
        self.device = get_device()
        self.data = data or load_processed(self.cfg)
        path = self.run_dir / "checkpoints" / ckpt_name
        if not path.exists():
            log.warning("%s missing, using last.pt", path)
            path = self.run_dir / "checkpoints" / "last.pt"
        ck = load_checkpoint(path)
        self.ckpt_path, self.ckpt_epoch = path, int(ck["epoch"])
        self.ckpt_metrics = ck.get("metrics", {})
        self.train_classes: List[int] = list(ck["train_classes"])
        self.model = build_method(_no_init(self.cfg), len(self.train_classes)).to(self.device)
        self.model.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
        self.model.eval()
        self.method = str(self.cfg.method.name)
        self.protocol = str(self.cfg.train.protocol)
        self._cache: Dict[int, torch.Tensor] = {}

    @property
    def needs_images(self) -> bool:
        return bool(self.model.needs_images)

    @torch.no_grad()
    def features(self, idx: Sequence[int], chunk: int = 512) -> torch.Tensor:
        """Per-image features (embedding or feature map) for dataset indices ``idx``."""
        idx = np.asarray(idx, dtype=np.int64)
        out = [self.model.features(self.data.normalize(self.data.batch(idx[i:i + chunk], self.device)))
               for i in range(0, len(idx), chunk)]
        return torch.cat(out) if out else torch.empty(0)

    def cached_features(self, idx: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Features for the unique images in ``idx`` plus a position map (dataset index -> row)."""
        uniq = np.unique(idx)
        pos = np.full(len(self.data.labels), -1, dtype=np.int64)
        pos[uniq] = np.arange(len(uniq))
        return self.features(uniq), pos

    def logits(self, sup: np.ndarray, query: np.ndarray, feats: Optional[torch.Tensor] = None,
               pos: Optional[np.ndarray] = None, q_chunk: int = 256) -> torch.Tensor:
        """[M, N] scores of queries against classes given class-major support ``sup`` [N, K]."""
        n, k = sup.shape
        if self.needs_images:
            xs = self.data.normalize(self.data.batch(sup.reshape(-1), self.device))
            outs = []
            for i in range(0, len(query), q_chunk):
                xq = self.data.normalize(self.data.batch(query[i:i + q_chunk], self.device))
                outs.append(self.model.episode_logits_images(xs, xq, n, k).detach())
            return torch.cat(outs)
        if feats is None:
            feats, pos = self.cached_features(np.concatenate([sup.reshape(-1), query]))
        sf, qf = feats[pos[sup.reshape(-1)]], feats[pos[query]]
        with torch.no_grad():
            return self.model.episode_logits(sf, qf, n, k).detach()

    def embedding_dim(self) -> int:
        return int(self.model.backbone.emb_dim)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def latency_ms(self, n: int = 64) -> float:
        """CPU inference time per single image (batch size 1) for the feature extractor."""
        idx = np.arange(min(n, len(self.data.labels)))
        x = self.data.normalize(self.data.batch(idx, torch.device("cpu")))
        m = self.model.to("cpu")
        for i in range(5):
            m.features(x[i:i + 1])
        import time
        t0 = time.perf_counter()
        for i in range(len(idx)):
            m.features(x[i:i + 1])
        m.to(self.device)
        return 1000.0 * (time.perf_counter() - t0) / len(idx)


def _no_init(cfg):
    """Config copy without ``method.init_from`` (weights come from the checkpoint anyway)."""
    import copy
    c = copy.deepcopy(cfg)
    c["method"]["init_from"] = None
    return c


# --- closed-set split over all classes of a part ---------------------------------------------
def stage_pools(data: ProcessedData, protocol: str, stage: str):
    """(support pools, query pools or None) for stage in val / test / hard."""
    return data.episode_pools(protocol, stage)


def fullset_split(data: ProcessedData, protocol: str, stage: str, k: int, seed: int,
                  classes: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Support [N, K] (class-major) + every remaining image of each class as a query.

    Protocol A: support from the train pool, queries = the stage pool. Classes with fewer
    than K train images are filled by sampling with replacement (logged).
    Protocol B: K support images per class drawn from the class's images, the rest are queries.
    """
    sp, qp = stage_pools(data, protocol, stage)
    rng = np.random.default_rng(seed)
    cls = sorted(sp) if classes is None else sorted(int(c) for c in classes)
    sup, q, qy, filled = [], [], [], 0
    for j, c in enumerate(cls):
        pool = sp[c]
        if qp is None:
            perm = rng.permutation(pool)
            s, rest = perm[:k], perm[k:]
        else:
            if len(pool) < k:
                filled += 1
            s = rng.choice(pool, size=k, replace=len(pool) < k)
            rest = qp.get(c, np.array([], dtype=np.int64))
        sup.append(s); q.append(rest); qy.append(np.full(len(rest), j))
    if filled:
        log.info("full-set %s/%s: %d classes had < %d support images (sampled with replacement)", protocol, stage, filled, k)
    return {"classes": np.array(cls), "support": np.stack(sup), "query": np.concatenate(q),
            "query_y": np.concatenate(qy).astype(np.int64), "filled": filled}


# --- calibration and metrics ------------------------------------------------------------------
@torch.no_grad()
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Temperature T > 0 minimising NLL of softmax(logits / T).

    A coarse-to-fine 1-D search over log T (robust even when logits are nearly flat, where
    gradient-based LBFGS stops immediately on its gradient tolerance)."""
    lg, yy = logits.detach().float().cpu(), y.cpu()
    nll = lambda lt: float(F.cross_entropy(lg / math.exp(lt), yy))  # noqa: E731
    lo, hi = math.log(1e-4), math.log(1e4)
    for _ in range(4):
        grid = np.linspace(lo, hi, 41)
        vals = [nll(g) for g in grid]
        i = int(np.argmin(vals))
        step = grid[1] - grid[0]
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    return float(math.exp(grid[i]))


def ece_bins(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> Tuple[float, List[Dict[str, float]]]:
    """Expected calibration error of top-1 confidences plus per-bin statistics."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            acc, conf = float(correct[m].mean()), float(probs[m].mean())
            ece += m.mean() * abs(acc - conf)
            rows.append({"lo": lo, "hi": hi, "count": int(m.sum()), "acc": acc, "conf": conf})
        else:
            rows.append({"lo": lo, "hi": hi, "count": 0, "acc": float("nan"), "conf": float("nan")})
    return float(ece), rows


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Probability a positive (enrolled) scores higher than a negative (unseen); ties count half."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return float(roc_auc_score(y, np.r_[pos_scores, neg_scores]))


def openset_split(classes: Sequence[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically split classes in half: (enrolled, unseen)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(sorted(classes)))
    h = len(perm) // 2
    return np.sort(perm[h:]), np.sort(perm[:h])


def ci95(x: Sequence[float]) -> float:
    return float(1.96 * np.std(x) / math.sqrt(max(len(x), 1)))
