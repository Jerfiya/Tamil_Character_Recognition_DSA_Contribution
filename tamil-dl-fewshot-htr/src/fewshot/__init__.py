"""Few-shot methods. All share one interface so the trainer and evaluator stay method-agnostic.

``features(x)``            per-image features (cacheable: computed once per image)
``episode_logits(...)``    [M, N] class scores for M queries given class-major support features
``train_loss(...)``        dict with ``loss`` (optimised), ``plain`` (comparable loss), ``acc``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import Embedder, build_backbone
from losses import smooth_ce

log = logging.getLogger(__name__)


@dataclass
class EpisodeInfo:
    """Label bookkeeping for one training step."""
    n_way: int
    k_shot: int
    query_y: torch.Tensor      # [M] episode-local labels of the queries
    all_local: torch.Tensor    # [N*K + M] episode-local labels (support first)
    all_global: torch.Tensor   # [N*K + M] indices into the training-class head
    smoothing: float = 0.0


class FewShotMethod(nn.Module):
    """Base class: episodic training with cross-entropy over query logits."""
    cacheable = True           # features can be computed per image, independent of the episode
    needs_images = False       # True for MAML (adapts on raw images)
    name = "base"

    def __init__(self, backbone: Embedder, mcfg: Mapping[str, Any], n_train_classes: int = 0) -> None:
        super().__init__()
        self.backbone = backbone
        self.mcfg = dict(mcfg)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def episode_logits(self, sf: torch.Tensor, qf: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        raise NotImplementedError

    def train_loss(self, feats: torch.Tensor, info: EpisodeInfo) -> Dict[str, Any]:
        ns = info.n_way * info.k_shot
        logits = self.episode_logits(feats[:ns], feats[ns:], info.n_way, info.k_shot)
        plain = F.cross_entropy(logits, info.query_y)
        loss = smooth_ce(logits, info.query_y, info.smoothing) if info.smoothing > 0 else plain
        return {"loss": loss, "plain": float(plain.detach()),
                "acc": float((logits.argmax(1) == info.query_y).float().mean())}

    def val_loss(self, sf: torch.Tensor, qf: torch.Tensor, query_y: torch.Tensor, n_way: int, k_shot: int,
                 logits: torch.Tensor) -> float:
        """Validation loss of one fixed episode (cross-entropy unless a method overrides it)."""
        return float(F.cross_entropy(logits, query_y))

    def pool_val(self, feats: torch.Tensor, y_global: torch.Tensor) -> Optional[Dict[str, float]]:
        """Optional non-episodic validation on a labelled pool (supervised baseline only)."""
        return None

    def stats(self) -> Dict[str, float]:
        """Scalar diagnostics logged every epoch (e.g. temperature)."""
        return {}


def build_method(cfg: Mapping[str, Any], n_train_classes: int = 0) -> FewShotMethod:
    """Instantiate backbone + method from the config."""
    from baselines import SupervisedClassifier
    from .arcface_pretrain import ArcFaceModel
    from .baseline_pp import BaselinePP
    from .maml import FOMAML
    from .matchingnet import MatchingNet
    from .protonet import ProtoNet
    from .relationnet import RelationNet
    from .siamese import SiameseNet
    registry = {"protonet": ProtoNet, "matchingnet": MatchingNet, "relationnet": RelationNet,
                "siamese": SiameseNet, "arcface": ArcFaceModel, "baselinepp": BaselinePP, "maml": FOMAML,
                "supervised": SupervisedClassifier}
    name = cfg["method"]["name"]
    if name not in registry:
        raise ValueError(f"unknown method: {name}")
    model = registry[name](build_backbone(cfg["model"]), cfg["method"], n_train_classes)
    init = cfg["method"].get("init_from")
    if init:
        load_backbone(model, Path(init))
    return model


def load_backbone(model: FewShotMethod, ckpt_path: Path) -> None:
    """Initialise the backbone from another run's checkpoint (EMA weights when present)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    sub = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(sub, strict=False)
    log.info("backbone initialised from %s (missing=%d unexpected=%d)", ckpt_path, len(missing), len(unexpected))
