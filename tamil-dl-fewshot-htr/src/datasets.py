"""In-memory access to the processed dataset and the Protocol A / B pools."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import Cfg, resolve
from preprocessing import normalize_batch

log = logging.getLogger(__name__)
Pools = Dict[int, np.ndarray]


@dataclass
class ProcessedData:
    """Preprocessed images plus manifest, class table and both protocol definitions."""
    images: np.ndarray                 # [N, H, W] uint8, ink white on black
    manifest: pd.DataFrame
    classes: List[Dict[str, Any]]      # class_names.json
    class_split: Dict[str, Any]
    cfg_pre: Any

    def __post_init__(self) -> None:
        self.labels = self.manifest.class_idx.to_numpy()
        self.class_names = [c["label"] for c in self.classes]
        self.name_to_idx = {c["label"]: int(c["index"]) for c in self.classes}
        self._tensor: Optional[torch.Tensor] = None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def class_ids(self, part: str) -> List[int]:
        """Class indices of a Protocol B part (base / val / novel / hard_novel)."""
        return [self.name_to_idx[x] for x in self.class_split[part]]

    def pools_a(self, part: str) -> Pools:
        """Protocol A pools: class -> image indices in ``part`` (train / val / test)."""
        m = self.manifest
        sel = m[m.split_A == part]
        return {int(c): g.idx.to_numpy() for c, g in sel.groupby("class_idx")}

    def pools_b(self, part: str) -> Pools:
        """Protocol B pools: every image of each class in ``part``."""
        ids = set(self.class_ids(part))
        m = self.manifest[self.manifest.class_idx.isin(ids)]
        return {int(c): g.idx.to_numpy() for c, g in m.groupby("class_idx")}

    def episode_pools(self, protocol: str, stage: str) -> Tuple[Pools, Optional[Pools]]:
        """(support_pools, query_pools) for ``stage`` in train / val / test / hard.

        Protocol A: support always from the train pool, queries from the stage's pool.
        Protocol B: support and query from the same (class-disjoint) part.
        """
        if protocol == "A":
            if stage == "train":
                return self.pools_a("train"), None
            return self.pools_a("train"), self.pools_a({"val": "val", "test": "test"}[stage])
        part = {"train": "base", "val": "val", "test": "novel", "hard": "hard_novel"}[stage]
        return self.pools_b(part), None

    def tensor(self) -> torch.Tensor:
        """All images as a uint8 tensor [N, 1, H, W] (shared, built lazily)."""
        if self._tensor is None:
            self._tensor = torch.from_numpy(self.images).unsqueeze(1)
        return self._tensor

    def batch(self, idx, device: torch.device) -> torch.Tensor:
        """Float batch in [0, 1] (not yet normalised) for image indices ``idx``."""
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return self.tensor()[idx_t].to(device).float().div_(255.0)

    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        return normalize_batch(x01, self.cfg_pre)


def load_processed(cfg: Cfg, size: Optional[int] = None) -> ProcessedData:
    """Load the cached dataset written by ``prepare_data.py``."""
    root = resolve(cfg, "processed")
    size = int(size or cfg.model.get("input_size") or cfg.preprocess.size)
    path = root / f"images_{size}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python src/prepare_data.py` first")
    manifest = pd.read_csv(root / "manifest.csv", encoding="utf-8-sig", keep_default_na=False)
    classes = json.loads((root / "class_names.json").read_text(encoding="utf-8"))
    class_split = json.loads((root / "class_split.json").read_text(encoding="utf-8"))
    return ProcessedData(np.load(path), manifest, classes, class_split, cfg.preprocess)


def cap_pools(pools: Pools, max_per_class: int, seed: int) -> Pools:
    """Deterministically keep at most ``max_per_class`` images per class."""
    rng = np.random.default_rng(seed)
    return {c: (np.sort(rng.choice(v, max_per_class, replace=False)) if len(v) > max_per_class else v)
            for c, v in sorted(pools.items())}
