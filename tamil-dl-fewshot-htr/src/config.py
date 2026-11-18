"""Configuration loading, CLI overrides, seeding and logging setup."""
from __future__ import annotations

import re

import argparse
import copy
import io
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class Cfg(dict):
    """Dictionary with attribute access (``cfg.train.lr``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo: Dict) -> "Cfg":
        return Cfg({k: copy.deepcopy(v, memo) for k, v in self.items()})


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj: Any) -> Any:
    """Convert a Cfg tree back to plain dicts (for YAML/JSON dumps)."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def apply_overrides(cfg: Cfg, overrides: Optional[Iterable[str]]) -> Cfg:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = Cfg()
            node = node[p]
        node[parts[-1]] = _wrap(yaml.safe_load(raw))
    return cfg


def load_config(path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> Cfg:
    """Load the YAML config and apply overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = _wrap(yaml.safe_load(fh))
    return apply_overrides(cfg, overrides)


def resolve(cfg: Cfg, key: str) -> Path:
    """Absolute path for ``cfg.paths[key]`` (relative paths are relative to the project root)."""
    p = Path(cfg.paths[key])
    return p if p.is_absolute() else (ROOT / p).resolve()


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments shared by every CLI."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. train.lr=0.0003 model.backbone=resnet12")
    parser.add_argument("--log-level", default="INFO")
    return parser


def force_utf8() -> None:
    """Make console streams UTF-8 so Tamil paths and labels never crash a print on Windows."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                setattr(sys, name, io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"))
            except Exception:
                pass


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging to the console (UTF-8) and optionally a file."""
    force_utf8()
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoaders so every worker has a reproducible seed."""
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % (2 ** 32))


def get_device() -> "Any":
    """CUDA when available, else CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_run_name(name: str) -> dict:
    """<YYYYMMDD_HHMMSS>_<method>_<backbone>_<protocol>[_<tag>] -> parts (backbones may contain '_')."""
    m = re.match(r"(\d{8}_\d{6})_([a-z]+)_(.+?)_([AB])(?:_(.*))?$", name)
    if not m:
        return {"method": "?", "backbone": "?", "protocol": "?", "tag": ""}
    return {"method": m.group(2), "backbone": m.group(3), "protocol": m.group(4), "tag": m.group(5) or ""}
