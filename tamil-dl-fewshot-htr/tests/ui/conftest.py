"""Fixtures for the web-UI tests.

Every test that can write (enrol) works on a temporary copy of ``data/support_set`` and a
temporary prototype cache, so the real support set, the dataset and all checkpoints stay untouched.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "ui"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
needs_ui_data = pytest.mark.skipif(not (MANIFEST.exists() and (ROOT / "models" / "deploy.json").exists()),
                                   reason="needs data/processed/manifest.csv and models/deploy.json")


def sample_images(n: int = 20, seed: int = 0) -> List[Dict[str, str]]:
    """``n`` Protocol-A TEST images (path + label), deterministic, spread over all categories."""
    import pandas as pd
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig", keep_default_na=False)
    t = m[m.split_A == "test"]
    t = t[t.path.map(lambda p: Path(p).exists())]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(t), size=min(n, len(t)), replace=False)
    return [{"path": r.path, "label": r.label, "kind": r.kind} for r in t.iloc[sorted(idx)].itertuples()]


def one_per_kind() -> Dict[str, Dict[str, str]]:
    import pandas as pd
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig", keep_default_na=False)
    t = m[(m.split_A == "test")]
    out = {}
    for kind in ("vowel", "consonant", "compound", "aytham"):
        r = t[t.kind == kind].iloc[0]
        out[kind] = {"path": r.path, "label": r.label}
    return out


@pytest.fixture(scope="session")
def tmp_cfg(tmp_path_factory):
    """Config whose support set and caches live in a temp folder (a copy of the real one)."""
    from config import load_config
    base = tmp_path_factory.mktemp("ui_support")
    sup = base / "support_set"
    shutil.copytree(ROOT / "data" / "support_set", sup)
    cfg = load_config()
    cfg["paths"]["support_set"] = str(sup)
    cfg["ui"]["cache_dir"] = str(base / "cache")
    return cfg


@pytest.fixture(scope="session")
def service(tmp_cfg):
    from service import RecognitionService
    s = RecognitionService(tmp_cfg)
    s.load()
    return s


@pytest.fixture(scope="session")
def client(service):
    from fastapi.testclient import TestClient
    from server import create_app
    with TestClient(create_app(service)) as c:
        yield c


def png_bytes(draw) -> bytes:
    import cv2
    g = np.full((256, 256), 255, np.uint8)
    draw(g)
    return cv2.imencode(".png", g)[1].tobytes()


def square_png() -> bytes:
    """A plain square: rejected as 'unknown' by the deployed open-set threshold."""
    import cv2
    return png_bytes(lambda g: cv2.rectangle(g, (60, 60), (196, 196), 0, 10))
