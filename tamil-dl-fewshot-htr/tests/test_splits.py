"""Protocol definitions: no leakage between splits, dedupe before split, class disjointness."""
import json

import numpy as np
import pandas as pd

from conftest import ROOT, needs_data
from episodes import EpisodeSampler


def _man():
    return pd.read_csv(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8-sig", keep_default_na=False)


@needs_data
def test_protocol_a_pools_disjoint():
    m = _man()
    assert set(m.split_A) <= {"train", "val", "test"}
    assert m.idx.is_unique and m.path.is_unique
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not set(m[m.split_A == a].path) & set(m[m.split_A == b].path)


@needs_data
def test_no_exact_duplicates_left():
    """Dedupe happened before splitting: no identical file content anywhere in the manifest."""
    m = _man()
    assert m.md5.is_unique
    assert (m.dup_of.astype(str).str.len() == 0).all()


@needs_data
def test_protocol_b_disjoint_and_complete():
    split = json.loads((ROOT / "data" / "processed" / "class_split.json").read_text(encoding="utf-8"))
    base, val, novel = set(split["base"]), set(split["val"]), set(split["novel"])
    assert not base & val and not base & novel and not val & novel
    assert set(split["hard_novel"]) <= novel
    labels = json.loads((ROOT / "data" / "processed" / "class_names.json").read_text(encoding="utf-8"))
    assert base | val | novel == {c["label"] for c in labels}


@needs_data
def test_episode_pools_respect_protocols(cfg):
    from datasets import load_processed
    d = load_processed(cfg)
    base, _ = d.episode_pools("B", "train")
    novel, _ = d.episode_pools("B", "test")
    assert not set(base) & set(novel)
    sp, qp = d.episode_pools("A", "test")
    tr = set(np.concatenate(list(sp.values())).tolist())
    te = set(np.concatenate(list(qp.values())).tolist())
    assert not tr & te
    s = EpisodeSampler(sp, qp, 20, 5, 5, seed=0)
    for _ in range(50):
        ep = s.sample()
        assert set(ep.support.reshape(-1).tolist()) <= tr and set(ep.query.tolist()) <= te


@needs_data
def test_class_names_dynamic():
    labels = json.loads((ROOT / "data" / "processed" / "class_names.json").read_text(encoding="utf-8"))
    assert [c["index"] for c in labels] == list(range(len(labels)))
    assert all({"label", "category"} <= set(c) for c in labels)
