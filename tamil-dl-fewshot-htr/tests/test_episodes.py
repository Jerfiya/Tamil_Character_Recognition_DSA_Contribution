"""Sampler invariants and Protocol B class disjointness."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from episodes import EpisodeSampler  # noqa: E402


def _pools(n_cls=12, per=20):
    return {c: np.arange(c * per, (c + 1) * per) for c in range(n_cls)}


def test_support_query_disjoint_single_pool():
    s = EpisodeSampler(_pools(), None, 5, 5, 5, seed=0)
    for _ in range(200):
        ep = s.sample()
        assert not set(ep.support.reshape(-1)) & set(ep.query)
        assert ep.support.shape == (5, 5) and len(ep.query) == 25
        for j, c in enumerate(ep.classes):          # labels point at the right class
            assert all(q // 20 == c for q in ep.query[ep.query_y == j])


def test_support_query_disjoint_two_pools():
    sp = _pools()
    qp = {c: v + 1000 for c, v in sp.items()}
    s = EpisodeSampler(sp, qp, "all", 3, 5, seed=1)
    ep = s.sample()
    assert ep.n_way == 12
    assert not set(ep.support.reshape(-1)) & set(ep.query)
    assert all(q >= 1000 for q in ep.query)


def test_fixed_episodes_deterministic():
    a = EpisodeSampler(_pools(), None, 5, 1, 3, seed=7).fixed(5)
    b = EpisodeSampler(_pools(), None, 5, 1, 3, seed=7).fixed(5)
    assert all((x.query == y.query).all() and (x.support == y.support).all() for x, y in zip(a, b))


def test_protocol_b_classes_disjoint():
    split = json.loads((ROOT / "data" / "processed" / "class_split.json").read_text(encoding="utf-8"))
    base, val, novel = set(split["base"]), set(split["val"]), set(split["novel"])
    assert not base & val and not base & novel and not val & novel
    assert set(split["hard_novel"]) <= novel
