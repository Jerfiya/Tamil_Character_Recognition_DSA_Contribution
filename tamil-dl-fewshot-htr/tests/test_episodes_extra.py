"""Episode sampler: N/K/Q, small classes, mixed shots, cross-source episodes."""
import numpy as np
import pytest

from episodes import EpisodeSampler


def test_nkq_counts():
    pools = {c: np.arange(c * 30, c * 30 + 30) for c in range(25)}
    ep = EpisodeSampler(pools, None, 20, 3, 4, seed=0).sample()
    assert ep.support.shape == (20, 3) and len(ep.query) == 80
    assert np.bincount(ep.query_y).tolist() == [4] * 20
    assert len(set(ep.classes.tolist())) == 20


def test_small_classes_excluded_or_q_reduced():
    pools = {0: np.arange(3), 1: np.arange(10, 20), 2: np.arange(20, 30), 3: np.arange(30, 33)}
    s = EpisodeSampler(pools, None, 2, 2, 5, seed=0)
    for _ in range(30):
        assert set(s.sample().classes.tolist()) <= {1, 2}      # classes 0/3 lack K+Q images
    ep = EpisodeSampler(pools, None, 4, 2, 5, seed=0).sample()  # all 4 needed -> Q auto-reduced
    assert ep.n_way == 4 and not set(ep.support.reshape(-1).tolist()) & set(ep.query.tolist())


def test_mixed_shot():
    pools = {c: np.arange(c * 30, c * 30 + 30) for c in range(10)}
    s = EpisodeSampler(pools, None, 5, [1, 5], 3, seed=0)
    assert {s.sample().k_shot for _ in range(40)} == {1, 5}


def test_cross_group_sources_differ():
    pools = {c: np.arange(c * 20, c * 20 + 20) for c in range(6)}
    groups = np.array(["a" if (i % 20) < 10 else "b" for i in range(120)])
    s = EpisodeSampler(pools, None, 5, 3, 3, seed=0, groups=groups)
    for _ in range(20):
        ep = s.sample(cross_group=True)
        for j in range(ep.n_way):
            sg = set(groups[ep.support[j]].tolist())
            assert len(sg) == 1 and not sg & set(groups[ep.query[ep.query_y == j]].tolist())


def test_too_few_classes_raises():
    with pytest.raises(ValueError):
        EpisodeSampler({0: np.arange(10)}, None, 5, 1, 1, seed=0).sample()
