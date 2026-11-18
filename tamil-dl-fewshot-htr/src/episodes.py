"""N-way K-shot Q-query episode sampling.

Invariants (unit-tested): classes are sampled uniformly (class-balanced by construction),
a class is used only if it has enough images, and no image is ever in both the support and
the query set of an episode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

log = logging.getLogger(__name__)
Pools = Dict[int, np.ndarray]


@dataclass
class Episode:
    """One episode. ``support`` is [N, K] image indices; queries are flat with episode-local labels."""
    classes: np.ndarray       # [N] global class ids
    support: np.ndarray       # [N, K] image indices
    query: np.ndarray         # [M] image indices
    query_y: np.ndarray       # [M] labels in 0..N-1

    @property
    def n_way(self) -> int:
        return int(self.support.shape[0])

    @property
    def k_shot(self) -> int:
        return int(self.support.shape[1])


class EpisodeSampler:
    """Samples episodes from per-class index pools.

    If ``query_pools`` is given (Protocol A evaluation) support comes from ``support_pools``
    and queries from ``query_pools``; otherwise both are drawn, disjointly, from one pool.
    ``k_shot`` may be a list for mixed-shot training. ``n_way='all'`` uses every class that has
    K support images and at least one query (queries per class may then be fewer than Q).
    """

    def __init__(self, support_pools: Pools, query_pools: Optional[Pools] = None,
                 n_way: Union[int, str] = 5, k_shot: Union[int, Sequence[int]] = 5, q_query: int = 5,
                 seed: int = 0, groups: Optional[np.ndarray] = None, name: str = "") -> None:
        self.sp = {int(c): np.asarray(v) for c, v in support_pools.items() if len(v)}
        self.qp = None if query_pools is None else {int(c): np.asarray(v) for c, v in query_pools.items() if len(v)}
        self.n_way, self.k_shot, self.q_query = n_way, k_shot, int(q_query)
        self.rng = np.random.default_rng(seed)
        self.groups = groups
        self.name = name
        self._warned: set = set()

    # -- state for exact resume -------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]

    def _warn(self, key: tuple, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("[%s] " + msg, self.name, *args)

    def _avail(self, c: int) -> int:
        """Images usable as queries for class ``c`` if all of its support pool could be used."""
        return len(self.qp[c]) if self.qp is not None and c in self.qp else (0 if self.qp is not None else len(self.sp[c]))

    def eligible(self, k: int, q: int) -> List[int]:
        """Classes that can supply K support and Q query images without overlap."""
        if self.qp is None:
            return [c for c, v in self.sp.items() if len(v) >= k + q]
        return [c for c, v in self.sp.items() if len(v) >= k and len(self.qp.get(c, ())) >= q]

    def sample(self, n_way: Optional[Union[int, str]] = None, k_shot: Optional[int] = None,
               q_query: Optional[int] = None, cross_group: bool = False) -> Episode:
        """Draw one episode (see class docstring). ``cross_group`` forces support and query
        images to come from different source groups."""
        n_way = self.n_way if n_way is None else n_way
        k = self.k_shot if k_shot is None else k_shot
        if not isinstance(k, (int, np.integer)):
            k = int(self.rng.choice(list(k)))
        q = self.q_query if q_query is None else int(q_query)
        if cross_group:
            return self._sample_cross(int(n_way), int(k), q)
        ragged = n_way == "all"
        if ragged:
            classes = np.array(sorted(self.eligible(k, 1)))
            short = sum(1 for c in classes if (len(self.sp[c]) - k if self.qp is None else len(self.qp[c])) < q)
            if short:
                self._warn(("all", k, q), "all-way K=%d: %d of %d classes have fewer than Q=%d queries (using what exists); "
                           "%d classes excluded for lack of images", k, short, len(classes), q, len(self.sp) - len(classes))
        else:
            n_way = int(n_way)
            q_eff, elig = q, self.eligible(k, q)
            while len(elig) < n_way and q_eff > 1:
                q_eff -= 1
                elig = self.eligible(k, q_eff)
            if q_eff != q:
                self._warn((n_way, k, q), "%d-way K=%d: only enough classes after reducing Q %d -> %d", n_way, k, q, q_eff)
                q = q_eff
            if len(elig) < n_way:
                self._warn((n_way, k, "n"), "%d-way K=%d: only %d eligible classes; reducing N", n_way, k, len(elig))
                n_way = len(elig)
            if n_way < 2:
                raise ValueError(f"not enough classes for an episode (K={k}, Q={q})")
            classes = self.rng.choice(np.array(elig), size=n_way, replace=False)
        support, query, qy = [], [], []
        for j, c in enumerate(classes):
            if self.qp is None:
                perm = self.rng.permutation(self.sp[c])
                s, rest = perm[:k], perm[k:k + q]
            else:
                s = self.rng.choice(self.sp[c], size=k, replace=False)
                cand = np.setdiff1d(self.qp[c], s)
                rest = self.rng.choice(cand, size=min(q, len(cand)), replace=False)
            support.append(s); query.append(rest); qy.append(np.full(len(rest), j))
        return Episode(np.asarray(classes), np.stack(support), np.concatenate(query), np.concatenate(qy))

    def _sample_cross(self, n_way: int, k: int, q: int) -> Episode:
        """Support from one source group, queries from the other groups."""
        assert self.groups is not None, "cross-group sampling needs per-image group keys"
        cand: Dict[int, List[str]] = {}
        for c, sp in self.sp.items():
            qpool = sp if self.qp is None else self.qp.get(c, np.array([], int))
            sg, qg = self.groups[sp], self.groups[qpool]
            ok = [g for g in np.unique(sg) if (sg == g).sum() >= k and (qg != g).sum() >= q]
            if ok:
                cand[c] = ok
        if len(cand) < n_way:
            self._warn(("x", n_way, k), "cross-group %d-way K=%d: only %d eligible classes", n_way, k, len(cand))
            n_way = len(cand)
        if n_way < 2:
            raise ValueError("not enough classes for a cross-group episode")
        classes = self.rng.choice(np.array(sorted(cand)), size=n_way, replace=False)
        support, query, qy = [], [], []
        for j, c in enumerate(classes):
            g = self.rng.choice(cand[int(c)])
            sp = self.sp[int(c)]
            qpool = sp if self.qp is None else self.qp[int(c)]
            s = self.rng.choice(sp[self.groups[sp] == g], size=k, replace=False)
            rest = self.rng.choice(qpool[self.groups[qpool] != g], size=q, replace=False)
            support.append(s); query.append(rest); qy.append(np.full(q, j))
        return Episode(np.asarray(classes), np.stack(support), np.concatenate(query), np.concatenate(qy))

    def fixed(self, n: int, **kw) -> List[Episode]:
        """A list of ``n`` episodes (deterministic for a freshly seeded sampler)."""
        return [self.sample(**kw) for _ in range(n)]


class ClassBalancedBatchSampler:
    """Batches of image indices where every class is equally likely (for supervised stages)."""

    def __init__(self, pools: Pools, batch_size: int, seed: int = 0) -> None:
        self.pools = {int(c): np.asarray(v) for c, v in pools.items() if len(v)}
        self.classes = np.array(sorted(self.pools))
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]

    def sample(self) -> np.ndarray:
        cls = self.rng.choice(self.classes, size=self.batch_size, replace=True)
        return np.array([self.pools[int(c)][self.rng.integers(len(self.pools[int(c)]))] for c in cls])
