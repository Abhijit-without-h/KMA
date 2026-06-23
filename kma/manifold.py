from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kma import geometry as G


# ---------------------------------------------------------------------------
# tangent-space k-means (the coarse clustering)
# ---------------------------------------------------------------------------
def _kmeans(points: NDArray, k: int, iters: int = 12, seed: int = 0
            ) -> tuple[NDArray, NDArray]:
    """Plain Lloyd's k-means in Euclidean space. Returns (centers[k,d], labels[N])."""
    rng = np.random.default_rng(seed)
    n = len(points)
    k = max(1, min(k, n))
    # k-means++ style seeding: first center random, rest far from chosen ones.
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    centers[0] = points[rng.integers(n)]
    for j in range(1, k):
        d2 = np.min(
            ((points[:, None, :] - centers[None, :j, :]) ** 2).sum(-1), axis=1
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers[j] = points[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # assign
        dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        # update
        for j in range(k):
            members = points[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
            else:  # empty cluster -> reseed to the worst-served point
                far = np.argmax(np.min(dists, axis=1))
                centers[j] = points[far]
    return centers, labels


# ---------------------------------------------------------------------------
# the region index (hyperbolic IVF)
# ---------------------------------------------------------------------------
@dataclass
class RegionIndex:
    """Inverted-file index over hyperbolic space: route a query to its regions."""

    ids: list[str]
    coords: NDArray            # [N, n] points in the ball
    c: float
    centers: NDArray          # [k, n] region centers (in the ball)
    assignment: NDArray       # [N] region label per point
    k: int
    # Inverted file: region -> row-indices of its members. Precomputed at build
    # so route() gathers candidates in O(members), never O(N) -- this is what
    # keeps the whole index genuinely sub-linear per query.
    members: list[NDArray]

    @classmethod
    def build(cls, ids: list[str], coords: NDArray, c: float,
              k: int | None = None, iters: int = 12, seed: int = 0) -> "RegionIndex":
        coords = np.asarray(coords, dtype=np.float64)
        n = len(ids)
        if k is None:
            k = max(1, int(np.ceil(np.sqrt(n))))
        k = max(1, min(k, n))
        tangent = G.logmap0_c_batch(coords, c)
        centers_t, labels = _kmeans(tangent, k, iters=iters, seed=seed)
        centers = np.stack([G.expmap0_c(ct, c) for ct in centers_t])
        kk = int(len(centers))
        members = [np.nonzero(labels == r)[0] for r in range(kk)]
        return cls(ids=list(ids), coords=coords, c=c, centers=centers,
                   assignment=labels, k=kk, members=members)

    def assign_one(self, coord: NDArray) -> int:
        """Nearest region for a new point -- O(k), for incremental insert."""
        return int(np.argmin(G.dist_c_batch(coord, self.centers, self.c)))

    def route(self, q_coord: NDArray, nprobe: int = 2) -> NDArray:
        """Row-indices of candidates: members of the nearest `nprobe` regions.

        Cost is O(k) to rank centers + O(sum of probed members) to gather -- it
        never touches the other N-members points, so the query stays sub-linear.
        """
        d = G.dist_c_batch(q_coord, self.centers, self.c)
        nprobe = max(1, min(nprobe, len(self.centers)))
        regions = np.argpartition(d, nprobe - 1)[:nprobe] if nprobe < len(d) \
            else np.arange(len(d))
        if nprobe == 1:
            return self.members[int(regions[0])]
        return np.concatenate([self.members[int(r)] for r in regions])

    def search(self, q_coord: NDArray, k: int = 5, nprobe: int = 3
               ) -> list[tuple[str, float]]:
        """Coarse-to-fine: route, then rank the shortlist by true dist_c."""
        cand = self.route(q_coord, nprobe=nprobe)
        if len(cand) == 0:
            return []
        d = G.dist_c_batch(q_coord, self.coords[cand], self.c)
        order = np.argsort(d)[:k]
        return [(self.ids[cand[i]], float(d[i])) for i in order]


# ---------------------------------------------------------------------------
# the geodesic walk ("roll along the curve")
# ---------------------------------------------------------------------------
def build_knn(coords: NDArray, c: float, m: int = 8) -> list[list[int]]:
    """Small m-NN proximity graph in the hyperbolic metric (for the walk)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    m = min(m, n - 1) if n > 1 else 0
    adj: list[list[int]] = []
    for i in range(n):
        d = G.dist_c_batch(coords[i], coords, c)
        d[i] = np.inf
        adj.append(np.argsort(d)[:m].tolist())
    return adj


def walk(q_coord: NDArray, coords: NDArray, adj: list[list[int]], seed: int,
         c: float, max_steps: int = 256) -> tuple[int, float]:
    """Greedy geodesic descent: from `seed`, hop to the neighbour closest to the
    query until no neighbour improves. Returns (node_index, distance)."""
    coords = np.asarray(coords, dtype=np.float64)
    cur = seed
    cur_d = float(G.dist_c(q_coord, coords[cur], c))
    for _ in range(max_steps):
        best, best_d = cur, cur_d
        for nb in adj[cur]:
            dd = float(G.dist_c(q_coord, coords[nb], c))
            if dd < best_d:
                best, best_d = nb, dd
        if best == cur:
            break
        cur, cur_d = best, best_d
    return cur, cur_d


# ---------------------------------------------------------------------------
# consolidation / folding (amortized self-organization)
# ---------------------------------------------------------------------------
def _greedy_fold(ids: list[str], radius: NDArray, close: NDArray
                 ) -> list[tuple[str, str]]:
    """Given a boolean `close[i,j]` matrix, greedily pair duplicates. Survivor is
    the more general memory (smaller radius); the other is absorbed."""
    n = len(ids)
    used: set[int] = set()
    folds: list[tuple[str, str]] = []
    for i in range(n):
        if i in used:
            continue
        for j in range(i + 1, n):
            if j in used or not close[i, j]:
                continue
            if radius[i] <= radius[j]:
                folds.append((ids[i], ids[j]))
                used.add(j)
            else:
                folds.append((ids[j], ids[i]))
                used.add(i)
                break
    return folds


def find_folds(ids: list[str], coords: NDArray, c: float, tau_fold: float
               ) -> list[tuple[str, str]]:
    """Pairs (survivor_id, absorbed_id) for memories within `tau_fold` hyperbolic
    distance -- used to merge crowded REGIONS. The survivor is the more general
    one (smaller radius). Greedy, single pass; consolidation is offline."""
    coords = np.asarray(coords, dtype=np.float64)
    radius = np.sqrt(c) * np.linalg.norm(coords, axis=1)
    n = len(ids)
    close = np.zeros((n, n), dtype=bool)
    for i in range(n):
        d = G.dist_c_batch(coords[i], coords, c)
        close[i] = d < tau_fold
    return _greedy_fold(ids, radius, close)


def find_folds_by_similarity(ids: list[str], embs: NDArray, coords: NDArray,
                             c: float, sim: float = 0.98) -> list[tuple[str, str]]:
    """Pairs (survivor_id, absorbed_id) for near-DUPLICATE CONTENT.

    Redundancy is a property of meaning, not placement: in the heuristic chart a
    duplicate is deliberately a STEP away from its parent, so embedding cosine --
    not ball distance -- is the honest duplicate signal. Survivor = more general
    (smaller radius), matching the hierarchy convention."""
    embs = np.asarray(embs, dtype=np.float64)        # already L2-normalized
    coords = np.asarray(coords, dtype=np.float64)
    radius = np.sqrt(c) * np.linalg.norm(coords, axis=1)
    sims = embs @ embs.T
    close = sims >= sim
    return _greedy_fold(ids, radius, close)
