"""Placement: embedding (+ parent, + depth) -> point in the Poincare ball.

Design (no training in v1):
  * DIRECTION carries semantics. We reduce the high-D embedding to the ball
    dimension with a fixed, seeded random projection (Johnson-Lindenstrauss
    style) and normalize -> a unit direction on the ball.
  * RADIUS carries generality. Children are placed by Mobius-translating the
    parent point by a fixed-length tangent step along the child's own
    direction. Because hyperbolic space expands toward the boundary, deeper
    nodes naturally land further out -> root concepts near center, details
    near the rim. dist(child, parent) ~= STEP for every edge.

A query has no parent/depth, so it is placed at the center, stepped out once
along its own direction (see `place_query`).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from kma import geometry as G

Vector = NDArray[np.float64]

# Hyperbolic length of one parent->child edge. Larger = branches spread more.
STEP = 0.9
SEED = 1234


@lru_cache(maxsize=8)
def _projection(in_dim: int, out_dim: int) -> NDArray[np.float64]:
    """Fixed seeded Gaussian random projection R^in_dim -> R^out_dim."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal((out_dim, in_dim)) / np.sqrt(out_dim)


def direction(embedding: Vector, dim: int) -> Vector:
    """Unit direction in ball-space derived from a semantic embedding."""
    embedding = np.asarray(embedding, dtype=np.float64)
    proj = _projection(embedding.shape[0], dim) @ embedding
    norm = float(np.linalg.norm(proj))
    if norm < G.EPS:
        proj = np.ones(dim)
        norm = float(np.linalg.norm(proj))
    return proj / norm


def place(embedding: Vector, dim: int, parent_coord: Vector | None) -> Vector:
    """Place a node. Root nodes sit one short step from the center."""
    d_hat = direction(embedding, dim)
    if parent_coord is None:
        # Root: small step from origin so root concepts cluster near center.
        return G.expmap0(0.5 * STEP * d_hat)
    local = G.expmap0(STEP * d_hat)            # tangent offset at origin
    return G.mobius_add(np.asarray(parent_coord, dtype=np.float64), local)


def place_query(embedding: Vector, dim: int) -> Vector:
    """Place a query point. No depth info, so step out once from center."""
    return G.expmap0(STEP * direction(embedding, dim))
