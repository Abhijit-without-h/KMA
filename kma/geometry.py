"""Poincare ball geometry — the math heart of KMA.

All operations are pure numpy and act on the open unit ball
    B^n = { x in R^n : ||x|| < 1 }
with curvature -1. We keep every point strictly inside the ball via
`project`, because Mobius maps and the distance formula blow up on the
boundary (this is research-risk 11.4 in the brief, handled explicitly).

References: Ungar, "Gyrovector spaces"; Ganea et al., "Hyperbolic NN".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]

# Largest allowed norm. Strictly < 1 so (1 - ||x||^2) never underflows to 0.
MAX_NORM = 1.0 - 1e-5
EPS = 1e-9


def project(x: Vector) -> Vector:
    """Clamp a point back inside the ball (||x|| <= MAX_NORM)."""
    x = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    if norm >= MAX_NORM:
        x = x * (MAX_NORM / (norm + EPS))
    return x


def mobius_add(x: Vector, y: Vector) -> Vector:
    """Mobius (gyro) addition x (+) y in the Poincare ball.

    This is the hyperbolic analogue of vector addition; it is the building
    block for translating a parent point by a child's tangent offset.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = float(np.dot(x, y))
    xx = float(np.dot(x, x))
    yy = float(np.dot(y, y))
    num = (1.0 + 2.0 * xy + yy) * x + (1.0 - xx) * y
    den = 1.0 + 2.0 * xy + xx * yy
    return project(num / (den + EPS))


def expmap0(v: Vector) -> Vector:
    """Exponential map at the origin: tangent vector v -> point in the ball.

    Maps a Euclidean direction/length at the center onto the manifold. A
    longer v lands closer to the boundary (hyperbolic distance from origin
    equals ||v||).
    """
    # Curvature -1 convention: d(0, expmap0(v)) == ||v||, so use tanh(||v||/2).
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < EPS:
        return np.zeros_like(v)
    return project(np.tanh(norm / 2.0) * v / norm)


def logmap0(x: Vector) -> Vector:
    """Inverse of expmap0: point in the ball -> tangent vector at origin."""
    x = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    if norm < EPS:
        return np.zeros_like(x)
    return 2.0 * np.arctanh(min(norm, MAX_NORM)) * x / norm


def dist(x: Vector, y: Vector) -> float:
    """Hyperbolic (geodesic) distance between two points in the ball."""
    x = project(x)
    y = project(y)
    diff = float(np.dot(x - y, x - y))
    xx = float(np.dot(x, x))
    yy = float(np.dot(y, y))
    arg = 1.0 + 2.0 * diff / ((1.0 - xx) * (1.0 - yy) + EPS)
    return float(np.arccosh(max(arg, 1.0)))


def dist_batch(q: Vector, pts: NDArray[np.float64]) -> Vector:
    """Hyperbolic distance from a single point q to each row of `pts`."""
    q = project(q)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    qn = float(np.dot(q, q))
    pn = np.einsum("ij,ij->i", pts, pts)
    diff = np.einsum("ij,ij->i", pts - q, pts - q)
    arg = 1.0 + 2.0 * diff / ((1.0 - qn) * (1.0 - pn) + EPS)
    return np.arccosh(np.maximum(arg, 1.0))


# ---------------------------------------------------------------------------
# Curvature-`c` (-c, c>0) generalization. The ball of curvature -c has radius
# 1/sqrt(c). We use the convention where the Euclidean norm of a tangent vector
# at the origin equals the geodesic distance from the origin, so:
#     expmap0_c(v) = tanh(sqrt(c)|v|/2)/sqrt(c) * v/|v|
#     dist_c(0,x)  = (2/sqrt(c)) artanh(sqrt(c)|x|)
# At c=1 these reduce exactly to the functions above (and the tests stay green).
# These power the *learnable-curvature* trained chart; the c=1 versions remain
# the default fast path for the heuristic engine.
# ---------------------------------------------------------------------------


def project_c(x: Vector, c: float) -> Vector:
    """Clamp x inside the curvature-c ball (sqrt(c)||x|| <= 1 - 1e-5)."""
    x = np.asarray(x, dtype=np.float64)
    max_norm = (1.0 - 1e-5) / np.sqrt(c)
    norm = float(np.linalg.norm(x))
    if norm >= max_norm:
        x = x * (max_norm / (norm + EPS))
    return x


def mobius_add_c(x: Vector, y: Vector, c: float) -> Vector:
    """Mobius addition in the curvature-c ball."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = float(np.dot(x, y))
    xx = float(np.dot(x, x))
    yy = float(np.dot(y, y))
    num = (1.0 + 2.0 * c * xy + c * yy) * x + (1.0 - c * xx) * y
    den = 1.0 + 2.0 * c * xy + c * c * xx * yy
    return project_c(num / (den + EPS), c)


def expmap0_c(v: Vector, c: float) -> Vector:
    """Exponential map at the origin for curvature c."""
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < EPS:
        return np.zeros_like(v)
    sc = np.sqrt(c)
    return project_c(np.tanh(sc * norm / 2.0) / sc * v / norm, c)


def logmap0_c(x: Vector, c: float) -> Vector:
    """Inverse exponential map at the origin for curvature c."""
    x = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    if norm < EPS:
        return np.zeros_like(x)
    sc = np.sqrt(c)
    return (2.0 / sc) * np.arctanh(min(sc * norm, 1.0 - 1e-7)) * x / norm


def dist_c(x: Vector, y: Vector, c: float) -> float:
    """Geodesic distance in the curvature-c ball."""
    sc = np.sqrt(c)
    diff = mobius_add_c(-np.asarray(x, dtype=np.float64), y, c)
    dn = float(np.linalg.norm(diff))
    return float((2.0 / sc) * np.arctanh(min(sc * dn, 1.0 - 1e-7)))


def dist_c_batch(q: Vector, pts: NDArray[np.float64], c: float) -> Vector:
    """Curvature-c distance from point q to each row of pts (vectorized)."""
    mq = -np.asarray(q, dtype=np.float64)            # x = -q in (-q) (+) y
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    qq = float(np.dot(mq, mq))
    xy = pts @ mq
    yy = np.einsum("ij,ij->i", pts, pts)
    num = (1.0 + 2.0 * c * xy + c * yy)[:, None] * mq[None, :] + (1.0 - c * qq) * pts
    den = (1.0 + 2.0 * c * xy + c * c * qq * yy)[:, None]
    diff = num / (den + EPS)
    dn = np.linalg.norm(diff, axis=1)
    sc = np.sqrt(c)
    return (2.0 / sc) * np.arctanh(np.minimum(sc * dn, 1.0 - 1e-7))
