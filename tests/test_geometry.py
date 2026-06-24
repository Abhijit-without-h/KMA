"""Invariants for the Poincare ball math."""

import numpy as np

from kma import geometry as G


def _rng():
    return np.random.default_rng(0)


def test_points_stay_in_ball():
    rng = _rng()
    for _ in range(200):
        v = rng.standard_normal(8) * 5
        x = G.expmap0(v)
        assert np.linalg.norm(x) < 1.0


def test_dist_is_symmetric_and_nonneg():
    rng = _rng()
    for _ in range(100):
        x = G.expmap0(rng.standard_normal(8))
        y = G.expmap0(rng.standard_normal(8))
        d1, d2 = G.dist(x, y), G.dist(y, x)
        assert d1 >= 0
        assert abs(d1 - d2) < 1e-9


def test_dist_to_self_is_zero():
    rng = _rng()
    for _ in range(50):
        x = G.expmap0(rng.standard_normal(6))
        assert G.dist(x, x) < 1e-6


def test_expmap0_distance_from_origin_matches_norm():
    rng = _rng()
    o = np.zeros(5)
    for _ in range(100):
        v = rng.standard_normal(5)
        x = G.expmap0(v)
        assert abs(G.dist(o, x) - np.linalg.norm(v)) < 1e-6


def test_logmap_inverts_expmap():
    rng = _rng()
    for _ in range(100):
        v = rng.standard_normal(5)
        back = G.logmap0(G.expmap0(v))
        assert np.allclose(back, v, atol=1e-5)


def test_mobius_add_left_identity():
    rng = _rng()
    zero = np.zeros(4)
    for _ in range(50):
        x = G.expmap0(rng.standard_normal(4))
        assert np.allclose(G.mobius_add(zero, x), x, atol=1e-9)


def test_dist_batch_matches_scalar():
    rng = _rng()
    q = G.expmap0(rng.standard_normal(7))
    pts = np.array([G.expmap0(rng.standard_normal(7)) for _ in range(30)])
    batch = G.dist_batch(q, pts)
    for i in range(len(pts)):
        assert abs(batch[i] - G.dist(q, pts[i])) < 1e-9
