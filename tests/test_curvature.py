"""Invariants for the curvature-c geometry; must reduce to c=1 cleanly."""

import numpy as np

from kma import geometry as G


def _rng():
    return np.random.default_rng(1)


def test_c1_matches_base_dist():
    rng = _rng()
    for _ in range(100):
        x = G.expmap0(rng.standard_normal(6))
        y = G.expmap0(rng.standard_normal(6))
        assert abs(G.dist_c(x, y, 1.0) - G.dist(x, y)) < 1e-7


def test_expmap0_c_distance_from_origin_matches_norm():
    rng = _rng()
    o = np.zeros(5)
    for c in (0.25, 1.0, 3.0):
        for _ in range(50):
            v = rng.standard_normal(5)
            x = G.expmap0_c(v, c)
            assert abs(G.dist_c(o, x, c) - np.linalg.norm(v)) < 1e-5


def test_points_stay_in_curved_ball():
    rng = _rng()
    for c in (0.25, 1.0, 4.0):
        for _ in range(100):
            x = G.expmap0_c(rng.standard_normal(7) * 4, c)
            assert np.sqrt(c) * np.linalg.norm(x) < 1.0


def test_logmap_inverts_expmap_c():
    rng = _rng()
    for c in (0.5, 1.0, 2.0):
        for _ in range(50):
            v = rng.standard_normal(5)
            back = G.logmap0_c(G.expmap0_c(v, c), c)
            assert np.allclose(back, v, atol=1e-5)


def test_mobius_add_c_left_identity():
    rng = _rng()
    zero = np.zeros(4)
    for c in (0.5, 1.0, 2.0):
        x = G.expmap0_c(rng.standard_normal(4), c)
        assert np.allclose(G.mobius_add_c(zero, x, c), x, atol=1e-9)


def test_dist_c_batch_matches_scalar():
    rng = _rng()
    for c in (0.5, 1.0, 2.0):
        q = G.expmap0_c(rng.standard_normal(7), c)
        pts = np.array([G.expmap0_c(rng.standard_normal(7), c) for _ in range(20)])
        batch = G.dist_c_batch(q, pts, c)
        for i in range(len(pts)):
            assert abs(batch[i] - G.dist_c(q, pts[i], c)) < 1e-7
