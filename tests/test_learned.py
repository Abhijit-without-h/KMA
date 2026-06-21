"""Integration: a trained chart beats cosine on ancestor retrieval, ties on parity.

This is the headline claim of the project, locked into CI. Training is small
(52-node taxonomy, ~250 epochs) so it runs in a few seconds.
"""

import numpy as np

from kma.chart import HyperbolicChart
from kma.data import build_store
from kma.engine import KMAEngine
from kma.eval import _tree, ancestor_map, parity_auc
from kma.train import train


def test_trained_chart_beats_cosine_on_ancestors():
    store = build_store().store
    _, idx, embs, root, ancestors, depth = _tree(store)

    rand = HyperbolicChart(in_dim=embs.shape[1], dim=64)
    trained = train(epochs=250, verbose=False)

    map_cos = ancestor_map(embs, None, 1.0, ancestors, method="cosine")
    map_rand = ancestor_map(embs, rand.encode(embs), rand.curvature(),
                            ancestors, method="hyp")
    map_tr = ancestor_map(embs, trained.encode(embs), trained.curvature(),
                          ancestors, method="hyp")

    # the win, and the honesty check that *training* (not geometry) caused it
    assert map_tr > map_cos + 0.15
    assert map_tr > map_rand + 0.2

    # parity guardrail: must not wreck flat semantic similarity
    auc_cos = parity_auc(embs, None, 1.0, store, idx, root, depth, method="cosine")
    auc_tr = parity_auc(embs, trained.encode(embs), trained.curvature(),
                        store, idx, root, depth, method="hyp")
    assert auc_tr >= auc_cos - 0.05


def test_chart_outputs_inside_ball():
    chart = HyperbolicChart(in_dim=8, dim=16)
    x = chart.encode(np.random.default_rng(0).standard_normal((5, 8)))
    sc = np.sqrt(chart.curvature())
    assert np.all(sc * np.linalg.norm(x, axis=1) < 1.0)


def test_engine_learned_mode_runs():
    trained = train(epochs=50, verbose=False)
    eng = KMAEngine(chart=trained)
    eng.insert("a feathered flyer with a beak")
    eng.insert("a wheeled road machine")
    hits = eng.query("a bird that flies", k=2, mode="learned")
    assert hits and hits[0].via == "learned"
