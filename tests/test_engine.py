"""Insertion / query round-trips and hierarchy geometry."""

import numpy as np

from kma import geometry as G
from kma import placement
from kma.engine import KMAEngine


def test_insert_creates_valid_node():
    eng = KMAEngine()
    node = eng.insert("hello world")
    assert node.depth == 0
    assert node.parent_id is None
    assert np.linalg.norm(node.coord) < 1.0
    assert len(node.embedding) == eng.embedder.dim


def test_parent_child_link_and_depth():
    eng = KMAEngine()
    root = eng.insert("animals")
    child = eng.insert("dogs", parent_id=root.id)
    assert child.depth == 1
    assert child.id in eng.store.get(root.id).children_ids
    # one edge ~= one hyperbolic STEP from the parent
    assert abs(G.dist(root.coord, child.coord) - placement.STEP) < 0.2


def test_children_push_outward():
    eng = KMAEngine()
    root = eng.insert("topic")
    child = eng.insert("subtopic detail", parent_id=root.id)
    assert np.linalg.norm(child.coord) > np.linalg.norm(root.coord)


def test_query_round_trip_finds_inserted_text():
    eng = KMAEngine()
    eng.insert("the quick brown fox jumps")
    eng.insert("completely unrelated cooking recipe")
    hits = eng.query("quick brown fox", k=1)
    assert hits
    assert "fox" in hits[0].node.text


def test_flat_baseline_recovers_pure_cosine_ordering():
    eng = KMAEngine()
    for t in ["alpha beta", "gamma delta", "alpha beta gamma"]:
        eng.insert(t)
    hits = eng.query("alpha beta", k=3, alpha=1.0, beta=0.0, expand=False)
    # top hit must be the highest-cosine node
    assert hits[0].cosine == max(h.cosine for h in hits)


def test_persistence_round_trip(tmp_path):
    eng = KMAEngine()
    r = eng.insert("root")
    eng.insert("child", parent_id=r.id)
    p = tmp_path / "store.json"
    eng.store.save(p)

    eng2 = KMAEngine()
    eng2.store.load(p)
    assert len(eng2.store) == 2
    assert eng2.store.get(r.id).children_ids
