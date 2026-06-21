"""A/B evaluation: cosine vs random-phi vs trained-phi.

Two tasks, chosen to be honest about where hyperbolic geometry can and cannot
beat cosine:

  (A) ANCESTOR RETRIEVAL (hierarchical reconstruction) -- THE WIN.
      For each node, rank all others as candidate is-a ancestors. Cosine is
      symmetric and has no notion of "more general than", so it ranks similar
      *siblings* above true (general) ancestors. A trained hyperbolic chart
      uses small distance + smaller radius to identify ancestors. Metric: MAP.

  (B) FLAT SIMILARITY PARITY -- the guardrail.
      Rank same-subtopic siblings above cross-branch nodes. Cosine is excellent
      here; the goal for trained-phi is PARITY, proving we didn't wreck
      semantics while gaining hierarchy. Metric: ranking AUC.

The random-phi column (untrained chart) isolates the effect of *training*: any
lift over it is due to learning, not to the geometry alone.

Run:  python -m kma.eval
"""

from __future__ import annotations

import numpy as np

from kma import geometry as G
from kma.chart import HyperbolicChart
from kma.data import build_store

# Ancestor score actively REWARDS generality (smaller radius), not just penalizes
# specificity -- otherwise siblings (same radius, also ~2 edges away) tie with
# true ancestors and MAP collapses.
GEN_REWARD = 3.0


def _tree(store):
    nodes = store.all()
    idx = {n.id: i for i, n in enumerate(nodes)}
    embs = np.array([n.embedding for n in nodes], dtype=np.float64)
    root = np.array([idx[(store.ancestors(n.id) or [n])[-1].id] for n in nodes])
    ancestors = [{idx[a.id] for a in store.ancestors(n.id)} for n in nodes]
    depth = np.array([n.depth for n in nodes])
    return nodes, idx, embs, root, ancestors, depth


def average_precision(scores: np.ndarray, gold: set[int], exclude: int) -> float:
    if not gold:
        return float("nan")
    order = [i for i in np.argsort(-scores) if i != exclude]
    hits = prec = 0.0
    for rank, i in enumerate(order, 1):
        if i in gold:
            hits += 1
            prec += hits / rank
    return prec / len(gold)


def ancestor_map(embs, coords, c, ancestors, *, method: str) -> float:
    aps = []
    radius = None if coords is None else np.sqrt(c) * np.linalg.norm(coords, axis=1)
    for v, gold in enumerate(ancestors):
        if not gold:
            continue
        if method == "cosine":
            scores = embs @ embs[v]
        else:  # hyperbolic: on-branch (small d) AND more general (smaller radius)
            d = G.dist_c_batch(coords[v], coords, c)
            scores = -d + GEN_REWARD * (radius[v] - radius)
        aps.append(average_precision(scores, gold, exclude=v))
    return float(np.nanmean(aps))


def parity_auc(embs, coords, c, store, idx, root, depth, *, method: str) -> float:
    leaves = [i for i in range(len(depth)) if depth[i] == 2]
    aucs = []
    for v in leaves:
        node = store.all()[v]
        sibs = [idx[s.id] for s in store.all()
                if s.parent_id == node.parent_id and idx[s.id] != v]
        negs = [i for i in range(len(root)) if root[i] != root[v]]
        if not sibs or not negs:
            continue
        if method == "cosine":
            sc = embs @ embs[v]
        else:
            sc = -G.dist_c_batch(coords[v], coords, c)
        wins = sum(sc[p] > sc[n] for p in sibs for n in negs)
        aucs.append(wins / (len(sibs) * len(negs)))
    return float(np.mean(aucs))


def main() -> None:
    from kma.train import CHECKPOINT, train

    store = build_store().store
    nodes, idx, embs, root, ancestors, depth = _tree(store)

    rng_chart = HyperbolicChart(in_dim=embs.shape[1], dim=64)        # untrained
    rand_coords = rng_chart.encode(embs)
    rand_c = rng_chart.curvature()

    try:
        trained = HyperbolicChart.load(CHECKPOINT)
    except Exception:
        print("(no checkpoint; training a chart now...)\n")
        trained = train(verbose=False)
    tr_coords = trained.encode(embs)
    tr_c = trained.curvature()

    # (A) ancestor retrieval MAP
    map_cos = ancestor_map(embs, None, 1.0, ancestors, method="cosine")
    map_rand = ancestor_map(embs, rand_coords, rand_c, ancestors, method="hyp")
    map_tr = ancestor_map(embs, tr_coords, tr_c, ancestors, method="hyp")

    # (B) flat similarity parity AUC
    auc_cos = parity_auc(embs, None, 1.0, store, idx, root, depth, method="cosine")
    auc_tr = parity_auc(embs, tr_coords, tr_c, store, idx, root, depth, method="hyp")

    print(f"nodes={len(nodes)}  ball_dim=64  trained curvature c={tr_c:.3f}\n")
    print("(A) ANCESTOR RETRIEVAL  (is-a reconstruction, higher = better)")
    print(f"      cosine        MAP = {map_cos:.3f}")
    print(f"      random-phi    MAP = {map_rand:.3f}")
    print(f"      trained-phi   MAP = {map_tr:.3f}   <- delta vs cosine "
          f"{map_tr - map_cos:+.3f}")
    print("\n(B) FLAT SIMILARITY PARITY  (sibling vs cross-branch AUC)")
    print(f"      cosine        AUC = {auc_cos:.3f}")
    print(f"      trained-phi   AUC = {auc_tr:.3f}   <- want ~parity "
          f"({auc_tr - auc_cos:+.3f})")


if __name__ == "__main__":
    main()
