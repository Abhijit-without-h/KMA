from __future__ import annotations

import time

import numpy as np

from kma import geometry as G
from kma.data import build_store
from kma.eval import ancestor_map, parity_auc, _tree
from kma.train import train


def _time_per_call(fn, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    store = build_store().store
    nodes, idx, embs, root, ancestors, depth = _tree(store)
    eng_embedder = _embedder()
    n = len(nodes)

    # ---- quality: baseline (cosine / vector-DB) vs KMA (trained hyperbolic) ----
    map_cos = ancestor_map(embs, None, 1.0, ancestors, method="cosine")
    auc_cos = parity_auc(embs, None, 1.0, store, idx, root, depth, method="cosine")

    chart = train(verbose=False)
    coords = chart.encode(embs)
    c = chart.curvature()
    map_kma = ancestor_map(embs, coords, c, ancestors, method="hyp")
    auc_kma = parity_auc(embs, coords, c, store, idx, root, depth, method="hyp")

    # ---- latency ----
    sample = [nd.text for nd in nodes[: min(8, n)]]
    t_embed = _time_per_call(lambda: eng_embedder.encode(sample), 3) / len(sample)
    t_phi = _time_per_call(lambda: chart.encode(embs), 3) / n
    q = embs[0]
    qc = coords[0]
    t_cos = _time_per_call(lambda: embs @ q, 50)
    t_hyp = _time_per_call(lambda: G.dist_c_batch(qc, coords, c), 50)

    # ---- report ----
    print(f"embedder   : {eng_embedder.name}  (dim={eng_embedder.dim})")
    print(f"corpus     : {n} nodes   |   ball_dim=64  curvature c={c:.3f}\n")
    print(f"{'':32}{'baseline (emb+cosine)':>22}{'KMA (hyperbolic)':>20}")
    print(f"{'(A) ancestor MAP (hierarchy)':32}{map_cos:>22.3f}{map_kma:>20.3f}")
    print(f"{'(B) flat similarity AUC':32}{auc_cos:>22.3f}{auc_kma:>20.3f}")
    print("\nlatency (seconds)")
    print(f"  embedding / item (unavoidable, both) : {t_embed*1e3:8.3f} ms")
    print(f"  KMA projection phi / item            : {t_phi*1e6:8.1f} us"
          f"   ({100*t_phi/max(t_embed,1e-12):.2f}% of embedding)")
    print(f"  retrieval over {n} nodes  cosine      : {t_cos*1e6:8.1f} us")
    print(f"  retrieval over {n} nodes  hyperbolic  : {t_hyp*1e6:8.1f} us"
          f"   ({t_hyp/max(t_cos,1e-12):.1f}x cosine)")
    print("\nNote: the baseline column == what a vector DB (Pinecone/Chroma) returns;")
    print("KMA adds the hierarchy/generality layer on the SAME embeddings.")


def _embedder():
    from kma.embeddings import get_embedder

    return get_embedder()


if __name__ == "__main__":
    main()
