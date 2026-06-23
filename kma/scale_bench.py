from __future__ import annotations

import argparse
import time

import numpy as np

from kma import geometry as G
from kma import placement
from kma.manifold import RegionIndex


# ---------------------------------------------------------------------------
# synthetic hierarchical embeddings -> real ball coordinates
# ---------------------------------------------------------------------------
def make_hierarchy(n_target: int, emb_dim: int = 64, ball_dim: int = 16,
                   seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a 3-level taxonomy of ~n_target leaves and place it in the ball.

    Topics are random centers; subtopics and leaves are noised children, so the
    embeddings are genuinely clustered/hierarchical (what real data looks like).
    Each node is placed with kma.placement (parent-stepping), giving real
    hierarchical Poincare coordinates. Returns (ids, embeddings, coords).
    """
    rng = np.random.default_rng(seed)
    n_topics = max(2, int(round(n_target ** (1 / 3))))
    n_sub = max(2, int(round((n_target / n_topics) ** 0.5)))
    n_leaf = max(2, int(round(n_target / (n_topics * n_sub))))

    def _unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    ids: list[str] = []
    embs: list[np.ndarray] = []
    coords: list[np.ndarray] = []

    for t in range(n_topics):
        tc = rng.standard_normal(emb_dim)
        t_emb = _unit(tc)
        t_coord = placement.place(t_emb, ball_dim, None)
        ids.append(f"t{t}"); embs.append(t_emb); coords.append(t_coord)
        for s in range(n_sub):
            sc = tc + 0.6 * rng.standard_normal(emb_dim)
            s_emb = _unit(sc)
            s_coord = placement.place(s_emb, ball_dim, t_coord)
            ids.append(f"t{t}.s{s}"); embs.append(s_emb); coords.append(s_coord)
            for leaf in range(n_leaf):
                lc = sc + 0.3 * rng.standard_normal(emb_dim)
                l_emb = _unit(lc)
                l_coord = placement.place(l_emb, ball_dim, s_coord)
                ids.append(f"t{t}.s{s}.l{leaf}")
                embs.append(l_emb); coords.append(l_coord)

    return ids, np.array(embs), np.array(coords)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def exact_topk(q_coord, coords, c, k):
    d = G.dist_c_batch(q_coord, coords, c)
    return set(np.argsort(d)[:k].tolist())


def recall_at_k(index: RegionIndex, queries, coords, c, k, nprobe):
    """Mean overlap of routed top-k with exact top-k over the query set."""
    id_to_row = {nid: i for i, nid in enumerate(index.ids)}
    recalls = []
    touched = []
    for qi in queries:
        gold = exact_topk(coords[qi], coords, c, k)
        cand = index.route(coords[qi], nprobe=nprobe)
        touched.append(len(cand) / len(coords))
        approx_ids = index.search(coords[qi], k=k, nprobe=nprobe)
        approx = {id_to_row[nid] for nid, _ in approx_ids}
        recalls.append(len(approx & gold) / k)
    return float(np.mean(recalls)), float(np.mean(touched))


def _time(fn, repeats):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------
def run(sizes, k, nprobe, n_queries, seed=0):
    c = 1.0
    rng = np.random.default_rng(seed)

    print(f"\nRegionIndex fidelity + latency   (k={k}, nprobe={nprobe}, "
          f"{n_queries} queries/size)")
    print("=" * 78)
    print(f"{'N':>7} {'regions':>8} {'recall@k':>10} {'touched':>9} "
          f"{'full ms':>9} {'routed ms':>10} {'speedup':>8}")
    print("-" * 78)

    rows = []
    for n in sizes:
        ids, _embs, coords = make_hierarchy(n, seed=seed)
        n_real = len(ids)
        index = RegionIndex.build(ids, coords, c)
        q = rng.choice(n_real, size=min(n_queries, n_real), replace=False)

        recall, touched = recall_at_k(index, q, coords, c, k, nprobe)

        qc = coords[q[0]]
        t_full = _time(lambda: np.argsort(G.dist_c_batch(qc, coords, c))[:k],
                       repeats=20)
        t_routed = _time(lambda: index.search(qc, k=k, nprobe=nprobe), repeats=20)
        speedup = t_full / max(t_routed, 1e-9)

        print(f"{n_real:>7} {index.k:>8} {recall:>10.3f} {touched:>8.1%} "
              f"{t_full*1e3:>9.3f} {t_routed*1e3:>10.3f} {speedup:>7.1f}x")
        rows.append((n_real, recall, t_full, t_routed, speedup))

    # sub-linearity check: latency growth vs N growth between first and last size.
    if len(rows) >= 2:
        (n0, _, f0, r0, _), (n1, _, f1, r1, _) = rows[0], rows[-1]
        print("-" * 78)
        nf = n1 / n0
        print(f"corpus grew {nf:.0f}x  ({n0} -> {n1}):")
        print(f"  full-scan latency grew {f1/f0:6.1f}x   (linear would be ~{nf:.0f}x)")
        print(f"  routed    latency grew {r1/r0:6.1f}x   (sub-linear is the win)")

    return rows


def nprobe_sweep(n, k, seed=0):
    c = 1.0
    ids, _e, coords = make_hierarchy(n, seed=seed)
    index = RegionIndex.build(ids, coords, c)
    rng = np.random.default_rng(seed + 1)
    q = rng.choice(len(ids), size=min(200, len(ids)), replace=False)
    print(f"\nnprobe sweep at N={len(ids)}  (k={k}, regions={index.k}) "
          f"-- the recall<->speed knob")
    print("-" * 50)
    print(f"{'nprobe':>7} {'recall@k':>10} {'touched':>9}")
    for npb in [1, 2, 3, 5, 8, index.k]:
        recall, touched = recall_at_k(index, q, coords, c, k, npb)
        label = f"{npb}" + (" (all)" if npb == index.k else "")
        print(f"{label:>7} {recall:>10.3f} {touched:>8.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="200,500,1000,2000,5000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=3)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    rows = run(sizes, args.k, args.nprobe, args.queries)
    nprobe_sweep(sizes[len(sizes) // 2], args.k)

    print("\nReading this: recall@k near 1.0 means routing keeps the right answers;")
    print("'touched' is the fraction of memories scored (lower = cheaper). The gap")
    print("between full-scan and routed latency growth is the sub-linear payoff.")


if __name__ == "__main__":
    main()
