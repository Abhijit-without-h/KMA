from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kma.embeddings import HashingEmbedder
from kma.engine import KMAEngine
from kma.memory import AgenticMemory

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "neural hyperbolic memory vector index region geodesic curvature manifold "
    "python rust travel hiking france paris model agent retrieval embedding "
    "capital language safety systems celebrity landmark tower mission planet"
).split()


def _sentence(rng) -> str:
    n = rng.integers(6, 14)
    return " ".join(rng.choice(_WORDS, size=n))


def build_memory(n: int, seed: int = 0) -> tuple[AgenticMemory, float, list[str]]:
    """Insert n synthetic memories; return (mem, build_seconds, query_texts)."""
    rng = np.random.default_rng(seed)
    eng = KMAEngine()
    eng.embedder = HashingEmbedder()
    mem = AgenticMemory(engine=eng)
    t0 = time.perf_counter()
    for _ in range(n):
        mem.add(_sentence(rng))
    build_s = time.perf_counter() - t0
    queries = [_sentence(rng) for _ in range(256)]
    return mem, build_s, queries


def _percentiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    s = sorted(latencies_ms)

    def pct(p):
        return s[min(len(s) - 1, int(p / 100 * len(s)))]

    return pct(50), pct(95), pct(99)


def run_load(
    mem: AgenticMemory, queries: list[str], *, mode: str, concurrency: int, total: int
) -> dict:
    """Fire `total` queries across `concurrency` threads; collect latencies."""
    lat: list[float] = []

    def one(i: int):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        mem.search(q, mode=mode, k=5)
        return (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    if concurrency == 1:
        lat = [one(i) for i in range(total)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            lat = list(pool.map(one, range(total)))
    wall = time.perf_counter() - t0
    p50, p95, p99 = _percentiles(lat)
    return {"qps": total / wall, "p50": p50, "p95": p95, "p99": p99, "mean": statistics.mean(lat)}


def run_batch(mem: AgenticMemory, queries: list[str], total: int) -> float:
    """Throughput of the batched orchestration path (one shared snapshot)."""
    qs = [queries[i % len(queries)] for i in range(total)]
    t0 = time.perf_counter()
    mem.search_batch(qs, mode="ensemble", k=5)
    return total / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="500,2000,10000")
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--mode", default="ensemble", choices=["ensemble", "heuristic"])
    ap.add_argument("--total", type=int, default=512)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    concs = [int(c) for c in args.concurrency.split(",")]

    print(f"\nKMA load test   mode={args.mode}   {args.total} queries/run " f"(hashing embedder)")
    print("=" * 76)
    for n in sizes:
        mem, build_s, queries = build_memory(n)
        mem.warmup()  # pay the one-time index build up front
        print(f"\nN={n:,}   built in {build_s:.2f}s " f"({n/build_s:,.0f} writes/s)")
        print(f"  {'concurrency':>11} {'throughput':>12} {'p50':>9} " f"{'p95':>9} {'p99':>9}")
        for c in concs:
            r = run_load(mem, queries, mode=args.mode, concurrency=c, total=args.total)
            print(
                f"  {c:>11} {r['qps']:>9,.0f} q/s {r['p50']:>7.2f}ms "
                f"{r['p95']:>7.2f}ms {r['p99']:>7.2f}ms"
            )
        batch_qps = run_batch(mem, queries, args.total)
        print(f"  {'batch':>11} {batch_qps:>9,.0f} q/s  (shared snapshot + " f"one batched embed)")


if __name__ == "__main__":
    main()
