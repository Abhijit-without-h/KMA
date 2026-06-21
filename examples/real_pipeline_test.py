"""End-to-end effectiveness test on a REAL stack.

  embeddings : sentence-transformers (KMA_ST_MODEL; default MiniLM).
               Swap to google/embeddinggemma-300m once you accept its HF license
               and install sentence-transformers>=5.0:  KMA_ST_MODEL=google/embeddinggemma-300m
  vector DB  : Pinecone (serverless) -- the baseline a real vector DB gives you.
  LLM        : OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free) -- the agent.

It answers two questions:
  1. Retrieval effectiveness: does KMA beat a real Pinecone cosine index on
     hierarchical (ancestor) retrieval, while holding parity on flat similarity?
  2. Memory effectiveness: does nemotron answer better WITH KMA memory than without?

Requires env: OPENROUTER_API_KEY, PINECONE_API_KEY.
    python examples/real_pipeline_test.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from the project .env (no dependency, no overwrite)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())
    # huggingface_hub reads HF_TOKEN; mirror it for older clients too.
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


load_dotenv()

import numpy as np

from kma.data import build_store
from kma.embeddings import get_embedder
from kma.eval import _tree, ancestor_map, parity_auc
from kma.extract import LLMFactExtractor
from kma.memory import AgenticMemory
from kma.train import train

OR_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
INDEX_NAME = "kma-bench"


def or_chat(messages: list[dict], retries: int = 5) -> str:
    """Call OpenRouter, with backoff for the :free model's rate limits / errors."""
    body = json.dumps({"model": OR_MODEL, "messages": messages}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            # error payload (often free-tier rate limit) -> back off and retry
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503) or attempt == retries - 1:
                raise
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("OpenRouter: no completion after retries (free-tier rate limit?)")


def pinecone_baseline(ids, embs, ancestors, store, idx, root, depth):
    """Build a real Pinecone index, then score ancestor MAP + sibling AUC from it."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    try:
        names = pc.list_indexes().names()
    except Exception:  # noqa: BLE001
        names = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME in names:
        pc.delete_index(INDEX_NAME)
        time.sleep(3)

    pc.create_index(INDEX_NAME, dimension=int(embs.shape[1]), metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    for _ in range(60):
        if pc.describe_index(INDEX_NAME).status["ready"]:
            break
        time.sleep(1)
    index = pc.Index(INDEX_NAME)

    try:
        index.upsert(vectors=[{"id": ids[i], "values": embs[i].tolist()}
                              for i in range(len(ids))])
        # poll until all vectors are queryable (serverless is eventually consistent)
        for _ in range(40):
            if index.describe_index_stats().get("total_vector_count", 0) >= len(ids):
                break
            time.sleep(2)
        time.sleep(2)

        sim, order = {}, {}
        for v in range(len(ids)):
            res = index.query(vector=embs[v].tolist(), top_k=len(ids), include_values=False)
            matches = res["matches"]
            order[v] = [m["id"] for m in matches if m["id"] != ids[v]]
            sim[v] = {m["id"]: m["score"] for m in matches}

        # ancestor MAP from Pinecone's ranking
        aps = []
        for v, gold in enumerate(ancestors):
            if not gold:
                continue
            gold_ids = {ids[g] for g in gold}
            hits = prec = 0.0
            for rank, mid in enumerate(order[v], 1):
                if mid in gold_ids:
                    hits += 1
                    prec += hits / rank
            aps.append(prec / len(gold_ids))
        pmap = float(np.mean(aps))

        # sibling-vs-cross-branch AUC from Pinecone scores
        leaves = [i for i in range(len(depth)) if depth[i] == 2]
        aucs = []
        for v in leaves:
            node = store.all()[v]
            sibs = [idx[s.id] for s in store.all()
                    if s.parent_id == node.parent_id and idx[s.id] != v]
            negs = [i for i in range(len(root)) if root[i] != root[v]]
            if not sibs or not negs:
                continue
            sv = sim[v]
            wins = sum(sv.get(ids[p], -9) > sv.get(ids[n], -9) for p in sibs for n in negs)
            aucs.append(wins / (len(sibs) * len(negs)))
        pauc = float(np.mean(aucs))
        return pmap, pauc
    finally:
        pc.delete_index(INDEX_NAME)  # keep the free tier clean


def retrieval_effectiveness():
    store = build_store().store
    nodes, idx, embs, root, ancestors, depth = _tree(store)
    ids = [n.id for n in nodes]
    emb = get_embedder()
    print(f"embedder : {emb.name} (dim={emb.dim})   corpus: {len(ids)} nodes\n")

    print("running Pinecone baseline (create index -> upsert -> query)...")
    pmap, pauc = pinecone_baseline(ids, embs, ancestors, store, idx, root, depth)

    print("training KMA hyperbolic chart...")
    chart = train(verbose=False)
    coords, c = chart.encode(embs), chart.curvature()
    kmap = ancestor_map(embs, coords, c, ancestors, method="hyp")
    kauc = parity_auc(embs, coords, c, store, idx, root, depth, method="hyp")

    print("\n================ RETRIEVAL EFFECTIVENESS ================")
    print(f"{'':30}{'Pinecone (cosine)':>20}{'KMA (hyperbolic)':>20}")
    print(f"{'ancestor MAP (hierarchy)':30}{pmap:>20.3f}{kmap:>20.3f}")
    print(f"{'flat similarity AUC':30}{pauc:>20.3f}{kauc:>20.3f}")
    print(f"\nhierarchy lift: {kmap - pmap:+.3f} MAP   |   flat parity delta: {kauc - pauc:+.3f}")


def memory_effectiveness():
    print("\n================ MEMORY EFFECTIVENESS (nemotron) ================")
    mem = AgenticMemory()
    extractor = LLMFactExtractor(or_chat)
    scope = {"user_id": "demo"}
    turns = [
        "hey so I'm building a travel app aimed at serious mountain hikers",
        "I'm writing the backend in python with fastapi, and postgres for data",
        "big thing: it has to work offline so hikers get maps with no signal",
    ]
    print("ingesting turns via nemotron fact extraction...")
    for t in turns:
        for r in mem.ingest(t, extractor=extractor, scope=scope):
            print(f"  + remembered: {r.node.text}")

    q = "Given what you know about me, what should I build next and why?"
    ctx = mem.get_context(q, scope=scope)
    print(f"\nquery: {q}\n--- memory injected ---\n{ctx}")
    with_mem = or_chat([{"role": "system", "content": "Use this memory about the user:\n" + ctx},
                        {"role": "user", "content": q}])
    without = or_chat([{"role": "user", "content": q}])
    print(f"\n[WITH KMA memory]\n{with_mem[:600]}")
    print(f"\n[WITHOUT memory]\n{without[:600]}")


if __name__ == "__main__":
    retrieval_effectiveness()
    memory_effectiveness()
