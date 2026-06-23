"""AgenticMemory — the agent-facing facade over the KMA engine.

This is what an LLM agent (or the MCP server) actually calls. It adds the pieces
that turn a retrieval engine into an *agentic memory*:

  * add()          auto-placement -- a new memory is attached under its nearest
                   existing concept automatically (no manual parent_id), so the
                   hierarchy grows on its own as the agent talks.
  * search()       scoped retrieval with PROVENANCE (the ancestor path that
                   explains *why* a memory was recalled).
  * get_context()  assemble a compact, prompt-ready memory block (top hits rolled
                   up with their ancestors) within a character budget.
  * update()/forget()  revise or remove a memory (children are reparented).

Scope (user_id / agent_id / session_id) is a metadata filter so one store can
serve many agents/sessions. All heavy lifting is delegated to KMAEngine; this
layer is pure-Python and dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kma.engine import KMAEngine
from kma.models import MemoryNode

# Cosine threshold for attaching a new memory as a child of its nearest node.
# Below it, the memory starts a new root concept. Tuned for short MiniLM text.
DEFAULT_TAU = 0.35


@dataclass
class Recall:
    node: MemoryNode
    score: float
    path: list[str]          # root -> ... -> parent text (provenance / roll-up)
    evidence: float = 0.0          # ensemble confidence in [0,1] (0 in other modes)
    voters: tuple[str, ...] = ()   # which retrievers corroborated this hit

    def explain(self) -> str:
        trail = " > ".join(p[:40] for p in self.path) or "(root)"
        tag = f"   [via: {trail}]"
        if self.voters:
            tag += f"   [evidence {self.evidence:.2f}, voters: {','.join(self.voters)}]"
        return f"{self.node.text}{tag}"


class AgenticMemory:
    def __init__(self, engine: KMAEngine | None = None, *, tau: float = DEFAULT_TAU) -> None:
        self.engine = engine or KMAEngine()
        self.tau = tau

    # --- write ---------------------------------------------------------------
    def add(self, text: str, *, scope: dict | None = None, source: str = "chat",
            importance: float = 0.5) -> Recall:
        emb = self.engine.embedder.encode([text])[0]
        parent_id = self._auto_parent(emb, scope)
        node = self.engine.insert(
            text, parent_id=parent_id, source=source,
            metadata={"scope": scope or {}}, embedding=emb,
        )
        node.importance = importance
        return Recall(node=node, score=1.0, path=self._path(node))

    def ingest(self, text: str, *, extractor, scope: dict | None = None,
               source: str = "chat", importance: float = 0.5) -> list[Recall]:
        """Distill `text` into atomic facts (via an extractor) and store each.

        Falls back to storing nothing if the extractor finds nothing worth
        keeping -- that's the point, vs. dumping the raw turn into memory.
        """
        facts = extractor.extract(text)
        return [self.add(f, scope=scope, source=source, importance=importance)
                for f in facts]

    def update(self, node_id: str, text: str) -> Recall | None:
        node = self.engine.store.get(node_id)
        if node is None:
            return None
        from kma.embeddings import content_hash

        emb = self.engine.embedder.encode([text])[0]
        node.text = text
        node.content_hash = content_hash(text)
        node.embedding = emb.tolist()
        node.version += 1
        return Recall(node=node, score=1.0, path=self._path(node))

    def forget(self, node_id: str) -> bool:
        if self.engine.store.get(node_id) is None:
            return False
        self.engine.store.remove(node_id)
        return True

    def consolidate(self, *, sim_fold: float = 0.98) -> int:
        """Fold near-duplicate memories together (amortized self-organization).

        Memories whose embeddings are at least `sim_fold` cosine-similar are
        merged: the more general one (smaller radius) survives and absorbs the
        duplicate's text into its metadata. Returns the number folded away. This
        is offline maintenance -- it is never run on the query path.
        """
        from kma.manifold import find_folds_by_similarity

        nodes = self.engine.store.all()
        if len(nodes) < 2:
            return 0
        ids = [n.id for n in nodes]
        embs = np.array([n.embedding for n in nodes], dtype=np.float64)
        coords = np.array([n.ball_coord for n in nodes], dtype=np.float64)
        folds = find_folds_by_similarity(ids, embs, coords,
                                         self.engine.curvature, sim_fold)
        folded = 0
        for survivor_id, absorbed_id in folds:
            survivor = self.engine.store.get(survivor_id)
            absorbed = self.engine.store.get(absorbed_id)
            if survivor is None or absorbed is None:
                continue
            survivor.metadata.setdefault("folded", []).append(absorbed.text)
            self.engine.store.remove(absorbed_id)
            folded += 1
        return folded

    # --- read ----------------------------------------------------------------
    def search(self, query: str, *, scope: dict | None = None, k: int = 5,
               mode: str = "heuristic") -> list[Recall]:
        if len(self.engine.store) == 0:
            return []
        hits = self.engine.query(query, k=max(k * 4, 12), mode=mode)
        out: list[Recall] = []
        for h in hits:
            if not self._in_scope(h.node, scope):
                continue
            # light importance tie-breaker on top of the engine score
            score = h.score + 0.1 * h.node.importance
            out.append(Recall(node=h.node, score=score, path=self._path(h.node),
                              evidence=h.evidence, voters=h.voters))
        out.sort(key=lambda r: -r.score)
        return out[:k]

    def search_batch(self, queries: list[str], *, scope: dict | None = None,
                     k: int = 5, mode: str = "ensemble") -> list[list[Recall]]:
        """Run many queries efficiently over one shared snapshot (see
        KMAEngine.query_batch). Returns one ranked Recall list per query."""
        if len(self.engine.store) == 0 or not queries:
            return [[] for _ in queries]
        batches = self.engine.query_batch(queries, k=max(k * 4, 12), mode=mode)
        results: list[list[Recall]] = []
        for hits in batches:
            out = [Recall(node=h.node, score=h.score + 0.1 * h.node.importance,
                          path=self._path(h.node), evidence=h.evidence,
                          voters=h.voters)
                   for h in hits if self._in_scope(h.node, scope)]
            out.sort(key=lambda r: -r.score)
            results.append(out[:k])
        return results

    def warmup(self) -> None:
        """Pre-build the query snapshot (matrices + BM25 + region index) so the
        first user query doesn't pay the one-time build cost. Call after bulk load."""
        snap = self.engine._snapshot()
        _ = snap.bm25
        _ = snap.region_index

    def get_context(self, query: str, *, scope: dict | None = None, k: int = 5,
                    budget_chars: int = 1500, mode: str = "heuristic") -> str:
        """Prompt-ready memory block: top hits rolled up with their ancestors."""
        lines: list[str] = []
        seen: set[str] = set()
        used = 0
        for r in self.search(query, scope=scope, k=k, mode=mode):
            chain = [*r.path, r.node.text]                 # general -> specific
            for text in chain:
                if text in seen:
                    continue
                line = f"- {text}"
                if used + len(line) > budget_chars:
                    return "\n".join(lines)
                lines.append(line)
                seen.add(text)
                used += len(line)
        return "\n".join(lines)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self.engine.store.save(path)

    def load(self, path: str | Path) -> None:
        if Path(path).exists():
            self.engine.store.load(path)

    # --- internals -----------------------------------------------------------
    def _auto_parent(self, emb: np.ndarray, scope: dict | None) -> str | None:
        ids, embs = self.engine.embedding_matrix()      # cached snapshot matrix
        if len(ids) == 0:
            return None
        sims = embs @ emb                                # one vectorized matmul
        if not scope:
            best = int(np.argmax(sims))
            return ids[best] if sims[best] >= self.tau else None
        # scoped: mask out-of-scope candidates before picking the best.
        store = self.engine.store
        best_i, best_s = -1, -1.0
        for i, nid in enumerate(ids):
            if sims[i] > best_s and self._in_scope(store.get(nid), scope):
                best_i, best_s = i, float(sims[i])
        return ids[best_i] if best_i >= 0 and best_s >= self.tau else None

    def _in_scope(self, node: MemoryNode, scope: dict | None) -> bool:
        if not scope:
            return True
        ns = node.metadata.get("scope", {})
        return all(ns.get(key) == val for key, val in scope.items())

    def _path(self, node: MemoryNode) -> list[str]:
        anc = self.engine.store.ancestors(node.id)        # parent -> ... -> root
        return [a.text for a in reversed(anc)]            # root -> ... -> parent
