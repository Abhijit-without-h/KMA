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

    def explain(self) -> str:
        trail = " > ".join(p[:40] for p in self.path) or "(root)"
        return f"{self.node.text}   [via: {trail}]"


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
            out.append(Recall(node=h.node, score=score, path=self._path(h.node)))
        out.sort(key=lambda r: -r.score)
        return out[:k]

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
        candidates = [n for n in self.engine.store.all() if self._in_scope(n, scope)]
        if not candidates:
            return None
        sims = np.array([float(np.dot(emb, n.emb)) for n in candidates])
        best = int(np.argmax(sims))
        return candidates[best].id if sims[best] >= self.tau else None

    def _in_scope(self, node: MemoryNode, scope: dict | None) -> bool:
        if not scope:
            return True
        ns = node.metadata.get("scope", {})
        return all(ns.get(key) == val for key, val in scope.items())

    def _path(self, node: MemoryNode) -> list[str]:
        anc = self.engine.store.ancestors(node.id)        # parent -> ... -> root
        return [a.text for a in reversed(anc)]            # root -> ... -> parent
