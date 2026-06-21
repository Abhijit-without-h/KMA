"""KMAEngine — insertion + hybrid retrieval.

Honest design note: the n-D ball DIRECTION is a degraded random projection of
the embedding, so hyperbolic distance to the *query* is a WORSE similarity
signal than cosine. We do NOT use it as a similarity metric. What the ball
faithfully encodes is the GIVEN tree (every parent->child edge ~= STEP), so
node-to-node hyperbolic distance is a real *structural* signal cosine lacks.

Hybrid retrieval:
  stage 1  RECALL   : flat cosine -> top `recall_k` candidates. The best one
                      is the ANCHOR (cosine is best at picking the match).
  stage 2  STRUCT   : add a structural bonus = closeness (in the ball) to the
                      anchor, surfacing the anchor's tree-relatives that cosine
                      ranked below k.
  stage 3  EXPAND   : optionally fold in the anchor's whole branch explicitly.

Set alpha=1, beta=0, expand=False to recover the pure-embedding baseline, so
the A/B comparison in eval.py is honest.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from kma import geometry as G
from kma import placement
from kma.embeddings import content_hash, get_embedder
from kma.models import MemoryNode
from kma.store import MemoryStore

BALL_DIM = 16


@dataclass
class Hit:
    node: MemoryNode
    score: float
    cosine: float
    hyp_to_anchor: float   # ball distance to the top cosine hit (tree-proximity)
    via: str               # "recall" or "expand"


class KMAEngine:
    def __init__(self, ball_dim: int = BALL_DIM, chart=None) -> None:
        # `chart` is an optional trained HyperbolicChart. When present, nodes are
        # placed by the learned phi (semantics AND hierarchy) instead of the
        # heuristic random projection, and learned-mode retrieval becomes useful.
        self.chart = chart
        self.curvature = float(chart.curvature()) if chart is not None else 1.0
        self.ball_dim = chart.dim if chart is not None else ball_dim
        self.store = MemoryStore()
        self.embedder = get_embedder()

    # --- insertion -----------------------------------------------------------
    def insert(
        self,
        text: str,
        parent_id: str | None = None,
        *,
        topic_label: str | None = None,
        source: str = "chat",
        metadata: dict | None = None,
        embedding: np.ndarray | None = None,
    ) -> MemoryNode:
        emb = self.embedder.encode([text])[0] if embedding is None else embedding
        parent = self.store.get(parent_id) if parent_id else None
        depth = (parent.depth + 1) if parent else 0
        if self.chart is not None:
            coord = self.chart.encode(emb)[0]              # learned placement
        else:
            parent_coord = parent.coord if parent else None
            coord = placement.place(emb, self.ball_dim, parent_coord)

        node = MemoryNode(
            id=str(uuid.uuid4()),
            text=text,
            content_hash=content_hash(text),
            embedding=emb.tolist(),
            ball_coord=coord.tolist(),
            depth=depth,
            parent_id=parent_id,
            topic_label=topic_label,
            source=source,
            metadata=metadata or {},
        )
        self.store.add(node)
        return node

    # --- retrieval -----------------------------------------------------------
    def query(
        self,
        text: str,
        *,
        k: int = 5,
        recall_k: int = 20,
        alpha: float = 0.6,
        beta: float = 0.4,
        expand: bool = True,
        mode: str = "heuristic",
    ) -> list[Hit]:
        if len(self.store) == 0:
            return []
        if mode == "learned":
            return self._query_learned(text, k=k, alpha=alpha, beta=beta)

        q_emb = self.embedder.encode([text])[0]
        ids, embs, coords = self.store.matrices()
        index = {nid: i for i, nid in enumerate(ids)}

        cos = embs @ q_emb                       # both L2-normalized -> cosine

        # stage 1: recall by cosine; the best hit is the structural anchor.
        order = np.argsort(-cos)[:recall_k]
        recall_ids = {ids[i] for i in order}
        anchor_i = int(order[0])
        anchor_coord = coords[anchor_i]

        # node-to-anchor hyperbolic distance encodes tree proximity.
        to_anchor = G.dist_batch(anchor_coord, coords)

        cand_ids = set(recall_ids)
        # stage 3: explicitly fold in the anchor's whole branch.
        if expand:
            cand_ids |= {n.id for n in self.store.branch(ids[anchor_i])}

        # stage 2: blend cosine (similarity) with structural closeness to anchor.
        cos_n = _normalize(cos)
        struct = 1.0 - _normalize(to_anchor)
        hits: list[Hit] = []
        for nid in cand_ids:
            i = index[nid]
            score = alpha * cos_n[i] + beta * struct[i]
            hits.append(
                Hit(
                    node=self.store.get(nid),
                    score=float(score),
                    cosine=float(cos[i]),
                    hyp_to_anchor=float(to_anchor[i]),
                    via="recall" if nid in recall_ids else "expand",
                )
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def _query_learned(self, text: str, *, k: int, alpha: float, beta: float) -> list[Hit]:
        """Learned-mode retrieval: trained phi makes hyperbolic distance a real
        signal, so we score by it directly, plus an asymmetric *generality* term
        (reward candidates more general than the query) that cosine cannot express.
        """
        if self.chart is None:
            raise ValueError("learned mode requires a trained chart")
        c = self.curvature
        sc = np.sqrt(c)
        q_emb = self.embedder.encode([text])[0]
        q_coord = self.chart.encode(q_emb)[0]
        ids, embs, coords = self.store.matrices()

        cos = embs @ q_emb
        hyp = G.dist_c_batch(q_coord, coords, c)
        # generality: candidate radius smaller than the query's => more general.
        node_r = sc * np.linalg.norm(coords, axis=1)
        gen = np.clip(sc * float(np.linalg.norm(q_coord)) - node_r, 0.0, None)

        sim = 1.0 - _normalize(hyp)
        score = alpha * sim + beta * _normalize(gen) + (1.0 - alpha - beta) * _normalize(cos)
        order = np.argsort(-score)[:k]
        return [
            Hit(node=self.store.get(ids[i]), score=float(score[i]),
                cosine=float(cos[i]), hyp_to_anchor=float(hyp[i]), via="learned")
            for i in order
        ]


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
