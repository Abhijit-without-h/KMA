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

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from kma import geometry as G
from kma import placement
from kma.embeddings import content_hash, get_embedder
from kma.models import MemoryNode
from kma.store import MemoryStore

BALL_DIM = 16


@dataclass
class _Snapshot:
    """Query-ready view of the store, built once per store version and shared by
    every subsequent query (and across threads) instead of rebuilt per call.

    The cheap parts (id list, matrices, cosine retriever) are built eagerly. The
    expensive parts (BM25 tokenization, the hyperbolic region index) are built
    LAZILY on first query use and memoized -- so a write-heavy workload, which
    invalidates the snapshot every insert, never pays for indices no query asked
    for, while a read-heavy workload pays for them exactly once.
    """

    version: int
    ids: list[str]
    index_map: dict[str, int]
    embs: np.ndarray
    coords: np.ndarray
    cosine: object                              # CosineRetriever (cheap, eager)
    curvature: float
    ball_dim: int
    route_threshold: int
    has_chart: bool
    texts: list[str] = field(default_factory=list)
    _bm25: object | None = None
    _region: object | None = None
    _region_built: bool = False

    @property
    def bm25(self):
        if self._bm25 is None:
            from kma.retrievers import BM25Retriever
            self._bm25 = BM25Retriever(self.ids, self.texts)
        return self._bm25

    @property
    def region_index(self):
        if not self._region_built:
            from kma.manifold import RegionIndex
            if self.has_chart and len(self.ids) >= self.route_threshold:
                self._region = RegionIndex.build(self.ids, self.coords, self.curvature)
            self._region_built = True
        return self._region


@dataclass
class Hit:
    node: MemoryNode
    score: float
    cosine: float
    hyp_to_anchor: float   # ball distance to the top cosine hit (tree-proximity)
    via: str               # "recall" | "expand" | "learned" | "ensemble"
    evidence: float = 0.0          # ensemble: confidence in [0,1] (0 otherwise)
    agreement: int = 0             # ensemble: # of retrievers that corroborated
    voters: tuple[str, ...] = ()   # ensemble: which retrievers voted for this


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
        # Query-ready snapshot, rebuilt only when the store version changes; a
        # lock guards the (rare) rebuild so concurrent queries share one build.
        self._snap: _Snapshot | None = None
        self._snap_lock = threading.Lock()
        self._reranker = None          # optional CrossEncoderReranker, set on demand
        # Append-only embedding buffer for O(1)-amortized auto-placement: inserts
        # append a row instead of restacking the whole matrix. A removal (which
        # bumps store.version out of sync) forces a one-time rebuild from store.
        self._emb_ids: list[str] = []
        self._emb_mat: np.ndarray | None = None
        self._emb_n = 0
        self._emb_version = -1

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
        self._emb_append(node.id, emb)
        return node

    # --- append-only embedding buffer (fast auto-placement) ------------------
    def _emb_append(self, node_id: str, emb: np.ndarray) -> None:
        """Append one row in amortized O(1) via capacity doubling."""
        if self._emb_version != self.store.version - 1:
            return  # buffer out of sync (e.g. after a removal); rebuilt on read
        row = np.asarray(emb, dtype=np.float64)
        if self._emb_mat is None:
            self._emb_mat = np.empty((8, row.shape[0]), dtype=np.float64)
        if self._emb_n == len(self._emb_mat):
            self._emb_mat = np.vstack([self._emb_mat, np.empty_like(self._emb_mat)])
        self._emb_mat[self._emb_n] = row
        self._emb_ids.append(node_id)
        self._emb_n += 1
        self._emb_version = self.store.version

    def embedding_matrix(self) -> tuple[list[str], np.ndarray]:
        """(ids, embeddings) for fast auto-placement. Uses the append-only buffer;
        rebuilds from the store only when it has drifted (after a remove/load)."""
        if self._emb_version != self.store.version:
            nodes = self.store.all()
            self._emb_ids = [n.id for n in nodes]
            self._emb_mat = (np.array([n.embedding for n in nodes], dtype=np.float64)
                             if nodes else None)
            self._emb_n = len(nodes)
            self._emb_version = self.store.version
        if self._emb_mat is None or self._emb_n == 0:
            return [], np.zeros((0, 1))
        return self._emb_ids, self._emb_mat[:self._emb_n]

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
        if mode == "ensemble":
            return self._query_ensemble(text, k=k, recall_k=recall_k)
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


    # --- ensemble (evidence-driven) retrieval --------------------------------
    ROUTE_THRESHOLD = 256          # below this, hyperbolic retriever scans (exact)

    def _snapshot(self) -> _Snapshot:
        """Return a query-ready snapshot, rebuilding only if the store changed.

        The fast path is lock-free: if the cached snapshot matches the current
        store version we return it directly. Only a genuine rebuild takes the
        lock, so concurrent readers don't serialize and don't each rebuild.
        """
        from kma.retrievers import CosineRetriever

        snap = self._snap
        version = self.store.version
        if snap is not None and snap.version == version:
            return snap
        with self._snap_lock:
            snap = self._snap
            if snap is not None and snap.version == self.store.version:
                return snap                         # built while we waited
            nodes = self.store.all()
            ids = [nd.id for nd in nodes]
            embs = np.array([nd.embedding for nd in nodes], dtype=np.float64) \
                if nodes else np.zeros((0, 1))
            coords = np.array([nd.ball_coord for nd in nodes], dtype=np.float64) \
                if nodes else np.zeros((0, self.ball_dim))
            snap = _Snapshot(
                version=self.store.version,
                ids=ids,
                index_map={nid: i for i, nid in enumerate(ids)},
                embs=embs,
                coords=coords,
                cosine=CosineRetriever(ids, embs),
                curvature=self.curvature,
                ball_dim=self.ball_dim,
                route_threshold=self.ROUTE_THRESHOLD,
                has_chart=self.chart is not None,
                texts=[nd.text for nd in nodes],
            )
            self._snap = snap
            return snap

    def embedding_matrix(self) -> tuple[list[str], np.ndarray]:
        """(ids, embeddings) from the cached snapshot -- for fast auto-placement."""
        snap = self._snapshot()
        return snap.ids, snap.embs

    def query_batch(self, texts: list[str], *, k: int = 5, recall_k: int = 20,
                    mode: str = "ensemble") -> list[list[Hit]]:
        """Orchestrate many queries over ONE shared snapshot + ONE batched embed.

        For multi-query traffic this is the efficient path: the corpus snapshot is
        built once, every query's text is embedded in a single batched call (the
        dominant cost with a real model), and only the per-query scoring differs.
        """
        if len(self.store) == 0 or not texts:
            return [[] for _ in texts]
        if mode != "ensemble":
            return [self.query(t, k=k, recall_k=recall_k, mode=mode) for t in texts]

        from kma import fusion
        from kma.retrievers import HyperbolicRetriever, Query

        snap = self._snapshot()
        q_embs = self.embedder.encode(list(texts))      # one batched embed call
        out: list[list[Hit]] = []
        for text, q_emb in zip(texts, q_embs):
            q_coord = self.chart.encode(q_emb)[0] if self.chart is not None else None
            query = Query(text=text, emb=q_emb, coord=q_coord)
            retrievers = [snap.cosine, snap.bm25]
            if q_coord is not None:
                retrievers.append(HyperbolicRetriever(
                    snap.ids, snap.coords, self.curvature, index=snap.region_index))
            lists = [r.search(query, recall_k) for r in retrievers]
            result = fusion.fuse(lists, n_retrievers=len(retrievers))
            cos = snap.embs @ q_emb
            out.append([
                Hit(node=self.store.get(fh.node_id), score=float(fh.evidence),
                    cosine=float(cos[snap.index_map[fh.node_id]]), hyp_to_anchor=0.0,
                    via="ensemble", evidence=float(fh.evidence),
                    agreement=int(fh.agreement), voters=fh.voters)
                for fh in result.hits[:k]
            ])
        return out

    def _query_ensemble(self, text: str, *, k: int, recall_k: int,
                        tau_evidence: float = 0.30, rerank: bool = False) -> list[Hit]:
        """Hybrid retrieval: several independent retrievers vote, RRF fuses them,
        and an evidence gate flags uncorroborated (likely hallucinated) answers.

        The manifold is just one voter here -- never the final authority. All the
        per-corpus work (matrices, BM25/region indices) comes from a cached
        snapshot, so a query only does query-sized work.
        """
        from kma import fusion
        from kma.retrievers import HyperbolicRetriever, Query

        snap = self._snapshot()
        q_emb = self.embedder.encode([text])[0]
        q_coord = self.chart.encode(q_emb)[0] if self.chart is not None else None
        query = Query(text=text, emb=q_emb, coord=q_coord)

        retrievers = [snap.cosine, snap.bm25]
        # Only trust the manifold as a voter when a chart gives a real ball coord.
        if q_coord is not None:
            retrievers.append(HyperbolicRetriever(
                snap.ids, snap.coords, self.curvature, index=snap.region_index))

        lists = [r.search(query, recall_k) for r in retrievers]
        result = fusion.fuse(lists, n_retrievers=len(retrievers),
                             tau_evidence=tau_evidence)

        if rerank and result.hits:
            if self._reranker is None:
                self._reranker = fusion.CrossEncoderReranker()
            texts = {nid: self.store.get(nid).text for nid in snap.ids}
            result.hits = self._reranker.rerank(text, result.hits, texts)

        cos = snap.embs @ q_emb
        hits: list[Hit] = []
        for fh in result.hits[:k]:
            i = snap.index_map[fh.node_id]
            hits.append(Hit(
                node=self.store.get(fh.node_id),
                score=float(fh.evidence),
                cosine=float(cos[i]),
                hyp_to_anchor=0.0,
                via="ensemble",
                evidence=float(fh.evidence),
                agreement=int(fh.agreement),
                voters=fh.voters,
            ))
        return hits


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
