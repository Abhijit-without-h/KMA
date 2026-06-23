"""Candidate generators for evidence-driven retrieval.

Each retriever is an independent "voter": given a query it returns a ranked list
of candidates. They disagree in useful ways -- cosine catches paraphrase, BM25
catches exact terms/names, the hyperbolic manifold catches structural/general
relatives. fusion.py combines their votes and only trusts a result that several
retrievers corroborate (this is the anti-hallucination move).

All retrievers share one interface:
    .name
    .search(query: Query, k: int) -> list[Candidate]
and are built from plain arrays so they are unit-testable without a store.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kma import geometry as G
from kma.manifold import RegionIndex

_TOKEN = re.compile(r"[a-z0-9]+")

# Lexical stopwords. Critical for the evidence gate: without this, two retrievers
# can "corroborate" a hit purely on a shared stopword ("and", "the", ...) and a
# completely unrelated query would look confidently answered. Corroboration must
# happen on content terms.
_STOPWORDS = frozenset("""
a an the of to in on at for and or but is are was were be been being this that
these those it its as by with from into about over under then than so such not no
do does did has have had will would can could should may might what which who whom
whose where when why how their there here they them he she you your our we us i me my
""".split())


def _tokenize(text: str, *, drop_stop: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stop:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


@dataclass
class Query:
    """Everything the retrievers might need about a query, computed once."""

    text: str
    emb: NDArray                      # L2-normalized embedding
    coord: NDArray | None = None     # ball coordinate (only if a chart exists)


@dataclass
class Candidate:
    node_id: str
    score: float                     # retriever-native score (higher = better)
    rank: int                        # 0-based rank within this retriever
    retriever: str


@dataclass
class CosineRetriever:
    """Flat semantic similarity -- the paraphrase catcher."""

    ids: list[str]
    embs: NDArray
    name: str = "cosine"

    def search(self, query: Query, k: int) -> list[Candidate]:
        s = self.embs @ query.emb
        order = np.argsort(-s)[:k]
        return [Candidate(self.ids[i], float(s[i]), r, self.name)
                for r, i in enumerate(order)]


class BM25Retriever:
    """Pure-python BM25 lexical search -- the exact-term / proper-noun catcher.

    Lexical retrieval is the cheapest hallucination guard: if the answer hinges
    on a name or number, an embedding may drift but BM25 will not.
    """

    name = "bm25"

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.ids = list(ids)
        self.k1, self.b = k1, b
        docs = [_tokenize(t, drop_stop=True) for t in texts]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        n = len(docs)
        # Inverted index: term -> postings [(doc_idx, term_freq)]. Built once;
        # search then touches ONLY documents containing a query term, instead of
        # scanning all N docs per query (the difference between O(N) and O(hits)).
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for t, f in Counter(doc).items():
                self.postings.setdefault(t, []).append((i, f))
        # BM25+ idf (always positive), avoids negative weights on common terms.
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}
        self._n = n

    def search(self, query: Query, k: int) -> list[Candidate]:
        if self._n == 0:
            return []
        scores: dict[int, float] = {}
        denom_const = self.k1 * (1 - self.b)
        for t in set(_tokenize(query.text, drop_stop=True)):
            postings = self.postings.get(t)
            if postings is None:
                continue
            idf = self.idf[t]
            for i, f in postings:
                denom = f + denom_const + self.k1 * self.b * self.doc_len[i] / (self.avgdl + 1e-9)
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [Candidate(self.ids[i], float(s), r, self.name)
                for r, (i, s) in enumerate(top)]


@dataclass
class HyperbolicRetriever:
    """Structure/generality retriever over the manifold -- the relatives catcher.

    Routes the query through a RegionIndex (sub-linear) when one is supplied,
    else scans all coords. Requires a query ball-coord, so the engine only
    enables it when a trained chart is present (heuristic ball coords are a
    degraded signal and would just add noise to the vote).
    """

    ids: list[str]
    coords: NDArray
    c: float
    index: RegionIndex | None = None
    nprobe: int = 3
    name: str = "hyperbolic"

    def search(self, query: Query, k: int) -> list[Candidate]:
        if query.coord is None or len(self.ids) == 0:
            return []
        if self.index is not None:
            ranked = self.index.search(query.coord, k=k, nprobe=self.nprobe)
            return [Candidate(nid, -dist, r, self.name)
                    for r, (nid, dist) in enumerate(ranked)]
        d = G.dist_c_batch(query.coord, self.coords, self.c)
        order = np.argsort(d)[:k]
        return [Candidate(self.ids[i], float(-d[i]), r, self.name)
                for r, i in enumerate(order)]
