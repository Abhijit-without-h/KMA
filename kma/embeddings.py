"""Embedding layer.

Tries SentenceTransformers (all-MiniLM-L6-v2, 384-D). If it is not installed
or cannot download, falls back to a deterministic hashing embedder so the
whole prototype still runs offline and tests stay reproducible.

Both backends return L2-normalized vectors, so dot product == cosine.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]

_TOKEN = re.compile(r"[a-z0-9]+")


def content_hash(text: str) -> str:
    """Stable content hash for caching / dedup (matches data-model field)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class HashingEmbedder:
    """Deterministic, dependency-free fallback embedder.

    Hashes word unigrams+bigrams into a fixed-width bag, then L2-normalizes.
    Crude, but it captures lexical overlap well enough to exercise and test
    the geometry/retrieval pipeline without a model download.
    """

    name = "hashing-fallback"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _grams(self, text: str) -> list[str]:
        toks = _TOKEN.findall(text.lower())
        grams = list(toks)
        grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return grams

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for g in self._grams(text):
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float64)


@lru_cache(maxsize=1)
def get_embedder():  # noqa: ANN201 - returns either backend
    """Return the best available embedder (model if present, else hashing)."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:  # noqa: BLE001 - any import/download failure -> fallback
        return HashingEmbedder()
