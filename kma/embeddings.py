from __future__ import annotations

import hashlib
import os
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
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        # Override with KMA_ST_MODEL, e.g. google/embeddinggemma-300m once you have
        # accepted its license and installed sentence-transformers>=5.0.
        model_name = model_name or os.environ.get("KMA_ST_MODEL", "all-MiniLM-L6-v2")
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers/{model_name}"

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float64)


class OpenAIEmbedder:
    """OpenAI embeddings via the REST API (`OPENAI_API_KEY` required).

    Works with any OpenAI-compatible endpoint through `OPENAI_BASE_URL`.
    Returns L2-normalized vectors so dot product == cosine, matching the others.
    """

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # lazy import

        self.model = model or os.environ.get("KMA_OPENAI_MODEL", "text-embedding-3-small")
        self._client = OpenAI()
        self.name = f"openai/{self.model}"
        self.dim = 3072 if "large" in self.model else 1536

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float64)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs


@lru_cache(maxsize=1)
def get_embedder():  # noqa: ANN201 - returns one of the backends
    """Pick an embedding backend.

    Override with KMA_EMBEDDER = openai | st | hashing. Default ("auto") tries
    sentence-transformers, then falls back to the dependency-free hashing model.
    """
    choice = os.environ.get("KMA_EMBEDDER", "auto").lower()
    if choice == "openai":
        return OpenAIEmbedder()
    if choice == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception:  # noqa: BLE001 - any import/download failure -> fallback
        if choice == "st":
            raise
        return HashingEmbedder()
