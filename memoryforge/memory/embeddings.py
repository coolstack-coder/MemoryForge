"""Text embeddings.

`MiniLMEmbedder` uses sentence-transformers (PyTorch). `HashEmbedder` is a
deterministic, dependency-free fallback: hashed word and character n-grams
projected to 384 dimensions. Vectors from the two are not comparable, so the
store records which one produced them.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

DIM = 384


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    name = "hash-ngram-384"
    dim = DIM

    def __init__(self, buckets: int = 1 << 15) -> None:
        self.buckets = buckets
        rng = np.random.default_rng(1234)
        self._proj = (rng.standard_normal((buckets, DIM)) / np.sqrt(DIM)).astype(np.float32)

    def _features(self, text: str) -> dict[int, float]:
        text = text.lower()
        words = re.findall(r"[a-z0-9_]+", text)
        grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
        squashed = " ".join(words)
        grams += [squashed[i : i + 4] for i in range(max(0, len(squashed) - 3))]
        feats: dict[int, float] = {}
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            idx, sign = h % self.buckets, 1.0 if (h >> 63) & 1 else -1.0
            feats[idx] = feats.get(idx, 0.0) + sign
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for idx, v in self._features(t).items():
                out[i] += v * self._proj[idx]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class MiniLMEmbedder:
    name = "all-MiniLM-L6-v2"
    dim = DIM

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=4)
def get_embedder(kind: str | None = None) -> Embedder:
    from ..config import settings

    kind = kind or settings.embedder
    if kind == "hash" or os.environ.get("MF_OFFLINE") == "1":
        return HashEmbedder()
    if kind in ("auto", "minilm"):
        try:
            return MiniLMEmbedder()
        except Exception:  # noqa: BLE001 - no package, no network, no cached weights
            if kind == "minilm":
                raise
    return HashEmbedder()
