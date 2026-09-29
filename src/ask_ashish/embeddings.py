"""Embedding backends.

MiniLM is the production embedder (local, no per-query API cost). Hash
embeddings are a deterministic bag-of-words stand-in so tests and
``./scripts/run_dev.sh`` run without a model download.
"""

from __future__ import annotations

import hashlib
import math
from functools import cached_property
from typing import Protocol, runtime_checkable

from .text import content_tokens, tokens

HASH_DIM = 256
HASH_MODEL_NAME = "hash-bow-256"


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """L2-normalized hashed bag of words. Same text always yields the same vector."""

    def __init__(self, dim: int = HASH_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def model_name(self) -> str:
        return HASH_MODEL_NAME

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_embed_hash(text, self._dim) for text in texts]


def _embed_hash(text: str, dim: int) -> list[float]:
    chosen = sorted(content_tokens(text)) or tokens(text) or ["empty"]
    vec = [0.0] * dim
    for tok in chosen:
        digest = hashlib.blake2s(tok.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(value * value for value in vec)) or 1.0
    return [value / norm for value in vec]


class SentenceTransformerEmbedder:
    """Local ``sentence-transformers`` model. The weights load on first use."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @cached_property
    def _model(self):  # noqa: ANN202 - third-party type
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "MiniLM embeddings require the optional extra. "
                "Install with: pip install 'ask-ashish[minilm]'"
            ) from exc
        return SentenceTransformer(self._model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()


def build_embedder(settings) -> Embedder:  # noqa: ANN001 - avoids a settings import cycle
    if settings.embeddings == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
