"""Embed a question and return the nearest chunks."""

from __future__ import annotations

from .embeddings import Embedder
from .store import Hit, VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        if not query.strip():
            return []
        vector = self._embedder.embed([query])[0]
        return self._store.query(vector, top_k=top_k)
