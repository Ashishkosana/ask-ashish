"""Persistent Chroma collection using cosine distance."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    chunk_index: int
    score: float
    url: str | None


class VectorStore:
    """IDs are ``<source>:<chunk_index>`` so a re-index upserts instead of duplicating.

    Call ``reset`` before a full rebuild so chunks removed from the corpus disappear.
    """

    def __init__(self, persist_dir: Path, collection: str) -> None:
        import chromadb

        self._name = collection
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[f"{chunk.source}:{chunk.index}" for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source": chunk.source,
                    "chunk_index": chunk.index,
                    "url": chunk.url or "",
                }
                for chunk in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[Hit]:
        available = self.count()
        if available <= 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        hits: list[Hit] = []
        for doc, meta, dist in zip(documents, metadatas, distances, strict=True):
            raw_url = str((meta or {}).get("url") or "")
            hits.append(
                Hit(
                    text=str(doc),
                    source=str((meta or {}).get("source") or ""),
                    chunk_index=int((meta or {}).get("chunk_index") or 0),
                    score=1.0 - float(dist),
                    url=raw_url or None,
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())
