"""Index a corpus, retrieve top-k, and generate a cited answer or abstain."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .embeddings import Embedder, build_embedder
from .generate import (
    ABSTAIN_ANSWER,
    ChatResult,
    Generator,
    LlmError,
    build_generator,
    citations_from_answer,
    is_abstention,
)
from .ingest import ingest_corpus
from .retrieve import Retriever
from .store import VectorStore
from .text import content_tokens, tokens


def _hash_hits(message: str, hits: list) -> list:
    """Keep hash-mode hits that actually cover the question's content words.

    A 5+ character word from the question must appear in the retrieved notes.
    That is what makes "Did you work at Initech?" abstain even though "work"
    shows up all over the corpus. MiniLM mode does not use this veto.
    """
    query_terms = content_tokens(message)
    if not query_terms:
        return hits
    kept = [hit for hit in hits if query_terms & content_tokens(hit.text)]
    if not kept:
        return []
    covered: set[str] = set()
    for hit in kept:
        covered |= content_tokens(hit.text)
    long_terms = {term for term in query_terms if len(term) >= 5}
    if long_terms - covered:
        return []

    def mentions(hit) -> int:
        return sum(1 for tok in tokens(hit.text) if tok in query_terms)

    grouped: dict[str, list] = {}
    for hit in kept:
        grouped.setdefault(hit.source, []).append(hit)

    def source_rank(item: tuple[str, list]) -> tuple[bool, int]:
        source, group = item
        # Filename match beats a repeated word in a URL. Mention count then
        # separates the skills note from a one-line "Stack:" on a project card.
        return (
            any(term in source.lower() for term in query_terms),
            max(mentions(hit) for hit in group),
        )

    best_source, _best_group = max(grouped.items(), key=source_rank)
    primary = sorted(
        (hit for hit in kept if hit.source == best_source),
        key=lambda hit: hit.chunk_index,
    )
    rest = sorted(
        (hit for hit in kept if hit.source != best_source),
        key=lambda hit: (-mentions(hit), hit.chunk_index),
    )
    return primary + rest


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.store = store or VectorStore(settings.persist_dir, settings.resolved_collection)
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = generator or build_generator(settings)

    def index(self, corpus: Path) -> int:
        chunks = ingest_corpus(
            Path(corpus),
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if chunks:
            self.store.add(chunks, self.embedder.embed([chunk.text for chunk in chunks]))
        return len(chunks)

    def reindex(self, corpus: Path) -> int:
        self.store.reset()
        return self.index(corpus)

    def answer(self, message: str, *, top_k: int | None = None) -> ChatResult:
        k = top_k or self.settings.top_k
        # Hash cosine prefers short notes. Pull the whole small corpus, then
        # re-rank by the question's words so "stack" can reach the skills note.
        fetch_k = k
        if self.settings.embeddings == "hash":
            fetch_k = max(k, self.store.count())
        hits = self.retriever.retrieve(message, top_k=fetch_k)
        kept = [hit for hit in hits if hit.score >= self.settings.resolved_min_score]
        if self.settings.embeddings == "hash":
            kept = _hash_hits(message, kept)[:k]
        if not kept:
            return ChatResult(answer=ABSTAIN_ANSWER, citations=[], abstained=True)

        try:
            text = self.generator.generate(message, kept).strip()
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError("generation failed") from exc
        if not text or is_abstention(text):
            return ChatResult(answer=text or ABSTAIN_ANSWER, citations=[], abstained=True)
        return ChatResult(
            answer=text,
            citations=citations_from_answer(text, kept),
            abstained=False,
        )
