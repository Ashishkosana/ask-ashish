"""Load ``.md``, ``.txt``, and ``.pdf`` files into chunks."""

from __future__ import annotations

import json
from pathlib import Path

from .chunking import Chunk, chunk_text

SUPPORTED = {".txt", ".md", ".pdf"}
_SKIP_NAMES = {"SOURCES.md", "sources.json"}


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_source_urls(corpus: Path) -> dict[str, str]:
    """Map a corpus-relative path to a public URL, from ``sources.json`` if present."""
    catalog = corpus / "sources.json" if corpus.is_dir() else corpus.parent / "sources.json"
    if not catalog.is_file():
        return {}
    raw = json.loads(catalog.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{catalog} must be a JSON object of path -> url")
    return {str(key): str(value) for key, value in raw.items()}


def discover(corpus: Path) -> list[Path]:
    if corpus.is_file():
        return [corpus] if corpus.suffix.lower() in SUPPORTED else []
    found = [
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED and path.name not in _SKIP_NAMES
    ]
    return sorted(found)


def _relative_source(corpus: Path, path: Path) -> str:
    if corpus.is_file():
        return path.name
    return path.relative_to(corpus).as_posix()


def ingest_corpus(corpus: Path, *, size: int, overlap: int) -> list[Chunk]:
    urls = load_source_urls(corpus)
    chunks: list[Chunk] = []
    for path in discover(corpus):
        text = load_text(path)
        if not text.strip():
            continue
        source = _relative_source(corpus, path)
        chunks.extend(
            chunk_text(text, source, size=size, overlap=overlap, url=urls.get(source))
        )
    return chunks
