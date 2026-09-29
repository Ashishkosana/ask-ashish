"""Paragraph-aware chunking with a character overlap.

Passages are packed up to ``size`` characters. ``overlap`` characters from the
end of a chunk are carried into the next one so a fact on a boundary stays
retrievable. Defaults used by the service are 800 / 120.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int
    url: str | None = None


def chunk_text(
    text: str,
    source: str,
    *,
    size: int,
    overlap: int,
    url: str | None = None,
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    chunks: list[Chunk] = []
    buf = ""

    def emit(body: str) -> None:
        cleaned = body.strip()
        if cleaned:
            chunks.append(Chunk(text=cleaned, source=source, index=len(chunks), url=url))

    for para in (part.strip() for part in _PARAGRAPH.split(text) if part.strip()):
        if len(para) > size:
            if buf.strip():
                emit(buf)
                buf = ""
            for piece in _hard_split(para, size, overlap):
                emit(piece)
            continue

        candidate = f"{buf}\n\n{para}" if buf else para
        if buf and len(candidate) > size:
            emit(buf)
            tail = _word_overlap(buf.strip(), overlap)
            buf = f"{tail}\n\n{para}" if tail else para
            if len(buf) > size:
                buf = para
        else:
            buf = candidate

    if buf.strip():
        emit(buf)
    return chunks


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split a single overlong paragraph on whitespace."""
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []

    def length_of(parts: list[str]) -> int:
        return len(" ".join(parts))

    for word in words:
        if current and length_of(current) + 1 + len(word) > size:
            pieces.append(" ".join(current))
            tail = _overlap_tail(current, overlap)
            # If the tail is the whole chunk, drop it so a long next word
            # cannot rebuild the same buffer forever.
            current = [] if tail == current else tail
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _word_overlap(text: str, overlap: int) -> str:
    """Last ``overlap`` characters, snapped forward to a word boundary."""
    if overlap <= 0 or not text:
        return ""
    raw = text[-overlap:]
    space = raw.find(" ")
    if 0 <= space < len(raw) - 1:
        return raw[space + 1 :]
    return raw


def _overlap_tail(words: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or not words:
        return []
    tail: list[str] = []
    for word in reversed(words):
        trial = [word, *tail]
        if tail and len(" ".join(trial)) > overlap:
            break
        tail = trial
    return tail
