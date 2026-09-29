"""Shared tokenization for hash embeddings and the offline overlap check."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.+#_-]*", re.IGNORECASE)

# Question glue. Kept short so names and stack words still count as content.
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "do",
        "does",
        "did",
        "what",
        "whats",
        "who",
        "whom",
        "whose",
        "which",
        "when",
        "where",
        "why",
        "how",
        "your",
        "you",
        "yours",
        "my",
        "me",
        "i",
        "we",
        "our",
        "ours",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "if",
        "then",
        "than",
        "into",
        "about",
        "can",
        "could",
        "would",
        "should",
        "please",
        "tell",
        "any",
        "some",
        "just",
        "not",
        "no",
        "yes",
        "have",
        "has",
        "had",
        "there",
        "here",
    }
)


def tokens(text: str) -> list[str]:
    found: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        tok = raw.strip("._-")
        if tok:
            found.append(tok)
    return found


def content_tokens(text: str) -> set[str]:
    return {tok for tok in tokens(text) if len(tok) >= 3 and tok not in _STOP}
