#!/usr/bin/env python3
"""Rebuild the local Chroma index from corpus/.

Reads ASK_ASHISH_* (and .env) the same way the API does. Clears the collection
first so files removed from the corpus do not linger. Does not call an LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from ask_ashish.config import Settings
    from ask_ashish.pipeline import RagPipeline

    settings = Settings()
    if not settings.corpus_dir.exists():
        raise SystemExit(f"corpus not found: {settings.corpus_dir}")
    pipeline = RagPipeline(settings)
    count = pipeline.reindex(settings.corpus_dir)
    print(
        f"indexed {count} chunks "
        f"embeddings={pipeline.embedder.model_name} "
        f"dir={settings.persist_dir}"
    )


if __name__ == "__main__":
    main()
