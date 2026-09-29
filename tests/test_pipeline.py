from pathlib import Path

from ask_ashish.config import Settings
from ask_ashish.generate import ABSTAIN_SENTENCE
from ask_ashish.pipeline import RagPipeline


def test_hash_pipeline_answers_and_abstains(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "projects").mkdir()
    (corpus / "projects" / "ledgerline.md").write_text(
        "I built ledgerline, an exactly-once payments backend.\n",
        encoding="utf-8",
    )
    (corpus / "sources.json").write_text(
        '{"projects/ledgerline.md": "https://github.com/Ashishkosana/ledgerline"}\n',
        encoding="utf-8",
    )
    settings = Settings(
        embeddings="hash",
        llm_api_key="",
        require_llm=False,
        persist_dir=tmp_path / "chroma",
        corpus_dir=corpus,
        rate_limit_per_hour=0,
    )
    pipeline = RagPipeline(settings)
    assert pipeline.reindex(corpus) == 1

    hit = pipeline.answer("What is ledgerline?")
    assert hit.abstained is False
    assert "ledgerline" in hit.answer.lower()
    assert hit.citations[0].source == "projects/ledgerline.md"
    assert hit.citations[0].url == "https://github.com/Ashishkosana/ledgerline"

    miss = pipeline.answer("Did you work at Initech?")
    assert miss.abstained is True
    assert miss.citations == []
    assert ABSTAIN_SENTENCE in miss.answer
