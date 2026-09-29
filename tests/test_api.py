import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ask_ashish.api import create_app
from ask_ashish.config import Settings
from ask_ashish.generate import ABSTAIN_SENTENCE, DISCLAIMER

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def _settings(tmp_path: Path, **overrides) -> Settings:
    data = {
        "embeddings": "hash",
        "llm_api_key": "",
        "require_llm": False,
        "persist_dir": tmp_path / "chroma",
        "corpus_dir": CORPUS,
        "rate_limit_per_hour": 0,
    }
    data.update(overrides)
    return Settings(**data)


def test_health_stats_and_chat_offline(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        stats = client.get("/stats")
        body = stats.json()
        assert stats.status_code == 200
        assert body["indexed_chunks"] > 0
        assert body["embedding_model"] == "hash-bow-256"
        assert body["embedding_mode"] == "hash"
        assert body["generator"] == "mock"
        assert body["answer_model"] is None
        assert "api_key" not in stats.text

        chat = client.post("/chat", json={"message": "What is ledgerline?", "top_k": 4})
        payload = chat.json()
        assert chat.status_code == 200
        assert payload["abstained"] is False
        assert "ledgerline" in payload["answer"].lower()
        assert payload["disclaimer"] == DISCLAIMER
        assert any(
            item["source"] == "projects/ledgerline.md"
            and item["url"] == "https://github.com/Ashishkosana/ledgerline"
            for item in payload["citations"]
        )

        missing = client.post("/chat", json={"message": "Did you work at Initech?"})
        assert missing.status_code == 200
        assert missing.json()["abstained"] is True
        assert missing.json()["citations"] == []
        assert ABSTAIN_SENTENCE in missing.json()["answer"]

        salary = client.post("/chat", json={"message": "What is your salary?"})
        assert salary.status_code == 200
        assert salary.json()["abstained"] is False
        assert "$" not in salary.json()["answer"]
        assert "not published" in salary.json()["answer"].lower()
        assert salary.json()["citations"][0]["source"] == "limits.md"

        stack = client.post("/chat", json={"message": "What's your stack?"})
        assert stack.status_code == 200
        assert stack.json()["citations"][0]["source"] == "bio.md"
        assert "fastapi" in stack.json()["answer"].lower()


def test_stats_hides_api_key(tmp_path: Path) -> None:
    secret = "sk-test-secret-value"
    with TestClient(_app(tmp_path, llm_api_key=secret)) as client:
        stats = client.get("/stats")
        assert stats.status_code == 200
        assert stats.json()["generator"] == "llm"
        assert secret not in stats.text
        assert secret not in json.dumps(stats.json())


def test_mock_generator_when_key_missing(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_api_key="")) as client:
        assert client.get("/stats").json()["generator"] == "mock"
        response = client.post("/chat", json={"message": "What is ledgerline?"})
        assert response.status_code == 200
        assert response.json()["abstained"] is False


def test_message_limits_and_session_id_ignored(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        too_long = client.post("/chat", json={"message": "a" * 501})
        assert too_long.status_code == 422
        blank = client.post("/chat", json={"message": "   "})
        assert blank.status_code == 422
        ok = client.post(
            "/chat",
            json={"message": "What is tick?", "session_id": "abc-123"},
        )
        assert ok.status_code == 200
        assert "session" not in ok.json()


def test_cors_allowlist(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        allowed = client.get("/health", headers={"Origin": "https://www.ashishkosana.com"})
        assert allowed.headers["access-control-allow-origin"] == "https://www.ashishkosana.com"
        apex = client.get("/health", headers={"Origin": "https://ashishkosana.com"})
        assert apex.headers["access-control-allow-origin"] == "https://ashishkosana.com"
        local = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert local.headers["access-control-allow-origin"] == "http://localhost:5173"
        blocked = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in blocked.headers


def test_rate_limit(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, rate_limit_per_hour=2)) as client:
        assert client.post("/chat", json={"message": "What is ledgerline?"}).status_code == 200
        assert client.post("/chat", json={"message": "What is tick?"}).status_code == 200
        blocked = client.post("/chat", json={"message": "What is Rythu?"})
        assert blocked.status_code == 429
        assert blocked.headers["retry-after"]


def test_require_llm_refuses_to_start(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, require_llm=True, llm_api_key=""))
    with pytest.raises(RuntimeError, match="ASK_ASHISH_REQUIRE_LLM"):
        with TestClient(app):
            pass


def test_empty_corpus_returns_503(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with TestClient(_app(tmp_path, corpus_dir=empty)) as client:
        response = client.post("/chat", json={"message": "What is ledgerline?"})
        assert response.status_code == 503


def test_sources_cover_corpus_and_env_example_has_no_secret() -> None:
    mapping = json.loads((CORPUS / "sources.json").read_text(encoding="utf-8"))
    files = {
        path.relative_to(CORPUS).as_posix()
        for path in CORPUS.rglob("*")
        if path.suffix.lower() in {".md", ".txt", ".pdf"} and path.name != "SOURCES.md"
    }
    assert files == set(mapping)
    assert all(url.startswith("https://") for url in mapping.values())
    sources = (CORPUS / "SOURCES.md").read_text(encoding="utf-8").lower()
    assert "green fact-bank" in sources
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ASK_ASHISH_LLM_API_KEY=" in example
    assert "sk-" not in example
    assert "ASK_ASHISH_REQUIRE_LLM" in (ROOT / "README.md").read_text(encoding="utf-8")


def _app(tmp_path: Path, **overrides):
    return create_app(_settings(tmp_path, **overrides))
