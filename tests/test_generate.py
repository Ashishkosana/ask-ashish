import sys
import types

from ask_ashish.config import DEFAULT_GROQ_MODEL, Settings
from ask_ashish.generate import (
    ABSTAIN_SENTENCE,
    GROQ_BASE_URL,
    SYSTEM,
    AnthropicGenerator,
    GroqGenerator,
    MockGenerator,
    OpenAIGenerator,
    build_generator,
    citations_from_answer,
    is_abstention,
)
from ask_ashish.store import Hit


def _hit(source: str, index: int = 0) -> Hit:
    return Hit(text="passage", source=source, chunk_index=index, score=0.9, url=f"https://ex/{source}")


def test_citations_follow_bracket_numbers() -> None:
    hits = [_hit("a.md", 0), _hit("b.md", 1)]
    cited = citations_from_answer("See [2] and again [2].", hits)
    assert len(cited) == 1
    assert cited[0].source == "b.md"
    assert cited[0].chunk_index == 1
    assert cited[0].url == "https://ex/b.md"


def test_abstain_sentence_detected() -> None:
    assert is_abstention(f'{ABSTAIN_SENTENCE} Email me.')
    assert not is_abstention("I built ledgerline [1].")


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _AnthropicResponse:
    def __init__(self) -> None:
        self.content = [_Block("I built ledgerline [1].")]


class _AnthropicMessages:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _AnthropicResponse()


class _AnthropicClient:
    def __init__(self) -> None:
        self.messages = _AnthropicMessages()


def test_anthropic_generator_sends_grounded_prompt() -> None:
    client = _AnthropicClient()
    text = AnthropicGenerator(
        api_key="secret", model="claude-test", max_tokens=100, client=client
    ).generate("What is ledgerline?", [_hit("projects/ledgerline.md")])
    assert text == "I built ledgerline [1]."
    assert client.messages.kwargs["model"] == "claude-test"
    assert client.messages.kwargs["system"] == SYSTEM
    assert "projects/ledgerline.md" in client.messages.kwargs["messages"][0]["content"]
    assert "secret" not in client.messages.kwargs["messages"][0]["content"]


class _OpenAIMessage:
    content = "Cited [1]."


class _OpenAIChoice:
    message = _OpenAIMessage()


class _OpenAIResponse:
    choices = [_OpenAIChoice()]


class _OpenAICompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _OpenAIResponse()


class _OpenAIChat:
    def __init__(self) -> None:
        self.completions = _OpenAICompletions()


class _OpenAIClient:
    def __init__(self) -> None:
        self.chat = _OpenAIChat()


def test_openai_generator_uses_max_completion_tokens() -> None:
    client = _OpenAIClient()
    text = OpenAIGenerator(
        api_key="secret", model="gpt-test", max_tokens=80, client=client
    ).generate("stack", [_hit("bio.md")])
    assert text == "Cited [1]."
    assert client.chat.completions.kwargs["model"] == "gpt-test"
    assert client.chat.completions.kwargs["max_completion_tokens"] == 80
    assert client.chat.completions.kwargs["messages"][0]["content"] == SYSTEM


def test_groq_uses_openai_client_without_network(monkeypatch) -> None:
    created: dict[str, str] = {}
    chats: list[_OpenAIChat] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs: str) -> None:
            created.update(kwargs)
            self.chat = _OpenAIChat()
            chats.append(self.chat)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    settings = Settings(
        llm_provider="groq",
        llm_api_key="test-groq-key",
        answer_model="",
        embeddings="hash",
    )
    assert settings.resolved_model == DEFAULT_GROQ_MODEL
    assert settings.resolved_model == "openai/gpt-oss-120b"

    generator = build_generator(settings)
    assert isinstance(generator, GroqGenerator)
    text = generator.generate("stack", [_hit("bio.md")])
    assert text == "Cited [1]."
    assert created == {"api_key": "test-groq-key", "base_url": GROQ_BASE_URL}
    assert created["base_url"] == "https://api.groq.com/openai/v1"
    sent = chats[0].completions.kwargs
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["max_completion_tokens"] == settings.max_tokens
    assert sent["messages"][0]["content"] == SYSTEM
    assert "bio.md" in sent["messages"][1]["content"]
    assert "test-groq-key" not in sent["messages"][1]["content"]

    offline = Settings(llm_provider="groq", llm_api_key="", embeddings="hash")
    assert isinstance(build_generator(offline), MockGenerator)
