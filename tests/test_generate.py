from ask_ashish.generate import (
    ABSTAIN_SENTENCE,
    SYSTEM,
    AnthropicGenerator,
    OpenAIGenerator,
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
