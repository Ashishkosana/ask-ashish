"""Grounded answer generation: a mock for offline use, or Anthropic / OpenAI / Groq."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from .store import Hit
from .text import content_tokens

logger = logging.getLogger("ask_ashish")

ABSTAIN_SENTENCE = "I don't have enough information to answer that."
ABSTAIN_ANSWER = (
    "I don't have enough information to answer that. "
    "You can email me at ashishkosana@gmail.com or find me on LinkedIn: "
    "https://www.linkedin.com/in/ashishkosana/"
)
DISCLAIMER = "Answers are generated from Ashish's published materials only."

SYSTEM = """You answer as Ashish Kosana, in the first person, for recruiters on his portfolio site.

Rules:
- Use ONLY the numbered passages. They are published materials (the public site
  and project notes). Do not use outside knowledge.
- Cite the passages you use with bracketed numbers like [1].
- If the passages do not contain the answer, reply with this exact sentence
  and nothing that invents a fact: "I don't have enough information to answer that."
  You may then add one short line with the public email ashishkosana@gmail.com
  or LinkedIn https://www.linkedin.com/in/ashishkosana/.
- Never invent employers, titles, dates, metrics, users, stars, or sponsorship outcomes.
- Work authorization: only restate what the passages say.
- Do not give a salary, a phone number, credentials, or private contact details
  unless the passages already publish them.
- Keep the answer to a few sentences.
"""

_CITE = re.compile(r"\[(\d+)\]")


class LlmError(RuntimeError):
    """The configured provider failed. Details stay in server logs."""


@dataclass(frozen=True)
class Citation:
    source: str
    chunk_index: int
    url: str | None


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: list[Citation]
    abstained: bool


class Generator(Protocol):
    def generate(self, message: str, hits: list[Hit]) -> str: ...


def build_user_prompt(message: str, hits: list[Hit]) -> str:
    if hits:
        blocks = "\n\n".join(
            f"[{index}] (source: {hit.source})\n{hit.text}"
            for index, hit in enumerate(hits, start=1)
        )
    else:
        blocks = "(no context retrieved)"
    return f"Context passages:\n\n{blocks}\n\nQuestion: {message}\n\nAnswer:"


def is_abstention(answer: str) -> bool:
    return ABSTAIN_SENTENCE in answer


def citations_from_answer(answer: str, hits: list[Hit]) -> list[Citation]:
    """Map [n] markers onto hits. If the model cites nothing, return every hit."""
    seen: set[int] = set()
    chosen: list[Hit] = []
    for raw in _CITE.findall(answer):
        number = int(raw)
        if number in seen or not 1 <= number <= len(hits):
            continue
        seen.add(number)
        chosen.append(hits[number - 1])
    if not chosen:
        chosen = list(hits)
    return [
        Citation(source=hit.source, chunk_index=hit.chunk_index, url=hit.url) for hit in chosen
    ]


def _focus_snippet(text: str, message: str, limit: int = 420) -> str:
    """Quote the passage, starting near the first question word so overlap tails don't lead."""
    snippet = " ".join(text.split())
    lowered = snippet.lower()
    positions = [lowered.find(term) for term in content_tokens(message)]
    positions = [pos for pos in positions if pos >= 0]
    start = 0
    if positions:
        start = max(0, min(positions) - 40)
        space = snippet.rfind(" ", 0, start)
        if space != -1:
            start = space + 1
    snippet = snippet[start:]
    if len(snippet) > limit:
        snippet = snippet[:limit].rsplit(" ", 1)[0] + "…"
    return snippet


class MockGenerator:
    """Quote the retrieved passage. Used when no LLM key is configured."""

    def generate(self, message: str, hits: list[Hit]) -> str:
        parts: list[str] = []
        for index, hit in enumerate(hits[:1], start=1):
            snippet = _focus_snippet(hit.text, message)
            parts.append(f"{snippet} [{index}]")
        return "\n\n".join(parts)


class AnthropicGenerator:
    def __init__(self, *, api_key: str, model: str, max_tokens: int, client=None) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = client

    def generate(self, message: str, hits: list[Hit]) -> str:
        client = self._client or self._build_client()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM,
                messages=[{"role": "user", "content": build_user_prompt(message, hits)}],
            )
        except Exception as exc:
            logger.error("anthropic request failed: %s", type(exc).__name__)
            raise LlmError("anthropic request failed") from exc
        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        ]
        return "\n".join(parts).strip()

    def _build_client(self):
        import anthropic

        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenAIGenerator:
    """Chat completions on an OpenAI-compatible API. Groq sets ``base_url``."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        client=None,
        base_url: str | None = None,
        provider: str = "openai",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = client
        self._base_url = base_url
        self._provider = provider

    def generate(self, message: str, hits: list[Hit]) -> str:
        client = self._client or self._build_client()
        try:
            response = client.chat.completions.create(
                model=self._model,
                max_completion_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user_prompt(message, hits)},
                ],
            )
        except Exception as exc:
            logger.error("%s request failed: %s", self._provider, type(exc).__name__)
            raise LlmError(f"{self._provider} request failed") from exc
        content = response.choices[0].message.content or ""
        return content.strip()

    def _build_client(self):
        from openai import OpenAI

        kwargs: dict[str, str] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        return self._client


class GroqGenerator(OpenAIGenerator):
    """Groq chat completions through the OpenAI client."""

    def __init__(self, *, api_key: str, model: str, max_tokens: int, client=None) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            client=client,
            base_url=GROQ_BASE_URL,
            provider="groq",
        )


def build_generator(settings) -> Generator:  # noqa: ANN001
    if not settings.llm_configured:
        return MockGenerator()
    kwargs = {
        "api_key": settings.llm_api_key.strip(),
        "model": settings.resolved_model,
        "max_tokens": settings.max_tokens,
    }
    if settings.llm_provider == "openai":
        return OpenAIGenerator(**kwargs)
    if settings.llm_provider == "groq":
        return GroqGenerator(**kwargs)
    return AnthropicGenerator(**kwargs)
