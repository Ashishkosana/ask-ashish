"""Runtime settings from the environment (prefix ``ASK_ASHISH_``) or a local ``.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class Settings(BaseSettings):
    """Tunable knobs. Init arguments win over environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ASK_ASHISH_",
        env_file=".env",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_provider: Literal["anthropic", "openai", "groq"] = "anthropic"
    require_llm: bool = False
    answer_model: str = ""
    max_tokens: int = Field(default=600, ge=64, le=2000)

    embeddings: Literal["minilm", "hash"] = "hash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)

    top_k: int = Field(default=4, ge=1, le=8)
    min_score: float | None = None

    persist_dir: Path = Path(".chroma")
    corpus_dir: Path = Path("corpus")
    collection: str = ""

    rate_limit_per_hour: int = Field(default=20, ge=0, le=100_000)

    @model_validator(mode="after")
    def _overlap_fits(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def resolved_model(self) -> str:
        chosen = self.answer_model.strip()
        if chosen:
            return chosen
        if self.llm_provider == "openai":
            return DEFAULT_OPENAI_MODEL
        if self.llm_provider == "groq":
            return DEFAULT_GROQ_MODEL
        return DEFAULT_ANTHROPIC_MODEL

    @property
    def resolved_min_score(self) -> float:
        if self.min_score is not None:
            return self.min_score
        # Hash vectors are lexical. A low floor plus a token-overlap check
        # abstains on off-corpus questions without dropping a rare name.
        if self.embeddings == "hash":
            return 0.05
        return 0.30

    @property
    def resolved_collection(self) -> str:
        chosen = self.collection.strip()
        if chosen:
            return chosen
        return f"ask-ashish-{self.embeddings}"

    @property
    def generator_name(self) -> str:
        return "llm" if self.llm_configured else "mock"
