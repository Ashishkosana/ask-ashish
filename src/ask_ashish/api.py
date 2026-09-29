"""FastAPI surface: ``GET /health``, ``GET /stats``, ``POST /chat``.

Ingest is not exposed. The process rebuilds the index from ``corpus/`` on startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings
from .generate import DISCLAIMER, LlmError
from .pipeline import RagPipeline
from .rate_limit import RateLimiter

logger = logging.getLogger("ask_ashish")

ALLOWED_ORIGINS = [
    "https://www.ashishkosana.com",
    "https://ashishkosana.com",
]
LOCAL_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=80)
    top_k: int | None = Field(default=None, ge=1, le=8)


class CitationOut(BaseModel):
    source: str
    chunk_index: int
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    abstained: bool
    disclaimer: str


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        current: Settings = app.state.settings
        if current.require_llm and not current.llm_configured:
            raise RuntimeError(
                "ASK_ASHISH_REQUIRE_LLM is set but ASK_ASHISH_LLM_API_KEY is empty. "
                "Refusing to serve mock answers."
            )
        pipeline = RagPipeline(current)
        corpus = current.corpus_dir
        if not corpus.exists():
            raise RuntimeError(f"corpus not found: {corpus}")
        indexed = pipeline.reindex(corpus)
        app.state.pipeline = pipeline
        app.state.limiter = RateLimiter(current.rate_limit_per_hour)
        logger.info(
            "ready chunks=%s embeddings=%s generator=%s",
            indexed,
            pipeline.embedder.model_name,
            current.generator_name,
        )
        yield

    app = FastAPI(
        title="ask-ashish",
        version="0.1.0",
        summary="Portfolio RAG chat grounded in Ashish Kosana's published materials.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=LOCAL_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict[str, object]:
        pipeline: RagPipeline = app.state.pipeline
        current: Settings = app.state.settings
        return {
            "indexed_chunks": pipeline.store.count(),
            "embedding_model": pipeline.embedder.model_name,
            "embedding_mode": current.embeddings,
            "llm_provider": current.llm_provider,
            "generator": current.generator_name,
            "answer_model": current.resolved_model if current.llm_configured else None,
            "top_k": current.top_k,
            "require_llm": current.require_llm,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message is empty")
        # session_id is accepted for a future short history. This MVP is stateless.
        del body.session_id

        limiter: RateLimiter = request.app.state.limiter
        key = _client_key(request)
        if not limiter.allow(key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(limiter.retry_after(key))},
            )

        pipeline: RagPipeline = request.app.state.pipeline
        if pipeline.store.count() == 0:
            raise HTTPException(status_code=503, detail="No documents are indexed.")
        try:
            result = pipeline.answer(message, top_k=body.top_k)
        except LlmError:
            raise HTTPException(
                status_code=502,
                detail="The answer service is unavailable right now.",
            ) from None
        logger.info("chat abstained=%s citations=%s", result.abstained, len(result.citations))
        return ChatResponse(
            answer=result.answer,
            citations=[
                CitationOut(source=item.source, chunk_index=item.chunk_index, url=item.url)
                for item in result.citations
            ],
            abstained=result.abstained,
            disclaimer=DISCLAIMER,
        )

    return app


app = create_app()
