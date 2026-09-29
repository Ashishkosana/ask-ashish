FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASK_ASHISH_EMBEDDINGS=minilm \
    ASK_ASHISH_REQUIRE_LLM=1 \
    ANONYMIZED_TELEMETRY=false \
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY corpus ./corpus

# MiniLM is baked at build time so a Railway boot does not download weights.
RUN pip install --upgrade pip \
    && pip install ".[minilm]" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

# Railway sets PORT. ASK_ASHISH_REQUIRE_LLM=1 refuses to boot without
# ASK_ASHISH_LLM_API_KEY, so a misconfigured deploy cannot serve mock answers.
CMD ["sh", "-c", "python -m uvicorn ask_ashish.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
