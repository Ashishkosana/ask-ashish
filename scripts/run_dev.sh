#!/usr/bin/env bash
# Offline dev server: hash embeddings + mock answers unless you export a key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export ASK_ASHISH_EMBEDDINGS="${ASK_ASHISH_EMBEDDINGS:-hash}"
export ASK_ASHISH_REQUIRE_LLM="${ASK_ASHISH_REQUIRE_LLM:-0}"
export ASK_ASHISH_LLM_API_KEY="${ASK_ASHISH_LLM_API_KEY:-}"
export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-false}"

PORT="${PORT:-8000}"
echo "ask-ashish dev http://127.0.0.1:${PORT}  embeddings=${ASK_ASHISH_EMBEDDINGS} require_llm=${ASK_ASHISH_REQUIRE_LLM}"
exec python -m uvicorn ask_ashish.api:app --host 127.0.0.1 --port "${PORT}" --reload
