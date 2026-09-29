# ask-ashish

Backend for a portfolio chatbot that answers **as Ashish Kosana** from a small, curated corpus of **published** materials. The site at [ashishkosana.com](https://www.ashishkosana.com/) can `POST /chat` with JSON only. The browser never sees an API key.

Answers cite the note they came from. If the corpus does not contain the fact, the API abstains. It does not invent employers, dates, or metrics.

This repository is the API. It does not add the chat widget to the static site, and the widget is not live there today.

## What a recruiter gets

`POST /chat` returns first-person text grounded in the notes under `corpus/`, plus citations:

```json
{
  "answer": "… [1]",
  "citations": [
    {
      "source": "projects/ledgerline.md",
      "chunk_index": 0,
      "url": "https://github.com/Ashishkosana/ledgerline"
    }
  ],
  "abstained": false,
  "disclaimer": "Answers are generated from Ashish's published materials only."
}
```

`abstained: true` means the service refused to guess. The answer then contains the sentence `I don't have enough information to answer that.`

## What this will not claim

- That the bot is Ashish, or that it remembers private interviews
- That every GitHub repo is in the index (it is not)
- Live traffic, latency SLOs, or star counts
- Employers, titles, dates, or metrics that are not already on the public site or in the public READMEs mirrored here
- That [ai-clone](https://github.com/Ashishkosana/ai-clone) is deployed (it is not this service)

Figures that do appear (GPA, the Crewtron coverage line, the work-card lines) are copied from the public site. See `corpus/SOURCES.md`.

## API

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{ "status": "ok" }` |
| `GET` | `/stats` | — | chunk count, embedding model, provider, generator. No secrets. |
| `POST` | `/chat` | `{ "message", "session_id"?, "top_k"? }` | `{ answer, citations, abstained, disclaimer }` |

`message` is 1–500 characters. `top_k` defaults to 4 (max 8). `session_id` is accepted and ignored: each call is stateless.

There is no public ingest route. Indexing is startup plus `scripts/ingest.py`.

CORS allows only:

- `https://www.ashishkosana.com`
- `https://ashishkosana.com`
- `http://localhost` and `http://127.0.0.1` (any port)

Other origins do not get `Access-Control-Allow-Origin`. Credentials are not enabled, so this is not `*` plus cookies.

`POST /chat` is limited per client IP (default 20 requests per hour, in memory, one process). Over the limit the API returns `429` and `Retry-After`. `GET /health` is not limited, so a platform health check can keep polling. Behind Railway the limit key is the first `X-Forwarded-For` hop.

## How retrieval works

```
corpus/  (.md .txt .pdf)
   → paragraph chunks (800 chars, 120 overlap)
   → embeddings
   → Chroma (cosine)
   → top-k passages
   → grounded answer with [n] citations, or abstain
```

Production embeddings are local `sentence-transformers/all-MiniLM-L6-v2`. Generation is Anthropic or OpenAI, chosen with `ASK_ASHISH_LLM_PROVIDER`. The model id is `ASK_ASHISH_ANSWER_MODEL` or a provider default (`claude-sonnet-4-5`, `gpt-4o-mini`). Those defaults are starting points, not a statement about which model a deploy is running.

The index is rebuilt from `corpus/` every time the process starts. A Railway volume is optional; the corpus is in the image. Ephemeral disk is enough.

## Offline mode (no API key)

Pytest and `./scripts/run_dev.sh` do not need a key or a model download.

- `ASK_ASHISH_EMBEDDINGS=hash` — deterministic hashed bag-of-words (`hash-bow-256`), not MiniLM
- `ASK_ASHISH_LLM_API_KEY` empty — the generator **quotes** the retrieved notes instead of calling a model
- Hash mode abstains when a content word of 5+ characters in the question is not in those notes. That catches off-corpus names. It is weaker than MiniLM on paraphrase ("college" vs "UMass Lowell"). Do not describe hash mode as the production embedder.

`/stats` reports `generator: "mock"` and `embedding_model: "hash-bow-256"` when this path is on.

## Run it locally

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # leave the key empty
./scripts/run_dev.sh          # http://127.0.0.1:8000
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/stats
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is ledgerline?"}'
```

`scripts/ingest.py` rebuilds `.chroma/` without starting the server. The API also reindexes on startup.

To call a real model locally:

```bash
pip install -e ".[minilm]"
export ASK_ASHISH_EMBEDDINGS=minilm
export ASK_ASHISH_LLM_PROVIDER=anthropic   # or openai
export ASK_ASHISH_LLM_API_KEY=...          # never commit this
export ASK_ASHISH_REQUIRE_LLM=1
./scripts/run_dev.sh
```

## Configuration

Copy `.env.example`. Every variable uses the prefix `ASK_ASHISH_`.

| Variable | Role |
| --- | --- |
| `ASK_ASHISH_LLM_PROVIDER` | `anthropic` or `openai` |
| `ASK_ASHISH_LLM_API_KEY` | Server-side key. Empty → mock generator. |
| `ASK_ASHISH_ANSWER_MODEL` | Optional model id for that provider. |
| `ASK_ASHISH_REQUIRE_LLM` | `1` refuses to boot if the key is empty. |
| `ASK_ASHISH_EMBEDDINGS` | `minilm` or `hash` |
| `ASK_ASHISH_EMBEDDING_MODEL` | Hugging Face id used when embeddings are `minilm` |
| `ASK_ASHISH_TOP_K` | Default 4 |
| `ASK_ASHISH_CHUNK_SIZE` / `ASK_ASHISH_CHUNK_OVERLAP` | Default 800 / 120 |
| `ASK_ASHISH_MAX_TOKENS` | Cap on the provider response. Default 600. |
| `ASK_ASHISH_RATE_LIMIT_PER_HOUR` | Per IP. `0` disables the limit. |
| `ASK_ASHISH_CORPUS_DIR` / `ASK_ASHISH_PERSIST_DIR` | Default `corpus` and `.chroma` |

### `ASK_ASHISH_REQUIRE_LLM`

The Docker image sets `ASK_ASHISH_REQUIRE_LLM=1` and `ASK_ASHISH_EMBEDDINGS=minilm`. If Railway starts the container without `ASK_ASHISH_LLM_API_KEY`, the process exits instead of serving quoted mock answers to the public site.

Local `./scripts/run_dev.sh` defaults `ASK_ASHISH_REQUIRE_LLM` to `0` so the offline path works. Set it to `1` in production even if you override the image env.

Do not commit `.env`. `.gitignore` ignores it.

## Corpus

`corpus/` is the only knowledge the bot may use. `corpus/SOURCES.md` maps each file to a public URL. `corpus/sources.json` is the same map for citations.

These markdown files are **placeholders** distilled from the live site and the public READMEs. A later **green fact-bank export** (approved facts only) should replace them. Until then, change a fact by editing the file and redeploying.

Not included: the résumé PDF (link it in; ingest already accepts `.pdf`), private `career/` notes, Gmail, secrets, and unpublished metrics. `review-lens` precision numbers and askdocs sample eval scores are omitted on purpose — those READMEs do not treat them as a stable published result for this bio.

## Deploy on Railway

The container listens on `0.0.0.0` and `$PORT`. `railway.toml` points the health check at `/health`.

1. Create a Railway project from this GitHub repo. The Dockerfile is the builder.
2. The image installs MiniLM at **build** time and sets `ASK_ASHISH_REQUIRE_LLM=1`. The first image build is large because of PyTorch.
3. Set variables on the service (not in git):
   - `ASK_ASHISH_LLM_PROVIDER` = `anthropic` or `openai`
   - `ASK_ASHISH_LLM_API_KEY` = your key
   - `ASK_ASHISH_REQUIRE_LLM` = `1`
   - `ASK_ASHISH_EMBEDDINGS` = `minilm` (already the image default)
   - `ASK_ASHISH_ANSWER_MODEL` = optional override
4. Deploy. `GET /health` should return `{"status":"ok"}`. `GET /stats` should show `embedding_mode` `minilm`, `generator` `llm`, and a chunk count above zero. It must not show the key.
5. Generate a public Railway domain. That origin is the API base the site will call. No path suffix.

A volume is unnecessary for this corpus. Changing a note means a new deploy, which reindexes on boot.

If you only want a private smoke test of the container without a key, set `ASK_ASHISH_REQUIRE_LLM=0` and `ASK_ASHISH_EMBEDDINGS=hash`. `/stats` will say `generator: mock`. Do not point the public site at that.

## Point the site at the API

The static site should learn the public URL from one meta tag. Do not put a provider key in the page.

```html
<meta name="ask-ashish-api" content="https://YOUR-SERVICE.up.railway.app" />
```

The widget (a later change to `Ashishkosana.github.io`, not this repo) should read it and POST JSON:

```js
const API_BASE = document
  .querySelector('meta[name="ask-ashish-api"]')
  .content.replace(/\/$/, "");
await fetch(API_BASE + "/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message }),
});
```

Use the Railway URL with no trailing slash. `https://www.ashishkosana.com` and `https://ashishkosana.com` are already on the CORS allow list. Add nothing else unless the API is changed.

Until that tag exists, the site has no chat backend. Do not describe the widget as shipped.

## Tests

GitHub Actions runs Ruff and pytest. Tests use hash embeddings and the mock generator. They do not download MiniLM and they do not call a provider.

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT. See `LICENSE`.
