# Agentic RAG over Financial Documents

> Evidence-grounded, cited question-answering over SEC filings (10-Ks, 10-Qs, credit agreements).
> Hybrid retrieval (dense + sparse) over **pgvector** *and* **Qdrant**, cross-encoder reranking, an
> **agentic** loop that decides what to retrieve and when, and a **citation/provenance trail** on
> every answer — served as a containerized FastAPI microservice.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why this project exists

In finance and compliance, an answer is only useful if you can **trace it to its source**. This
system retrieves the relevant passages from financial filings, lets the model run **multi-step**
retrieval for complex (multi-hop) questions, and returns every answer with the exact chunks — source
document, page, and relevance score — that support it.

It demonstrates the full Tier-1 RAG skill stack in one artifact: retrieval as an engineering
discipline (chunking, hybrid search, reranking), evaluation (Ragas + Langfuse), an agentic
orchestration layer, two vector-DB backends to justify a real engineering choice, and production
deployment.

## Architecture

```
                          ┌──────────────────────────────────────────────┐
  SEC EDGAR PDFs  ──▶  ingest: extract → chunk (3 strategies) → embed     │
                          │                                    │          │
                          ▼                                    ▼          │
                  ┌───────────────┐                    ┌───────────────┐  │
                  │  pgvector     │   (switchable)     │   Qdrant      │  │
                  │ halfvec+HNSW  │ ◀───────────────▶  │ named vectors │  │
                  │ + FTS (sparse)│                    │ query_points  │  │
                  └───────┬───────┘                    └───────┬───────┘  │
                          │   dense + sparse → RRF fusion       │         │
                          └──────────────┬──────────────────────┘         │
                                         ▼                                │
                              cross-encoder rerank (top-k)                │
                                         ▼                                │
                    ┌──────────────────────────────────────┐             │
   user question ──▶│  AGENT LOOP (Claude decides re-query) │             │
                    │  reason → search tool → observe → ...  │             │
                    └──────────────────┬───────────────────┘             │
                                       ▼                                  │
                       grounded answer + CITATION LEDGER                  │
                                       ▼                                  │
                          FastAPI  /ask   ·   /healthz                    │
                                       ▼                                  │
                        Docker image → (optional) GKE                     │
   Langfuse traces the whole path · Ragas scores it offline ─────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the component breakdown, and [`docs/MINIKUBE.md`](docs/MINIKUBE.md) for local Kubernetes deployment.

## Features

- **Ingestion pipeline** for financial PDFs with three switchable chunking strategies
  (fixed / recursive / semantic).
- **Hybrid retrieval** — dense vectors + sparse keyword search fused with Reciprocal Rank Fusion.
- **Two vector-DB backends** — pgvector (`halfvec` + HNSW + Postgres FTS) **and** Qdrant
  (`query_points`, named vectors) — switchable by config.
- **Cross-encoder reranking** (`BAAI/bge-reranker-v2-m3`) for a precision pass.
- **Agentic loop** — the model issues its own search queries and retrieves multiple times for
  multi-hop questions, with a hard step cap.
- **Citation / provenance** on every answer (source, page, chunk id, score), de-duplicated.
- **Evaluation** — Ragas (faithfulness, answer relevancy, context precision/recall) + Langfuse
  tracing, with published numbers.
- **FastAPI microservice**, containerized, with Kubernetes manifests.

## Production targets (from the eval suite)

| Metric | Target | Notes |
|---|---|---|
| Faithfulness | ≥ 0.90 | answer is grounded in retrieved context |
| Answer relevancy | ≥ 0.85 | answer addresses the question |
| Context precision | ≥ 0.80 | retrieved chunks are on-point |

> Replace these with your measured numbers once you run `scripts/run_eval.py`.

---

## Quick start

### 0. Prerequisites
- Python 3.12+
- Docker (for Qdrant and/or Postgres)
- API keys: `OPENAI_API_KEY` (embeddings), `ANTHROPIC_API_KEY` (generation)

### 1. Clone & install
```bash
git clone https://github.com/<you>/agentic-rag-financial.git
cd agentic-rag-financial
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # installs the package + dev tools
cp .env.example .env             # then fill in your keys
```

### 2. Start a vector store

**Option A — Qdrant via Docker (recommended for a quick demo):**
```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  -e QDRANT__SERVICE__API_KEY="$(openssl rand -hex 32)" \
  qdrant/qdrant:latest
```
- **Expose both ports.** 6333 is REST (dashboard at <http://localhost:6333/dashboard>), 6334 is
  gRPC — the Python client uses gRPC by default and will hang if only 6333 is open.
- The `API_KEY` line stops the instance being an open door. Put the same key in `.env` as
  `QDRANT_API_KEY`.
- For tests you don't even need Docker: `QdrantClient(":memory:")` or `QdrantClient(path="./qdrant_data")`.

**Option B — Postgres + pgvector via Docker:**
```bash
docker run -d --name pgvector \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16
# then set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb in .env
```

Pick your backend in `config.yaml` (`vector_store.backend: qdrant | pgvector`).

### 3. Get some documents
```bash
python scripts/download_filings.py        # downloads a few public SEC filings into ./data
```
EDGAR serves filings as **HTML**, so these save as `.htm` by default — the ingest pipeline extracts
clean text from HTML directly (which is actually better than PDF, where tables get mangled). If you
specifically want PDFs, add `--pdf` (requires `pip install pdfkit` + the `wkhtmltopdf` binary on your
PATH):
```bash
python scripts/download_filings.py --tickers AAPL MSFT JPM --forms 10-K 10-Q          # .htm
python scripts/download_filings.py --tickers AAPL MSFT JPM --forms 10-K 10-Q --pdf    # .pdf
```
(See **“Which documents to download”** below if you'd rather grab them by hand.)

### 4. Ingest
```bash
python scripts/ingest.py --config config.yaml --path "data/*"
# the "data/*" glob matches .htm AND .pdf; extracts → chunks → embeds → indexes; prints chunk count
```

### 5. Ask
```bash
# CLI
python scripts/ask.py --config config.yaml "What was total revenue in the most recent fiscal year?"

# or run the API
uvicorn agentic_rag.api:app --reload --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What was total revenue in the most recent fiscal year?"}' | jq
```

### 6. Evaluate (optional but recommended)
```bash
python scripts/run_eval.py --config config.yaml --eval-set data/eval_set.jsonl
# prints the Ragas table; traces appear in your Langfuse project
```

---

## Which PDFs to download

All free and public from **SEC EDGAR** (<https://www.sec.gov/cgi-bin/browse-edgar>) or company
investor-relations pages. Aim for **5–10 documents with variety**, because RAG only does real work
when the answer lives in one specific section.

**Good starter set** (search EDGAR full-text search at <https://efts.sec.gov/LATEST/search-index?q=>):

| Type | What to grab | Why it's useful |
|---|---|---|
| **10-K** (annual report) | Apple, Microsoft, or Tesla's latest 10-K | Dense, well-structured; great for "what was FY revenue / net income" factual Qs |
| **10-K** (second company) | A bank or insurer (e.g. JPMorgan) | Different structure; tests retrieval across document styles |
| **10-Q** (quarterly) | Any of the above, most recent quarter | Shorter, period-specific; tests date-scoped questions |
| **Credit / loan agreement** | Search EDGAR exhibits for "credit agreement" | Legal language, covenants — tests retrieval on non-narrative text |
| **Earnings release** (8-K exhibit) | A recent quarterly earnings 8-K | Tables and highlights; tests messy table extraction |

**How to download a 10-K by hand:**
1. Go to <https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany> and enter a ticker (e.g. `AAPL`).
2. Filter "Filing Type" to `10-K`, open the most recent, and download the primary document.
3. EDGAR serves these as HTML; either save as PDF from your browser, or use the `.pdf` exhibit if
   present. `scripts/download_filings.py` handles this conversion automatically.

> **Tip:** pick companies whose filings contain clear numeric answers (revenue, net income, segment
> breakdowns) and at least one document with covenants or risk-factor prose — that mix exercises both
> the dense (meaning) and sparse (exact-term) sides of hybrid retrieval.

---

## Configuration

Everything is driven by [`config.yaml`](config.yaml). Key knobs:

```yaml
vector_store:
  backend: qdrant            # qdrant | pgvector
chunking:
  strategy: recursive        # fixed | recursive | semantic
retrieval:
  use_hybrid: true           # dense + sparse via RRF
  use_reranker: true         # cross-encoder precision pass
  top_k: 5
agent:
  max_steps: 6               # hard cap on retrieval round-trips
```

## Project layout

```
agentic-rag-financial/
├── src/agentic_rag/
│   ├── config.py          # typed config + YAML loader
│   ├── ingest.py          # extract → chunk → embed → store
│   ├── chunking.py        # fixed / recursive / semantic
│   ├── embeddings.py      # OpenAI text-embedding-3-large
│   ├── stores.py          # pgvector + qdrant behind one interface
│   ├── retrieval.py       # hybrid (RRF) + cross-encoder rerank
│   ├── agent.py           # the agentic loop + provenance ledger
│   └── api.py             # FastAPI service (/ask, /healthz)
├── scripts/
│   ├── download_filings.py
│   ├── ingest.py · ask.py · run_eval.py
├── tests/                 # unit tests (pure logic, services stubbed)
├── data/                  # PDFs + eval_set.jsonl (gitignored except sample)
├── docs/ARCHITECTURE.md
├── k8s/                   # Deployment + Service manifests
├── Dockerfile · docker-compose.yml · config.yaml · pyproject.toml
└── .env.example · README.md · LICENSE
```

## Evaluation

The eval suite (`scripts/run_eval.py`) runs a labeled question set through the pipeline and reports
**Ragas** metrics, with every query **traced in Langfuse**. Build the eval set as JSONL:

```json
{"question": "What was total net revenue in the most recent fiscal year?", "ground_truth": "..."}
```

Then publish the resulting table in this README (replace the targets above).

## What I'd improve next

- **Conversational memory:** add per-session history + a "retrieve over past turns" tool so the
  agent handles follow-up questions ("and the year before?") without re-stating context.
- **True Qdrant hybrid:** add a SPLADE/BM42 sparse named vector and fuse with `query_points`
  prefetch, instead of the current keyword fallback.
- **Table-aware extraction:** financial tables become text soup in naive PDF extraction; a
  table-structure model would lift accuracy on numeric questions.
- **Prompt-caching** the system prompt + tool definitions to cut multi-step agent cost.
- **Streaming** the answer over SSE for a responsive UI.

## License

MIT — see [LICENSE](LICENSE).