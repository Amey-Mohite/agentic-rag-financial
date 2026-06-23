# Architecture

## Flow

**Ingest (once):**
`PDF → extract text (per page) → chunk (fixed/recursive/semantic) → embed (text-embedding-3-large,
3072-d, normalized) → store (pgvector halfvec+HNSW & FTS, or Qdrant named vectors)`

**Answer (per question):**
`question → agent loop: Claude decides to call search_documents → retriever embeds the query →
dense ANN + sparse keyword → RRF fusion → cross-encoder rerank → top-k chunks back to Claude →
(repeat for multi-hop) → grounded answer + provenance ledger`

## Components

| Module | Responsibility | Key choices |
|---|---|---|
| `config.py` | Typed config from YAML (+ `${ENV}`) | fail-fast validation |
| `chunking.py` | Split text into token chunks | 3 strategies; provenance per chunk |
| `embeddings.py` | Text → normalized vectors | same model for query + doc |
| `stores.py` | Persist + search vectors | pgvector **and** Qdrant, one interface |
| `retrieval.py` | Hybrid + rerank | RRF fusion, cross-encoder precision pass |
| `agent.py` | The agentic loop + ledger | `stop_reason` loop, max_steps cap, dedup citations |
| `api.py` | HTTP surface | `lifespan` startup, `/ask` + `/healthz` |

## Why two vector stores?

Not for production redundancy — to demonstrate a justified engineering choice.

- **pgvector** when vectors are a *feature* of an app already on Postgres: vectors live next to
  relational data, one fewer service to operate. Dense via `halfvec` + HNSW; sparse via native
  full-text search.
- **Qdrant** when vector search is the *core* workload: native ANN tuning, named vectors for hybrid,
  the modern `query_points` API, trivial to stand up as a container.

The retriever's interface is identical across both, so the backend is a one-line config change.

## The agent loop

```
messages = [question]
while steps <= max_steps:
    resp = claude(messages, tools=[search_documents])
    if resp.stop_reason != "tool_use":
        return answer + citations            # done
    for each tool_use block:
        hits = retriever.search(block.query)  # YOUR hybrid+rerank retriever
        ledger.update(hits)                   # keep best score per chunk id
    messages += tool_results                  # feed chunks back; loop
```

The model writes its own search queries and may retrieve multiple times (multi-hop). The **provenance
ledger** accumulates every chunk seen and ships with the answer, making each claim auditable — the
core requirement for finance/compliance use.

## Evaluation & observability

- **Ragas** scores faithfulness, answer relevancy, context precision/recall on a labeled set.
- **Langfuse** traces each query (retrieval, tokens, latency) so production traffic is observable the
  same way the eval set is.
