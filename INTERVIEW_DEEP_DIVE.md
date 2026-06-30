# Deep Dive — Concepts, Design Rationale, Interview Answers & Production Notes

A complete companion to the codebase. Read this to **understand every concept**, **explain why the
code is built the way it is**, **answer interview questions confidently**, and **talk credibly about
production**. Code references point at `src/agentic_rag/`.

---

## 0. The 60-second pitch (memorize this)

> "It's an **Agentic RAG** system that answers questions about documents and **cites its sources**.
> Documents are chunked, embedded, and stored in a vector database. At query time the LLM is given a
> search tool and **decides for itself** what to search and how many times — so multi-part questions
> get full coverage. Retrieval is **hybrid** (dense + sparse) with **cross-encoder reranking**, every
> answer ships with a **citation trail** (source, page, score), and it streams over a FastAPI service
> with a bring-your-own-keys web UI. It's typed, config-driven, tested, containerized, and has an
> evaluation harness with tracing."

That paragraph hits: RAG, agentic, retrieval engineering, grounding/citations, serving, and LLMOps —
the exact spread an interviewer is probing for.

---

## Part A — Concepts (what it is · why · interview framing)

### A1. RAG vs Agentic RAG
- **What:** RAG = Retrieval-Augmented Generation: fetch relevant text from your data, then have the
  LLM answer grounded in it. It fixes two LLM weaknesses — no knowledge of *your private data* and
  hallucination.
- **Plain RAG** = fixed pipeline: always retrieve once → stuff into prompt → answer.
- **Agentic RAG** (this project) = the retriever is a **tool** the model can call; the model loops:
  reason → search → observe → repeat → answer, bounded by `max_steps`.
- **Why it matters:** real questions are multi-part ("revenue this year *and* last year, and why?").
  One retrieval misses parts; letting the model issue several targeted searches covers them.
- **Code:** `agent.py` → `SEARCH_TOOL`, `answer()` loop checking `stop_reason == "tool_use"`.
- **Interview line:** *"The defining property of agentic RAG is that the model controls its own
  retrieval. In my code that's the loop in `answer()` — it keeps calling the search tool until it has
  enough, capped by `max_steps` so cost is bounded."*

### A2. Chunking — the #1 silent failure in RAG
- **What:** splitting documents into retrievable pieces, sized in **tokens**.
- **Why it matters most:** the retriever returns *whole chunks*. If the sentence that answers a
  question is split across a boundary, neither chunk holds the full fact, both embeddings are diluted,
  and retrieval quietly fails — no error, just a wrong answer. **Overlap** guards boundaries.
- **Three strategies (`chunking.py`):** `fixed` (every N tokens — predictable baseline), `recursive`
  (paragraph→sentence packing — best default), `semantic` (split on headings/topic shifts).
- **Interview line:** *"Chunking is where most RAG quality is won or lost. I size in tokens, keep
  overlap so a fact on a boundary survives, and prefer recursive splitting so I don't cut mid-sentence."*

### A3. Embeddings (dense vectors)
- **What:** a model turns text into a fixed-length vector (3072 numbers for `text-embedding-3-large`)
  that captures *meaning*; similar meaning → nearby vectors.
- **Why L2-normalize (`embeddings.py`):** after scaling to unit length, cosine similarity = dot
  product, which vector DBs compute fastest, and scores live in a consistent range.
- **Golden rule:** query and documents must use the **same** model + dimensions, or distances are
  meaningless. (That's why dims are tied to the model in the code.)
- **Interview line:** *"Dense embeddings give semantic search — it finds 'net sales' when you ask
  about 'revenue' even with no shared words."*

### A4. Sparse retrieval — keyword, SPLADE, BM42
- **What:** sparse vectors are mostly zeros — they store only the vocabulary terms that matter, each
  weighted. Classic version is BM25/keyword; **learned** versions (SPLADE, BM42) use a model to weight
  terms *and add related terms* (expansion).
- **Why:** dense embeddings blur exact terms, numbers, tickers, acronyms; sparse nails them. They
  cover each other's blind spots.
- **Code:** `embeddings.SparseEmbedder` (fastembed) + `stores.QdrantStore` named sparse vector.
- **Interview line:** *"SPLADE is 'smart keyword search the model learned' — it weights important
  words and expands synonyms, then I store it as a sparse vector and let Qdrant fuse it with dense."*

### A5. Hybrid retrieval + Reciprocal Rank Fusion (RRF)
- **What:** run dense **and** sparse, then merge the two ranked lists.
- **Why RRF (not adding scores):** dense cosine (~0–1) and sparse scores (unbounded) live on different
  scales — you can't add them. RRF fuses by **rank**: `score = Σ 1/(rrf_k + rank)`. Items high in
  *either* list rise; items in *both* rise most.
- **Code:** `retrieval.reciprocal_rank_fusion()` (app-side) or Qdrant's server-side `FusionQuery(RRF)`
  in `QdrantStore.hybrid_search()`.
- **Interview line:** *"RRF sidesteps incomparable score scales by using position. 60 is the standard
  smoothing constant."*

### A6. Reranking (cross-encoder)
- **What:** a bi-encoder (embeddings) scores query and chunk *separately*; a **cross-encoder** reads
  them *together* and outputs a precise relevance score. Slow but accurate.
- **Why the two-stage design:** cheap retrieval pulls ~20 candidates; the expensive cross-encoder
  re-scores only those to pick the best 5. You get precision without paying it on the whole corpus.
- **Code:** `retrieval.Retriever.search()` → `_get_reranker()` (`BAAI/bge-reranker-v2-m3`).
- **Interview line:** *"Retrieve-then-rerank: fast recall first, precise reranking on a small pool.
  It's the standard pattern for accuracy without blowing latency."*

### A7. Vector stores, ANN & HNSW
- **What:** a database that stores vectors and finds nearest neighbors fast. Exact search is O(N);
  **ANN** (approximate nearest neighbor) via **HNSW** (a navigable small-world graph) is ~O(log N).
- **Why it's not the bottleneck:** ANN over 148 chunks or 148 million is roughly constant time — your
  latency budget is the LLM calls, not the vector search.
- **Code:** `stores.py` — one `VectorStore` Protocol, two backends (Qdrant, pgvector) + an in-memory
  Qdrant. `make_store()` picks one.
- **Interview line:** *"I put both backends behind one interface so switching is a config change. HNSW
  gives sub-linear search; that's why corpus size barely affects query latency."*

### A8. The agent loop & tool use
- **What:** the LLM is told a tool exists (`search_documents`) with a JSON schema; it emits a
  structured request to call it; *our code* executes the real retrieval and feeds results back.
- **Why a step cap:** agents that loop unbounded burn money/time; `max_steps` is the safety valve.
- **Code:** `agent.answer()` / `answer_stream()` + `_run_tool_blocks()`.
- **Interview line:** *"The model never runs code — it requests a tool call, I run retrieval, return a
  `tool_result`, and loop. The cap bounds cost and prevents runaway loops."*

### A9. Provenance / citations
- **What:** a **ledger** records every chunk the model sees (keeping the max score per chunk id);
  the final answer ships with `[{source, page, chunk_id, score}]`.
- **Why:** in finance/legal/compliance an answer is only useful if you can trace it to the source. It
  also builds trust and lets a human verify.
- **Code:** the `ledger` dict in `answer()` → `_citations_from()`.
- **Interview line:** *"Groundedness isn't just a system prompt — I track exactly which chunks fed the
  answer and return them, so every claim is auditable."*

### A10. Conversational memory
- **What:** a per-session rolling history of clean (user, assistant) turns, replayed so follow-ups
  ("and the year before?") resolve in context. Bounded so context doesn't grow forever.
- **Why store only clean turns:** the bulky tool-call/tool-result blocks would bloat context and cost.
- **Code:** `memory.py` (`SessionStore` Protocol + bounded `deque`); injected via `_initial_messages()`.
- **Interview line:** *"I persist only the question/answer transcript, not the intermediate tool noise,
  and bound it — that's the production pattern. The store is an interface so I can swap in Redis."*

### A11. Prompt caching
- **What:** mark the static system prompt + tool schema with `cache_control: ephemeral` so the
  provider caches those tokens; repeat reads in a multi-step loop bill at ~10%.
- **Why:** the same large preamble is re-sent on every step of the loop; caching cuts cost and latency.
- **Code:** `agent._system_param()` / `_tools_param()`.
- **Interview line:** *"In an agent loop you re-send the system prompt every step. Caching it is a
  cheap, large cost win — I made it a config toggle."*

### A12. Streaming (Server-Sent Events)
- **What:** the answer is sent token-by-token over SSE; the tool steps are emitted as `step` events.
- **Why:** the model takes the same total time, but the user sees text in ~1s instead of staring at a
  spinner — and can cancel early. Table-stakes chat UX.
- **Why SSE not WebSockets:** one-directional server→client streaming; simpler, works over plain HTTP.
- **Code:** `agent.answer_stream()` (a generator) + `api.ask_stream()` (`StreamingResponse`).
- **Interview line:** *"Tool decisions happen up front — you can't stream a decision to search — but
  once the model writes the answer I stream tokens. SSE because it's one-way and HTTP-native."*

### A13. Table-aware ingestion
- **What:** PDFs' tables are detected (pdfplumber) and rendered as **Markdown pipe-tables** so rows/
  columns stay aligned, instead of flattening into "Revenue 2025 2024 294 298" soup.
- **Why:** numeric Q&A fails if the model can't tell which number belongs to which column/year.
- **Code:** `ingest._pdf_with_tables()` + `_table_to_markdown()`.

### A14. Evaluation & observability (LLMOps)
- **What:** an **eval harness** runs a labeled question set and scores it with **Ragas** (faithfulness,
  answer relevancy, context precision/recall); **Langfuse** traces every LLM/tool call.
- **Why:** "it looks right" doesn't scale. To improve or compare configs you must measure. This is the
  "Ops" layer buyers actually pay for.
- **Code:** `scripts/run_eval.py`.
- **Interview line:** *"I treat prompts/configs as things to measure, not guess. Faithfulness catches
  hallucination; context precision/recall tells me if retrieval or generation is the problem."*

---

## Part B — Why the code is structured this way (design decisions)

- **Typed, config-driven (`config.py` dataclasses).** Behavior changes by editing config, not code —
  easy to test, tune, and reason about. *Interview: "separation of policy from mechanism."*
- **Swappable backends behind a `Protocol` (`stores.py`, `memory.py`).** Qdrant ↔ pgvector ↔ in-memory
  with one factory call. *Interview: "program to an interface; the retriever doesn't know which DB."*
- **Lazy imports of heavy deps.** `openai`, `anthropic`, `qdrant`, `fastembed`, the reranker are
  imported inside functions, so importing a module is cheap and tests run offline.
- **Build the pipeline once (`api.lifespan`).** Clients/models are expensive — built at startup and
  reused per request, not per call.
- **Per-session pipelines + shared models.** Each web session gets its own keys/store; the heavy
  *local* models (reranker, sparse) are cached **process-wide** so memory stays bounded. *Interview:
  "multi-tenant on keys, single-tenant on models."*
- **Bring-your-own-keys.** Lets anyone test with their own credentials; keys held in memory per
  session, never logged. *Interview: honest about it being a demo pattern, not how you'd do prod auth.*
- **Provenance ledger + citations everywhere.** Trust and auditability are first-class, not bolted on.

---

## Part C — Rapid-fire interview Q&A

**Q: How is this different from just uploading a PDF to ChatGPT?**
A: That's a one-off lookup by a person. RAG is a *product* that answers over a corpus too big to paste,
for many users, automatically, with citations and access control. The user often doesn't even know
*which* document holds the answer — retrieval finds it. Plus cost/latency: you send ~5 relevant chunks
per query, not the whole document every time.

**Q: Why not just use a long context window and skip retrieval?**
A: Even 1–2M tokens can't hold an enterprise corpus, it's expensive (you pay for all tokens every
query), slow, and accuracy drops ("lost in the middle"). Retrieval keeps it cheap, fast, and precise.

**Q: How do you stop hallucination?**
A: Three layers — (1) a system prompt that says answer only from retrieved facts and say "I don't
know" otherwise; (2) the citation ledger so every claim is traceable; (3) a Ragas **faithfulness**
metric in the eval harness that flags ungrounded answers.

**Q: Dense vs sparse — when does each win?**
A: Dense wins on meaning/synonyms; sparse wins on exact terms, numbers, tickers, acronyms. Hybrid +
RRF gets both. On a small/512MB host I fall back to dense + keyword to save memory.

**Q: Why RRF instead of a weighted score blend?**
A: Dense and sparse scores aren't on comparable scales; blending needs fragile tuning. RRF fuses by
rank, which is scale-free and robust. 60 is the standard constant.

**Q: How do you evaluate it?**
A: A versioned JSONL question set through the pipeline, scored with Ragas (faithfulness, answer
relevancy, context precision/recall), traced in Langfuse, comparing two configs and printing a table.
Context precision/recall tells me whether a bad answer is a *retrieval* problem or a *generation* one.

**Q: How does the agent decide to stop searching?**
A: Each turn I check the model's `stop_reason`; if it's not `tool_use`, it produced a final answer.
A `max_steps` cap force-stops runaway loops.

**Q: How would you scale this to many users?**
A: See Part D — externalize memory to Redis, use a shared managed vector DB, make ingestion async
(queue + workers), put the reranker on GPU or a separate service, and add autoscaling + rate limiting.

---

## Part D — Production scenarios & problems (be ready for these)

This is where senior candidates separate themselves: knowing the limits of your own demo.

### D1. Memory / the reranker
- **Problem:** the cross-encoder is ~600 MB and needs ~2 GB RAM; on a small host it OOMs (you saw the
  Windows page-file error). 
- **Fixes:** turn reranker off on tiny tiers (`config.lite.yaml`); run it on GPU or as a separate
  microservice; share the model across sessions (already done); or use a hosted rerank API.

### D2. Persistence
- **Problem:** the in-memory vector store and in-process session memory are **ephemeral** — lost on
  restart/redeploy/sleep.
- **Fixes:** Qdrant Cloud or Supabase/pgvector for vectors (persistent); Redis/Postgres for sessions.

### D3. Multi-tenancy & scaling
- **Problem:** `replicas: 1` is required today because session memory + the indexed counter live
  in-process; the LRU session cap means evicted users lose their pipeline.
- **Fixes:** stateless app + shared Redis (sessions) + shared vector DB (already shared), then scale
  horizontally; per-tenant collections + access control on retrieval so users can't read each other's
  data.

### D4. Ingestion at scale
- **Problem:** upload ingests **synchronously** behind a lock — fine for a demo, but a 500-page PDF
  would block the request and serialize concurrent uploads.
- **Fixes:** async ingestion (Celery/RQ + a queue), chunked/streamed upload, progress via webhook or
  polling, and batch embedding (already batched at 128).

### D5. Cost & latency
- **Problem:** every question is several sequential LLM calls (the agent loop) + embedding calls; cost
  and latency scale with steps, not corpus size.
- **Fixes:** prompt caching (done), lower `max_steps`, smaller/cheaper models for simple queries, cache
  frequent queries, and stream for perceived speed.

### D6. Cold starts & model downloads
- **Problem:** first request downloads/loads the reranker (and SPLADE) — slow, and rate-limited on the
  HF Hub without a token. Serverless/scale-to-zero makes every cold start pay this.
- **Fixes:** pre-bake models in the image (the Dockerfile does), set `HF_TOKEN`, keep a warm instance,
  or warm on startup.

### D7. Security
- **Problems:** (1) holding user keys server-side is a demo pattern, not prod; (2) **prompt injection** —
  a malicious document could try to hijack the model; (3) PII in documents; (4) no per-user access
  control on retrieval; (5) the public demo link spends whoever's keys and is open.
- **Fixes:** secrets manager / per-request keys; input/output guardrails (Guardrails AI / NeMo) and
  treating retrieved text as untrusted; PII detection/redaction; row-level / collection-level access
  control; auth + rate limiting on the endpoints.

### D8. Reliability & provider limits
- **Problem:** OpenAI/Anthropic rate limits, timeouts, transient 5xx, quota exhaustion.
- **Fixes:** retries with exponential backoff, timeouts, circuit breakers, graceful degradation, and
  structured error responses (the API already turns failures into clean messages).

### D9. Data freshness & lifecycle
- **Problem:** the index is a snapshot; documents change, get deleted, or expire. There's no
  re-ingest/delete/version flow yet.
- **Fixes:** incremental ingestion, document versioning, deletion/TTL, and scheduled re-indexing.

### D10. Quality drift & monitoring
- **Problem:** quality can silently regress as data or models change.
- **Fixes:** the eval suite gates deploys (golden cases that must pass); Langfuse dashboards/alerts on
  faithfulness drift, error rate, latency, and cost; log + sample real traffic for offline scoring.

### D11. Correctness edge cases
- Tables across page boundaries, scanned/image PDFs (need OCR), non-English text, very long single
  chunks, empty retrieval ("I don't know" is the correct behavior), and conflicting sources (cite both).

> **Strong closing line for an interview:** *"The demo is intentionally simple where production would
> be hard — in-memory store, in-process memory, synchronous ingest, single replica. I know exactly
> which of those to replace first (persistent store → Redis sessions → async ingest → guardrails +
> auth) and why, which is the part that actually matters for shipping."*

---

## Part E — Map: concept → where it lives in the code

| Concept | File / function |
|---|---|
| Config (typed, env expansion) | `config.py` — `AppConfig`, `_expand_env` |
| Chunking strategies | `chunking.py` — `chunk_document`, `_fixed/_recursive/_semantic` |
| Dense embeddings | `embeddings.py` — `Embedder` |
| Sparse (SPLADE/BM42) | `embeddings.py` — `SparseEmbedder` |
| Vector stores + in-memory | `stores.py` — `VectorStore`, `QdrantStore`, `PgVectorStore`, `make_store` |
| Native hybrid (server-side RRF) | `stores.py` — `QdrantStore.hybrid_search` |
| App-side fusion | `retrieval.py` — `reciprocal_rank_fusion` |
| Retrieval + rerank | `retrieval.py` — `Retriever.search`, `_get_reranker` |
| Memory | `memory.py` — `SessionStore`, `InMemorySessionStore` |
| Agent loop + caching + streaming | `agent.py` — `answer`, `answer_stream`, `_system_param`, `_run_tool_blocks` |
| Ingestion + tables | `ingest.py` — `ingest_paths`, `extract`, `_pdf_with_tables` |
| API + per-session pipelines | `api.py` — `configure`, `upload`, `ask`, `ask_stream` |
| Web UI | `web/index.html` |
| Eval | `scripts/run_eval.py` |

Use **`agentic_rag_walkthrough.ipynb`** to see each of these run with annotated, offline demos.
