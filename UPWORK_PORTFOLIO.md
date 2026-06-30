# Agentic RAG over Financial Documents — Upwork Portfolio Showcase

> Ready-to-paste copy for an Upwork **Portfolio** entry, plus an "About this project" writeup and a
> short CTA you can reuse in proposals. Edit the bracketed `[...]` bits before publishing, and
> replace illustrative metrics with your own measured numbers from `scripts/run_eval.py`.

---

## 1. Portfolio title (pick one)

- **Production-Grade Agentic RAG for Financial Documents (cited answers over SEC filings)**
- **AI Q&A System over 10-Ks & 10-Qs — Hybrid Retrieval, Reranking, Citations, Streaming API**
- **LLM Document-Intelligence Pipeline: Agentic RAG with Source-Traceable Answers**

## 2. One-line tagline

> An evidence-grounded question-answering system over SEC filings: every answer is traced to the
> exact source document, page, and passage — served as a streaming, containerized API.

## 3. Project overview (paste into the description box)

In finance and compliance an answer is only useful if you can **trace it to its source**. I built
an end-to-end **Agentic RAG** system that answers natural-language questions about financial
filings (10-Ks, 10-Qs, credit agreements) and returns every answer with a **citation trail** —
source document, page, chunk, and relevance score.

Unlike a one-shot RAG pipeline, the LLM here is given the retriever **as a tool** and decides what
to search for and how many times — so multi-part questions ("How did revenue change from FY24 to
FY25, and why?") get full coverage. The system runs **hybrid retrieval** (dense embeddings +
learned sparse SPLADE/BM42 vectors) with **cross-encoder reranking**, supports **two vector-DB
backends** (Qdrant and Postgres/pgvector) behind one interface, remembers **conversation context**
for follow-up questions, and **streams** answers token-by-token over an API. It ships as a
**FastAPI microservice** with Docker + Kubernetes manifests, plus an offline **evaluation** suite
(Ragas + Langfuse).

## 4. The problem & my solution

| Problem | What I built |
|---|---|
| LLMs hallucinate financial figures | Answers are constrained to retrieved passages, with a **citation ledger** on every response |
| One retrieval misses multi-part questions | An **agentic loop** lets the model issue several targeted searches (with a hard step cap) |
| Keyword search misses meaning; vector search misses exact terms | **Hybrid retrieval** — dense + learned-sparse, fused with Reciprocal Rank Fusion, then reranked |
| Financial tables turn into "text soup" | **Table-aware ingestion** renders tables as structured Markdown so numbers stay aligned |
| Follow-up questions lose context | **Per-session conversational memory** |
| Multi-step agents are expensive & feel slow | **Prompt caching** to cut cost + **SSE streaming** for instant feedback |

## 5. Key features (bullet list for the highlights box)

- **Agentic, multi-hop retrieval** — the model plans its own searches.
- **Source-traceable answers** — citation (doc, page, chunk, score) on every claim.
- **True hybrid search** — dense vectors + SPLADE/BM42 sparse, fused server-side in Qdrant.
- **Cross-encoder reranking** for a precision pass (`BAAI/bge-reranker-v2-m3`).
- **Two swappable vector stores** — Qdrant and Postgres/pgvector — chosen by config.
- **Conversational memory** for natural follow-up questions.
- **Table-aware PDF ingestion** for reliable numeric Q&A.
- **Prompt caching + token streaming** for low cost and a responsive UX.
- **Evaluation** with Ragas metrics and Langfuse tracing.
- **Production packaging** — typed config, tests, Docker, Kubernetes.

## 6. Tech stack (tags)

`Python` · `LLM / Anthropic Claude` · `OpenAI embeddings` · `RAG` · `Agentic AI` · `Qdrant` ·
`pgvector / PostgreSQL` · `SPLADE / BM42 (fastembed)` · `Cross-encoder reranking` · `FastAPI` ·
`Server-Sent Events` · `Docker` · `Kubernetes` · `Ragas` · `Langfuse` · `pytest`

## 7. Architecture (one diagram, paste as an image or code block)

```
INGEST   files → extract (table-aware) → chunk → embed (dense + sparse) → vector store
RETRIEVE query → dense + sparse → RRF fusion (native in Qdrant) → cross-encoder rerank → top-k
AGENT    question (+ session memory) → [ LLM → search tool → observe ]* → grounded, cited answer
SERVE    FastAPI  /ask · /ask/stream (SSE) · /healthz   →   Docker → Kubernetes
OBSERVE  Langfuse traces every step · Ragas scores the eval set offline
```

## 8. Results / impact (replace with your measured numbers)

> These are the eval **targets** the suite checks against — run `scripts/run_eval.py` on your own
> question set and paste the real figures here. Reviewers trust specific, owned numbers far more
> than round claims.

| Metric | Target | Meaning |
|---|---|---|
| Faithfulness | ≥ 0.90 | answer is grounded in retrieved context |
| Answer relevancy | ≥ 0.85 | answer actually addresses the question |
| Context precision | ≥ 0.80 | retrieved chunks are on-point |

Other talking points to quantify once measured: prompt-caching cost reduction across multi-step
queries, latency-to-first-token with streaming, and hybrid-vs-dense recall on your eval set.

## 9. What makes this stand out (for the proposal narrative)

- It's **not a notebook demo** — it's a typed, tested, containerized service with two production
  vector-DB backends and an evaluation harness.
- It demonstrates the **full RAG engineering discipline**: chunking strategy, hybrid retrieval,
  reranking, agentic orchestration, evaluation, and deployment — the exact stack clients hire for.
- Every design choice is **documented and justified** (there's an annotated walkthrough notebook
  explaining the *why* behind each module), which signals maintainable, hand-off-ready work.

## 10. Services I can deliver (CTA block)

> I build document-intelligence and RAG systems like this one for [legal, finance, healthcare,
> support, internal-knowledge] use cases. I can:
> - stand up a **cited Q&A system over your documents** (PDFs, filings, contracts, wikis),
> - upgrade an existing RAG app with **hybrid search, reranking, evaluation, and guardrails**,
> - or ship a **production API** (streaming, auth, observability, Docker/K8s).
>
> Send me a sample of your documents and the questions you need answered, and I'll propose an
> approach and a measurable accuracy target. **[Your name] · [portfolio/GitHub link]**

## 11. Suggested portfolio media

- **Cover image:** the architecture diagram above (export as PNG, ~1200×675).
- **Screenshot 1:** a terminal/API response showing an answer **with its citations** (the
  trust-builder — blur any keys).
- **Screenshot 2:** the streaming endpoint producing tokens live, or the Langfuse trace of a
  multi-step query.
- **Optional:** a short Loom (60–90s) asking a multi-part question and showing the cited answer.
- **Link** the GitHub repo and the annotated walkthrough notebook.

## 12. Short blurb (for the portfolio card / proposals, ~60 words)

> Built a production Agentic RAG system that answers questions about SEC financial filings and
> **cites its sources** (document, page, passage). Features hybrid dense+sparse retrieval,
> cross-encoder reranking, an agentic multi-search loop, conversational memory, table-aware
> ingestion, prompt caching, and a streaming FastAPI service with Qdrant/pgvector backends, Docker,
> Kubernetes, and a Ragas/Langfuse evaluation suite.
