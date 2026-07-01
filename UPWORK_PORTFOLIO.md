# Upwork Portfolio Showcase — Agentic RAG over Documents

Ready-to-paste copy for an Upwork **Portfolio** entry, plus a proposal blurb. Edit the bracketed
`[...]` bits and replace illustrative metrics with your own measured numbers from `scripts/run_eval.py`.

---

## 1. Portfolio title (pick one)
- **Production Agentic RAG — cited Q&A over your documents (web app + API)**
- **Document-Intelligence AI: hybrid retrieval, reranking, citations, streaming — deployable**
- **LLM RAG system with a bring-your-own-keys web UI, evaluation, and LLMOps**

## 2. One-line tagline
> An AI app that answers questions about your documents and **cites the exact source** (document,
> page, score) — with a web UI where anyone can plug in their own keys and try it in a browser.

## 3. Overview (paste into the description)
I built an end-to-end **Agentic RAG** system that answers natural-language questions over documents
(PDFs, HTML, text) and returns every answer with a **citation trail**. Unlike a one-shot RAG pipeline,
the LLM is given the retriever **as a tool** and decides what to search for and how many times — so
multi-part questions get full coverage.

It runs **hybrid retrieval** (dense embeddings + learned sparse SPLADE/BM42) with **cross-encoder
reranking**, supports **three vector stores** (zero-setup in-memory, Qdrant, and Postgres/pgvector via
Supabase), remembers **conversation context** for follow-ups, and **streams** answers token-by-token.
It ships as a **FastAPI** service with a **single-page web UI** (upload documents, ask questions, tune
every setting) plus Docker and Kubernetes manifests and an offline **evaluation** harness (Ragas +
Langfuse tracing).

## 4. Problem → solution
| Problem | What I built |
|---|---|
| LLMs hallucinate and can't see your private data | Retrieval-grounded answers with a citation ledger on every claim |
| One retrieval misses multi-part questions | An agentic loop where the model runs several targeted searches (with a step cap) |
| Keyword search misses meaning; vector search misses exact terms | Hybrid retrieval (dense + sparse) fused with Reciprocal Rank Fusion |
| Financial tables turn into unreadable "soup" | Table-aware ingestion that renders tables as structured Markdown |
| Follow-up questions lose context | Per-session conversational memory |
| Hard for non-technical users to try | A web UI where anyone pastes their own keys and uses it in the browser |

## 5. Key features (highlights box)
- Agentic, multi-hop retrieval — the model plans its own searches
- Source-traceable answers (document, page, chunk, score) on every response
- Hybrid retrieval (dense + SPLADE/BM42) + cross-encoder reranking
- Three swappable vector stores: in-memory · Qdrant · Postgres/pgvector (Supabase)
- Conversational memory · prompt caching · token streaming (SSE)
- Bring-your-own-keys web UI with full config controls
- Evaluation (Ragas) + tracing (Langfuse); Docker + Kubernetes

## 6. Tech stack (tags)
`Python` · `FastAPI` · `Anthropic Claude` · `OpenAI embeddings` · `RAG` · `Agentic AI` · `Qdrant` ·
`pgvector / Supabase` · `SPLADE / BM42 (fastembed)` · `cross-encoder reranking` · `Server-Sent Events` ·
`Docker` · `Kubernetes` · `Ragas` · `Langfuse` · `pytest`

## 7. Architecture (paste as an image or code block)
```
UPLOAD:  files → extract (table-aware) → chunk → embed (dense + sparse) → vector store
ASK:     question (+ memory) → [ LLM → search tool → retrieve ]* → grounded, cited answer (streamed)
                                 │
            retrieval: dense + sparse → RRF fusion → cross-encoder rerank → top-k
SERVE:   FastAPI  /  · /upload · /ask · /ask/stream   →   Docker → Hugging Face Spaces / k8s
OBSERVE: Langfuse traces every step · Ragas scores the eval set offline
```

## 8. Results / impact (replace with measured numbers)
Run `scripts/run_eval.py` on your own question set and paste real figures — reviewers trust specific,
owned numbers over round claims.

| Metric | Target | Meaning |
|---|---|---|
| Faithfulness | ≥ 0.90 | answer is grounded in retrieved context |
| Answer relevancy | ≥ 0.85 | answer actually addresses the question |
| Context precision | ≥ 0.80 | retrieved chunks are on-point |

Also quantify once measured: prompt-caching cost reduction across multi-step queries, latency-to-first-
token with streaming, and hybrid-vs-dense recall on your eval set.

## 9. What makes it stand out (proposal narrative)
- Not a notebook demo — a typed, tested, containerized service with three vector-DB backends, a real
  web UI, and an evaluation harness.
- Demonstrates the full RAG engineering discipline: chunking, hybrid retrieval, reranking, agentic
  orchestration, evaluation, and deployment — exactly the stack clients hire for.
- Every design choice is documented and justified (there's an annotated walkthrough notebook and a
  deep-dive doc), signaling maintainable, hand-off-ready work.

## 10. Services I can deliver (CTA)
> I build document-intelligence and RAG systems like this one for [legal, finance, healthcare, support,
> internal-knowledge] use cases. I can: stand up a **cited Q&A system over your documents**; upgrade an
> existing RAG app with **hybrid search, reranking, evaluation, and guardrails**; or ship a **production
> API** (streaming, auth, observability, Docker/K8s). Send me a sample of your documents and the
> questions you need answered, and I'll propose an approach and a measurable accuracy target.
> **[Your name] · [portfolio / GitHub link]**

## 11. Suggested portfolio media
- **Cover:** the architecture diagram (export ~1200×675).
- **Screenshot 1:** the **Ask** tab showing an answer **with its citation chips** (the trust-builder).
- **Screenshot 2:** the **Settings** tab (shows the depth — keys, vector store, all the knobs).
- **Optional:** a 60–90s Loom asking a multi-part question and showing the streamed, cited answer.
- Link the live demo (Hugging Face Space), the GitHub repo, and `agentic_rag_walkthrough.ipynb`.

## 12. Short blurb (~60 words, for the portfolio card / proposals)
> Built a production **Agentic RAG** app that answers questions about your documents and **cites its
> sources** (document, page, passage). Hybrid dense+sparse retrieval, cross-encoder reranking, an
> agentic multi-search loop, conversational memory, table-aware ingestion, prompt caching, and a
> streaming FastAPI service with a bring-your-own-keys web UI — three vector-store options, Docker,
> Kubernetes, and a Ragas/Langfuse evaluation suite.

---

### Quick tips
- Lead with **measurable outcomes** (accuracy, cost/latency), not model names.
- The **live demo link** (visitors bring their own keys) is your strongest asset — it lets a client
  try it in 30 seconds. Pre-warm it before sharing.
- Pair the portfolio entry with the deep-dive doc so a technical reviewer sees the depth.
