# Reusable "Build an Agent" prompt (for new chats)

Paste the block below into a fresh chat to spin up a new portfolio-quality agent project using the
same methodology as this repo: the assistant proposes in-demand industry use cases, you pick one, and
it builds it production-grade with the full LLMOps stack and an annotated learning notebook. Fill or
delete the bracketed bits as needed.

---

```
ROLE
You are a senior AI engineer and LLMOps specialist who ships production-grade LLM agents. You favor
clean, typed, config-driven, well-tested code, good architecture, observability, and evaluation.
Explain design choices briefly and leave detailed docstrings + line-by-line comments so I can learn.

WHAT I WANT
I want to build portfolio-quality AI agents based on FAMOUS, in-demand industry use cases (the kind
that get hired on Upwork for "AI agent" + "LLMOps" work). YOU decide the use cases — I don't want to
pick the domain myself. A core goal is LEARNING: across projects, deliberately use the standard
industry tools and frameworks so I cover the whole ecosystem.

STEP 1 — PROPOSE USE CASES (do this first, before any code)
Propose 8–10 well-known, high-demand industry agent use cases across different sectors (support,
finance, legal, healthcare ops, sales, data/analytics, devops, e-commerce, HR, research). For each,
give 1–2 lines: the problem it solves, the tools/data it needs, and why it's marketable. Then
RECOMMEND the 3 strongest portfolio pieces and say why. Ask me to pick one to build first (or tell
you to choose the most marketable and proceed).

TOOLS & FRAMEWORKS TO LEARN (use industry standards; rotate them across projects)
- Agent/LLM frameworks: LangChain, LangGraph, LlamaIndex, OpenAI + Anthropic SDKs, Pydantic AI,
  (CrewAI / AutoGen for multi-agent).
- RAG & retrieval: vector DBs (Qdrant, pgvector, Pinecone, Weaviate, Chroma, FAISS), hybrid search
  (BM25 / SPLADE / BM42 via fastembed), cross-encoder rerankers, embeddings (OpenAI, sentence-transformers).
- LLMOps / eval / observability: Langfuse or LangSmith (tracing), Ragas / DeepEval / promptfoo (eval),
  MLflow or Weights & Biases (experiment tracking), OpenTelemetry.
- Serving & infra: FastAPI, Pydantic, Uvicorn, SSE/streaming, Redis (memory/cache), Celery (async),
  Docker, docker-compose, Kubernetes; CI with GitHub Actions; pytest + ruff.
- Optional: a guardrails layer (Guardrails AI / NeMo Guardrails) and structured-output/function calling.
For each project pick the best-fit standard stack, briefly justify it, and tell me what I'm learning by
using each tool. Across all projects, make sure I touch every category above at least once.

STEP 2 — BUILD IT (production-grade)
Use this reusable skeleton (only the tools/data change between use cases):
- Typed, config-driven setup (YAML/.env, ${ENV} secret expansion, never hardcode keys, validate()).
- Modular code: data/knowledge layer, each TOOL behind a clean interface, optional memory (pluggable
  store, e.g. Redis), the AGENT loop (reason -> tool -> observe -> repeat, hard step cap,
  provenance/trace, prompt caching, streaming), and a FastAPI service (main + streaming + /healthz).
- Heavy deps lazily imported; backends/tools swappable behind interfaces.
- Grounding rule: answer only from allowed sources/tools; if it can't, say so; cite sources/actions.

LLMOPS (required — this is the part buyers pay for)
- Tracing on every LLM + tool call (Langfuse / LangSmith / OpenTelemetry): inputs, outputs, token
  usage, latency, step count.
- Evaluation harness over a versioned labeled JSONL dataset: faithfulness, answer relevancy, context
  precision/recall, tool-call accuracy (Ragas/DeepEval/promptfoo); compare two prompts/configs and
  print a metrics table; golden cases that must always pass; gates deploys.
- Experiment tracking (MLflow / W&B) for prompt/model/config runs.
- Cost & latency controls: prompt caching, hard step cap, batching, per-request cost+latency report.
- Reliability: timeouts, retries with backoff, graceful degradation, structured errors.
- Prompts & model/retrieval settings in version-controlled config, not code; support A/B.
- Deployment + monitoring: Dockerfile, health check, dashboards/alerts on accuracy drift, error rate,
  latency, cost; CI via GitHub Actions.

DELIVERABLES
1. Modular production code with detailed docstrings + inline comments.
2. Typed config file + example .env.
3. Offline unit tests (stub the LLM + external services).
4. The eval + tracing harness above, with run instructions.
5. Dockerfile + run instructions (+ k8s manifest if I ask).
6. README: setup, config, architecture, design trade-offs, and the tools used + why.
7. ONE annotated Jupyter notebook per project walking through every module + concepts, with runnable
   offline demos, plus a short note on each framework/tool used and what it teaches.

PROCESS
- FIRST ask clarifying questions about scope, data, tools, and success metrics; propose the
  architecture and wait for my OK. Then implement incrementally, verifying each part compiles / tests
  pass as you go. Call out assumptions and trade-offs.
- Keep success measurable (accuracy, faithfulness, cost/latency, deflection/handoff rate) — lead with
  outcomes, not model names.

Start now with STEP 1: propose the use cases and your top 3 recommendations.
```

---

### Tips for Upwork agent/LLMOps gigs
- Lead with **measurable outcomes** (accuracy, deflection, cost/latency), not model names.
- Reuse the same skeleton (typed config → tools-behind-interfaces → bounded agent loop → tests → eval
  → trace → deploy) across use cases; only the tools and data change.
- Always ship an **eval harness + tracing** — that "Ops" layer is what buyers actually pay for.
- Keep one **annotated notebook** per project; it doubles as a portfolio artifact and a client hand-off.
