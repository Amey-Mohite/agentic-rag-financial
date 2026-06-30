"""
api.py — a FastAPI microservice that exposes the agent over HTTP, with a "bring your own keys" UI.

ENDPOINTS
---------
  GET  /                     -> the single-page web UI
  GET  /healthz              -> {"status": "ok"} (K8s probe) — works even with no keys configured
  POST /session/configure    -> {settings}  : build a per-session pipeline from the user's keys+config
  GET  /session/{sid}        -> whether a session is configured + its indexed counts
  POST /upload               -> multipart files (+ session_id) : ingest docs into that session's index
  POST /ask                  -> {question, session_id}         : grounded, cited answer
  POST /ask/stream           -> same, streamed as Server-Sent Events (step/token/final)

THE "BRING YOUR OWN KEYS" MODEL
-------------------------------
Anyone can open the UI, paste THEIR OWN OpenAI + Anthropic keys, choose where vectors are stored
(zero-setup in-memory, or their Qdrant Cloud), tune the config, and test — without the operator's
keys. Each browser session gets its OWN pipeline (its own clients, keys, and vector data), built on
`/session/configure` and cached server-side keyed by a random session id the browser generates.

SECURITY NOTE: keys are held only in server memory for the life of the session and are never logged
or returned. This is a demo pattern; for production you'd avoid sending keys to a shared server.

WHY A SESSION REGISTRY (not one global pipeline): keys/config differ per user, and an in-memory
vector DB lives inside its client — so each session needs its own pipeline. Heavy LOCAL models
(reranker, sparse) are still shared process-wide (see retrieval/embeddings caches), so memory stays
bounded; we also cap how many sessions are kept (LRU eviction).
"""

from __future__ import annotations

import os
import json
import shutil
import pathlib
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from .agent import AgenticRAG
from .config import (AppConfig, EmbeddingConfig, GeneratorConfig, VectorStoreConfig,
                     ChunkingConfig, RetrievalConfig, AgentConfig, MemoryConfig, IngestionConfig)


# Optional default config (used only if the operator set env keys); the UI normally drives everything.
CONFIG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
UPLOAD_DIR = pathlib.Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
WEB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "web"
INDEX_HTML = WEB_DIR / "index.html"

ALLOWED_EXT = {".pdf", ".htm", ".html", ".txt", ".md"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "6"))   # cap cached pipelines (LRU evicts oldest)

# Known embedding models → their output dimensionality (must match the vector store schema).
EMBED_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}

_INGEST_LOCK = threading.Lock()


class Session:
    """One user's pipeline + bookkeeping, kept in the in-memory registry."""
    def __init__(self, rag: AgenticRAG):
        self.rag = rag
        self.stats = {"documents": 0, "chunks": 0}
        self.last_used = time.time()


# session_id -> Session. OrderedDict gives us cheap LRU (move_to_end on use, popitem(last=False) to evict).
_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
_SESSIONS_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------
# Request/response schemas
# ----------------------------------------------------------------------------------------------
class RuntimeSettings(BaseModel):
    """Everything the Settings page sends to build a session pipeline. All but the keys have
    sensible defaults, so a minimal request is just the two API keys."""
    session_id: str
    # --- API keys (the user's own) ---
    openai_api_key: str
    anthropic_api_key: str
    # --- where vectors are stored ---
    # "memory" (zero setup) | "qdrant" (Qdrant Cloud) | "pgvector" (Supabase / any Postgres)
    vector_store: str = "memory"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    postgres_dsn: str | None = None         # Supabase/Postgres connection string (when vector_store=pgvector)
    # --- models ---
    embedding_model: str = "text-embedding-3-large"
    generator_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0
    use_prompt_caching: bool = True
    # --- chunking ---
    chunk_strategy: str = "recursive"       # fixed | recursive | semantic
    chunk_tokens: int = 500
    overlap_tokens: int = 50
    # --- retrieval ---
    top_k: int = 5
    candidate_pool: int = 20
    use_hybrid: bool = True
    use_reranker: bool = True
    sparse_backend: str = "keyword"         # keyword | splade (splade needs the [hybrid] extra)
    rrf_k: int = 60
    # --- agent / memory / ingestion ---
    max_steps: int = 4
    memory_enabled: bool = True
    max_turns: int = 6
    extract_tables: bool = True


class AskRequest(BaseModel):
    """POST /ask body. `session_id` selects the pipeline AND the conversation memory."""
    question: str
    session_id: str


class Citation(BaseModel):
    chunk_id: int
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    steps: int
    usage: dict


# ----------------------------------------------------------------------------------------------
# Pipeline construction from runtime settings
# ----------------------------------------------------------------------------------------------
def _build_config(s: RuntimeSettings) -> AppConfig:
    """Turn the UI's RuntimeSettings into a typed AppConfig (with keys + chosen vector store)."""
    dims = EMBED_DIMS.get(s.embedding_model, 3072)   # dimensionality is fixed by the model
    # Choose the vector store backend from the UI selection:
    if s.vector_store == "memory":
        # Embedded in-process Qdrant (url ":memory:") — zero setup, ephemeral.
        vs = VectorStoreConfig(backend="qdrant", url=":memory:", api_key=None, collection="filings")
    elif s.vector_store == "pgvector":
        # Supabase / any Postgres with the pgvector extension. dsn = the connection string.
        if not s.postgres_dsn:
            raise HTTPException(400, "Postgres/Supabase selected but no connection string provided")
        vs = VectorStoreConfig(backend="pgvector", dsn=s.postgres_dsn, collection="filings")
    else:  # "qdrant"
        if not s.qdrant_url:
            raise HTTPException(400, "Qdrant selected but no qdrant_url provided")
        vs = VectorStoreConfig(backend="qdrant", url=s.qdrant_url,
                               api_key=s.qdrant_api_key, collection="filings")
    return AppConfig(
        embedding=EmbeddingConfig(model=s.embedding_model, dims=dims),
        generator=GeneratorConfig(model=s.generator_model, max_tokens=s.max_tokens,
                                  temperature=s.temperature, use_prompt_caching=s.use_prompt_caching),
        vector_store=vs,
        chunking=ChunkingConfig(strategy=s.chunk_strategy, chunk_tokens=s.chunk_tokens,
                                overlap_tokens=s.overlap_tokens),
        retrieval=RetrievalConfig(top_k=s.top_k, candidate_pool=s.candidate_pool,
                                  use_hybrid=s.use_hybrid, use_reranker=s.use_reranker,
                                  sparse_backend=s.sparse_backend, rrf_k=s.rrf_k),
        agent=AgentConfig(max_steps=s.max_steps),
        memory=MemoryConfig(enabled=s.memory_enabled, max_turns=s.max_turns),
        ingestion=IngestionConfig(extract_tables=s.extract_tables),
        openai_api_key=s.openai_api_key,
        anthropic_api_key=s.anthropic_api_key,
    )


def _get_session(session_id: str) -> Session:
    """Look up a configured session or raise a friendly 409 telling the UI to configure first."""
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is None:
            raise HTTPException(409, "No pipeline for this session. Open Settings and click 'Save & connect' first.")
        _SESSIONS.move_to_end(session_id)   # mark as most-recently-used (LRU)
        sess.last_used = time.time()
        return sess


# ----------------------------------------------------------------------------------------------
# App + lifespan
# ----------------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot WITHOUT requiring keys. If the operator set env keys + a config, we pre-build a default
    pipeline; otherwise the app still starts and waits for users to configure their own session."""
    app.state.default_rag = None
    try:
        if os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
            app.state.default_rag = AgenticRAG.from_config(CONFIG_PATH)
    except Exception as e:                          # never let a config problem stop the app booting
        print(f"[startup] no default pipeline ({e}); waiting for per-session configuration")
    yield


app = FastAPI(title="Agentic RAG — Document Q&A", lifespan=lifespan)


# ----------------------------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------------------------
@app.get("/")
def index():
    """Serve the single-page web UI (web/index.html)."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    raise HTTPException(status_code=404, detail="web/index.html not found")


@app.get("/healthz")
def healthz():
    """Liveness probe — intentionally has NO dependency on keys so the container reports healthy."""
    return {"status": "ok", "sessions": len(_SESSIONS)}


@app.post("/session/configure")
def configure(settings: RuntimeSettings):
    """Build (or rebuild) this session's pipeline from the user's keys + config, and validate it.

    Validation does the cheap, high-signal checks: connect to the vector store (`store.setup()`)
    and embed a tiny string to confirm the OpenAI key works. The Anthropic key is validated on the
    first question (to avoid spending tokens here); errors there are surfaced clearly.
    """
    cfg = _build_config(settings)
    try:
        rag = AgenticRAG(cfg)              # constructs clients (no network calls yet)
        rag.store.setup()                 # connects to / creates the vector collection
        rag.embedder.embed_query("ping")  # 1 tiny embed → verifies the OpenAI key + connectivity
    except HTTPException:
        raise
    except Exception as e:
        # Turn SDK/connection errors into a clean message the UI can show next to the form.
        raise HTTPException(400, f"Could not connect with these settings: {e}")

    # Store/replace the session pipeline; evict the oldest if we're over the cap.
    with _SESSIONS_LOCK:
        _SESSIONS[settings.session_id] = Session(rag)
        _SESSIONS.move_to_end(settings.session_id)
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)   # drop least-recently-used session (frees its keys + data)

    return {"status": "ok", "vector_store": settings.vector_store,
            "reranker": settings.use_reranker, "sparse": settings.sparse_backend}


@app.get("/session/{session_id}")
def session_info(session_id: str):
    """Tell the UI whether this session is configured and how much it has indexed."""
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
    if sess is None:
        return {"configured": False, "documents": 0, "chunks": 0}
    return {"configured": True, **sess.stats}


@app.post("/upload")
async def upload(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    """Validate, save, and ingest documents into THIS session's vector index (reusing its warm models)."""
    sess = _get_session(session_id)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[pathlib.Path] = []

    for f in files:
        ext = pathlib.Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
        # Namespace the saved file by session so different users' uploads don't collide on disk.
        dest = UPLOAD_DIR / f"{session_id}_{pathlib.Path(f.filename).name}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            dest.unlink(missing_ok=True)
            raise HTTPException(413, f"'{f.filename}' is {size_mb:.1f}MB > {MAX_UPLOAD_MB}MB limit")
        saved.append(dest)

    results = []
    with _INGEST_LOCK:                     # serialize ingest (shared model loads / store writes)
        for path in saved:
            try:
                n = sess.rag.ingest([str(path)])
            except Exception as e:
                raise HTTPException(500, f"failed to ingest '{path.name}': {e}")
            finally:
                # The file was only needed to extract → chunk → embed. The searchable data now
                # lives as VECTORS in the store, so delete the raw file to keep the server stateless
                # and avoid disk bloat (important on small/ephemeral hosts).
                path.unlink(missing_ok=True)
            sess.stats["documents"] += 1
            sess.stats["chunks"] += n
            results.append({"filename": path.name.split("_", 1)[-1], "chunks": n})

    return {"results": results, "totals": sess.stats}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question using this session's pipeline (session_id also drives conversation memory)."""
    sess = _get_session(req.session_id)
    try:
        result = sess.rag.answer(req.question, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(502, f"agent error (check your Anthropic/OpenAI keys & quota): {e}")
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
        steps=result.steps,
        usage=result.usage,
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Stream this session's answer as Server-Sent Events (step → token → final)."""
    sess = _get_session(req.session_id)

    def event_generator():
        try:
            for event in sess.rag.answer_stream(req.question, session_id=req.session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            # Surface errors to the client as a final event instead of dropping the stream.
            yield f"data: {json.dumps({'type': 'final', 'answer': f'Error: {e}', 'citations': [], 'steps': 0, 'usage': {}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
