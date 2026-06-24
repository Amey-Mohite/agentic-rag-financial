"""
api.py — a FastAPI microservice that exposes the agent over HTTP.

ENDPOINTS
---------
  POST /ask      body {"question": "..."} -> {answer, citations, steps, usage}
  GET  /healthz                           -> {"status": "ok"}  (Kubernetes liveness probe)

HOW TO RUN
----------
  uvicorn agentic_rag.api:app --reload --port 8000

KEY DESIGN POINT: BUILD THE PIPELINE ONCE
-----------------------------------------
Constructing AgenticRAG creates API clients, loads config, and (on first search) a reranker —
all expensive. We do it ONCE at startup via the `lifespan` hook and stash it on `app.state`,
so every request reuses the same warm pipeline instead of rebuilding it per call.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager  # for the startup/shutdown lifespan context manager

from fastapi import FastAPI
from pydantic import BaseModel  # request/response schemas with automatic validation

from .agent import AgenticRAG


# Which config file to load. Overridable via the RAG_CONFIG env var; defaults to config.yaml.
CONFIG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")


class AskRequest(BaseModel):
    """Schema for the POST /ask request body."""
    question: str  # the user's natural-language question


class Citation(BaseModel):
    """One provenance entry returned alongside the answer."""
    chunk_id: int  # stable id of the cited chunk
    source: str    # document name
    page: int      # page within that document
    score: float   # relevance score (post-rerank / fused)


class AskResponse(BaseModel):
    """Schema for the POST /ask response body. FastAPI validates outgoing data against this."""
    answer: str
    citations: list[Citation]
    steps: int
    usage: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Code before `yield` runs ONCE at startup; after, at shutdown.

    We build the (expensive) AgenticRAG pipeline here and store it on `app.state.rag`. WHY:
    creating API clients and loading models per-request would make every call slow and wasteful.
    """
    app.state.rag = AgenticRAG.from_config(CONFIG_PATH)  # warm the pipeline once
    yield  # ← the app serves requests while suspended here; teardown (if any) would go after


# Create the FastAPI app and attach the lifespan hook above.
app = FastAPI(title="Agentic RAG over Financial Documents", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """Liveness probe for orchestrators (Kubernetes). Returns {'status': 'ok'} when up."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question with the agent and return the answer plus its provenance ledger.

    Parameters
    ----------
    req : AskRequest
        Parsed/validated JSON body containing `question`.

    Returns
    -------
    AskResponse
        answer + citations + steps + usage.
    """
    # Run the agent using the pre-built pipeline stored on app.state.
    result = app.state.rag.answer(req.question)
    # Repackage the AgentAnswer into the response schema (Citation(**c) validates each dict).
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
        steps=result.steps,
        usage=result.usage,
    )
