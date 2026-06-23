"""
api.py — FastAPI microservice exposing the agent.

  POST /ask     {"question": "..."}  -> {answer, citations, steps, usage}
  GET  /healthz                      -> {"status": "ok"}   (Kubernetes liveness probe)

Run:  uvicorn agentic_rag.api:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import AgenticRAG


CONFIG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")


class AskRequest(BaseModel):
    """POST /ask body. question: the user's query."""
    question: str


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the (expensive) pipeline once at startup. WHY: avoid re-creating clients per request."""
    app.state.rag = AgenticRAG.from_config(CONFIG_PATH)
    yield


app = FastAPI(title="Agentic RAG over Financial Documents", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """Liveness probe. RETURNS {'status': 'ok'}."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question with the agent and return the provenance ledger.

    PARAM req: AskRequest.
    RETURNS: AskResponse(answer, citations, steps, usage).
    """
    result = app.state.rag.answer(req.question)
    return AskResponse(
        answer=result.text,
        citations=[Citation(**c) for c in result.citations],
        steps=result.steps,
        usage=result.usage,
    )
