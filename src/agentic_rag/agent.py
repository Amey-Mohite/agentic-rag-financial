"""
agent.py — the agentic loop. The model is given the retriever as a TOOL and decides whether, what,
and how many times to search before answering. Every retrieved chunk is recorded in a provenance
ledger so the final answer ships with citations (source, page, score), de-duplicated by chunk id.

This is what turns a one-pass RAG pipeline into *Agentic* RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from .config import AppConfig
from .embeddings import Embedder
from .stores import make_store, Hit
from .retrieval import Retriever


SYSTEM_PROMPT = (
    "You are a financial-filings analyst. Answer ONLY using facts returned by the search_documents "
    "tool. If the documents do not contain the answer, say you don't know. For every factual claim, "
    "cite the source document and page. Prefer calling the tool more than once over guessing when a "
    "question has multiple parts."
)

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Search the SEC financial filings for passages relevant to a query. Use whenever you need "
        "facts from the documents. You may call it multiple times with different queries to answer "
        "multi-step questions. Returns the most relevant chunks with source document and page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The specific thing you need to find."},
        },
        "required": ["query"],
    },
}


@dataclass
class AgentAnswer:
    """What answer() returns — grounded text + provenance + cost/step signals."""
    text: str
    citations: list[dict]
    steps: int
    usage: dict


class AgenticRAG:
    """End-to-end agentic RAG assembled from AppConfig."""

    def __init__(self, cfg: AppConfig):
        cfg.validate()
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedding)
        self.store = make_store(cfg.vector_store, cfg.embedding.dims)
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval)
        from anthropic import Anthropic
        self._llm = Anthropic()          # reads ANTHROPIC_API_KEY

    @classmethod
    def from_config(cls, path: str) -> "AgenticRAG":
        return cls(AppConfig.from_yaml(path))

    # ------------------------------------------------------------------------------------------
    def answer(self, question: str) -> AgentAnswer:
        """Run the agent loop and return a grounded, cited answer.

        Loop: ask Claude with the search tool → if it requests a search, run the retriever and feed
        chunks back → repeat until it answers or max_steps is hit. The provenance ledger accumulates
        every chunk seen, keeping the highest score per chunk id.
        PARAM question: the user's natural-language question.
        RETURNS: AgentAnswer(text, citations, steps, usage).
        """
        messages = [{"role": "user", "content": question}]
        ledger: dict[int, Hit] = {}
        steps = in_tok = out_tok = 0

        while steps <= self.cfg.agent.max_steps:
            resp = self._llm.messages.create(
                model=self.cfg.generator.model,
                max_tokens=self.cfg.generator.max_tokens,
                temperature=self.cfg.generator.temperature,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL],
                messages=messages,
            )
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                citations = [
                    {"chunk_id": h.id, "source": h.source, "page": h.page, "score": round(h.score, 4)}
                    for h in sorted(ledger.values(), key=lambda h: h.score, reverse=True)
                ]
                return AgentAnswer(text=text, citations=citations, steps=steps,
                                   usage={"input_tokens": in_tok, "output_tokens": out_tok})

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                hits = self.retriever.search(**block.input)
                steps += 1
                for h in hits:
                    prev = ledger.get(h.id)
                    if prev is None or h.score > prev.score:
                        ledger[h.id] = h
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps([
                        {"chunk_id": h.id, "source": h.source, "page": h.page, "text": h.text}
                        for h in hits]),
                })
            messages.append({"role": "user", "content": results})

        # safety stop
        citations = [{"chunk_id": h.id, "source": h.source, "page": h.page, "score": round(h.score, 4)}
                     for h in sorted(ledger.values(), key=lambda h: h.score, reverse=True)]
        return AgentAnswer(text="Stopped after max_steps without a final answer.",
                           citations=citations, steps=steps,
                           usage={"input_tokens": in_tok, "output_tokens": out_tok})
