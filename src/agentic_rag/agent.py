"""
agent.py — the AGENTIC loop. This is what makes this "Agentic RAG" rather than plain RAG.

PLAIN RAG vs AGENTIC RAG
------------------------
Plain RAG is a fixed pipeline: ALWAYS retrieve once, then answer. Agentic RAG instead hands
the retriever to the model AS A TOOL and lets the MODEL decide: should I search? with what
query? do I need to search again for a second sub-question? When have I gathered enough to
answer? The model drives a loop of tool calls until it's ready to respond.

WHY THAT'S BETTER FOR FINANCIAL Q&A
-----------------------------------
Real questions are multi-part ("How did revenue change from FY22 to FY23 and why?"). A single
retrieval rarely covers every part. By letting the model issue several targeted searches, we
get better coverage. The trade-off is cost/latency, which we cap with `max_steps`.

THE PROVENANCE LEDGER
---------------------
Every chunk the model ever sees is recorded in a `ledger` keyed by chunk id, keeping the
HIGHEST score seen for each. When the model finally answers, we turn the ledger into a sorted,
de-duplicated citation list. So the answer always ships with "here's exactly what it was based
on (source, page, score)" — essential for trust in a financial setting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json  # used to serialize tool results into the string the API expects

# The config + the building blocks we assemble into the full pipeline.
from .config import AppConfig
from .embeddings import Embedder
from .stores import make_store, Hit
from .retrieval import Retriever


# The system prompt fixes the model's behavior: answer ONLY from retrieved facts, say "I don't
# know" otherwise, cite every claim, and prefer multiple searches over guessing. This is the
# single biggest lever on answer quality and groundedness.
SYSTEM_PROMPT = (
    "You are a financial-filings analyst. Answer ONLY using facts returned by the search_documents "
    "tool. If the documents do not contain the answer, say you don't know. For every factual claim, "
    "cite the source document and page. Prefer calling the tool more than once over guessing when a "
    "question has multiple parts."
)

# The tool definition we advertise to the model. The model never runs code itself — it emits a
# structured request to call "search_documents" with a `query`, and OUR loop executes the real
# retrieval and feeds the results back. `input_schema` is JSON Schema describing the arguments.
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
        "required": ["query"],  # the model MUST supply a query
    },
}


@dataclass
class AgentAnswer:
    """The structured result of answer().

    Fields
    ------
    text : str
        The final natural-language answer.
    citations : list[dict]
        The provenance ledger, sorted by score desc: each dict has chunk_id, source, page, score.
    steps : int
        How many tool/search calls the agent made (a complexity/cost signal).
    usage : dict
        Token usage totals {"input_tokens", "output_tokens"} summed across all model calls.
    """
    text: str
    citations: list[dict]
    steps: int
    usage: dict


class AgenticRAG:
    """The end-to-end system, assembled from an AppConfig.

    Construction wires together the embedder, vector store, retriever, and the Anthropic LLM
    client. `answer()` runs the agentic loop.
    """

    def __init__(self, cfg: AppConfig):
        """Validate config, then build every component the loop needs."""
        cfg.validate()                                    # fail fast on misconfiguration
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedding)           # query embedder
        self.store = make_store(cfg.vector_store, cfg.embedding.dims)  # vector DB backend
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval)  # search orchestrator
        from anthropic import Anthropic  # local import: only needed to actually run the agent
        self._llm = Anthropic()          # reads ANTHROPIC_API_KEY from the environment

    @classmethod
    def from_config(cls, path: str) -> "AgenticRAG":
        """Convenience constructor: load AppConfig from a YAML path, then build the system."""
        return cls(AppConfig.from_yaml(path))

    # ------------------------------------------------------------------------------------------
    def answer(self, question: str) -> AgentAnswer:
        """Run the agentic loop and return a grounded, cited answer.

        THE LOOP (in words):
          repeat up to max_steps:
            - call Claude with the question-so-far and the search tool available
            - if Claude returns a final answer (stop_reason != "tool_use") → done
            - else Claude asked to search: run the retriever for each requested query,
              record hits in the ledger, feed the chunks back as a tool_result, and loop

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        AgentAnswer
            text + citations + steps + usage.
        """
        # `messages` is the running conversation we send to the model each turn.
        messages = [{"role": "user", "content": question}]
        # The provenance ledger: chunk id -> the best (highest-scoring) Hit seen for that chunk.
        ledger: dict[int, Hit] = {}
        # Counters: number of search calls, and cumulative input/output tokens.
        steps = in_tok = out_tok = 0

        # Loop until the model answers or we hit the safety cap.
        while steps <= self.cfg.agent.max_steps:
            # --- Ask the model. It sees the system prompt, the search tool, and the convo so far.
            resp = self._llm.messages.create(
                model=self.cfg.generator.model,
                max_tokens=self.cfg.generator.max_tokens,
                temperature=self.cfg.generator.temperature,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL],
                messages=messages,
            )
            # Accumulate token usage for cost reporting.
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens

            # --- CASE A: the model did NOT ask for a tool → it produced the final answer.
            if resp.stop_reason != "tool_use":
                # Concatenate all text blocks into the answer string.
                text = "".join(b.text for b in resp.content if b.type == "text")
                # Turn the ledger into a citation list, sorted by score (best first).
                citations = [
                    {"chunk_id": h.id, "source": h.source, "page": h.page, "score": round(h.score, 4)}
                    for h in sorted(ledger.values(), key=lambda h: h.score, reverse=True)
                ]
                return AgentAnswer(text=text, citations=citations, steps=steps,
                                   usage={"input_tokens": in_tok, "output_tokens": out_tok})

            # --- CASE B: the model asked to use the tool. Record its turn, then service the calls.
            messages.append({"role": "assistant", "content": resp.content})  # echo back its request
            results = []  # collect one tool_result per tool_use block
            for block in resp.content:
                if block.type != "tool_use":
                    continue  # skip any text blocks the model included alongside the tool call
                # Run the REAL retriever with the model's requested query (block.input = {"query": ...}).
                hits = self.retriever.search(**block.input)
                steps += 1  # count this search
                # Merge hits into the ledger, keeping the highest score per chunk id (dedupe).
                for h in hits:
                    prev = ledger.get(h.id)
                    if prev is None or h.score > prev.score:
                        ledger[h.id] = h
                # Build the tool_result the API expects: it must reference the tool_use id and
                # carry a string content. We JSON-encode the chunks (id/source/page/text) for the model.
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps([
                        {"chunk_id": h.id, "source": h.source, "page": h.page, "text": h.text}
                        for h in hits]),
                })
            # Feed all tool results back as the next user turn, then loop so the model can continue.
            messages.append({"role": "user", "content": results})

        # --- SAFETY STOP: we exhausted max_steps without the model committing to an answer.
        citations = [{"chunk_id": h.id, "source": h.source, "page": h.page, "score": round(h.score, 4)}
                     for h in sorted(ledger.values(), key=lambda h: h.score, reverse=True)]
        return AgentAnswer(text="Stopped after max_steps without a final answer.",
                           citations=citations, steps=steps,
                           usage={"input_tokens": in_tok, "output_tokens": out_tok})
