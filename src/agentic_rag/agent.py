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
    "You are a helpful assistant that answers questions about the user's OWN uploaded documents — "
    "which may be insurance policies, contracts, financial filings, reports, or any other text. "
    "ALWAYS call the search_documents tool to look for the answer before responding, even for short or "
    "simple questions. IMPORTANT: the documents belong to the user, so details that look 'personal' — "
    "names, policy/account numbers, dates, registration numbers, amounts — ARE in these documents and "
    "should be retrieved and returned; do NOT refuse them as private information. Answer ONLY using "
    "facts found via the tool, and cite the source document and page for each fact. Only say you don't "
    "know if the documents genuinely don't contain the answer (after searching). Prefer calling the "
    "tool more than once for multi-part questions."
)

# The tool definition we advertise to the model. The model never runs code itself — it emits a
# structured request to call "search_documents" with a `query`, and OUR loop executes the real
# retrieval and feeds the results back. `input_schema` is JSON Schema describing the arguments.
SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Search the user's uploaded documents for passages relevant to a query. Use it for EVERY "
        "question that could be answered from the documents — including looking up specific values "
        "like names, policy numbers, dates, or amounts. You may call it multiple times with different "
        "queries to answer multi-step questions. Returns the most relevant chunks with source and page."
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
        # Pass the (optional) runtime OpenAI key explicitly; falls back to env when None.
        self.embedder = Embedder(cfg.embedding, api_key=cfg.openai_api_key)  # dense query embedder

        # Build a learned-sparse embedder when configured, for true hybrid retrieval.
        # GRACEFUL FALLBACK: sparse_backend="splade" needs the optional `fastembed` package. If it
        # isn't installed, degrade to keyword retrieval instead of crashing at ingest time.
        enable_sparse = cfg.retrieval.sparse_backend == "splade"
        self.sparse_embedder = None
        if enable_sparse:
            try:
                import fastembed  # noqa: F401  (import check — fails fast if the extra is missing)
                from .embeddings import SparseEmbedder
                self.sparse_embedder = SparseEmbedder(cfg.retrieval)
            except Exception as e:
                print(f"[warn] sparse_backend='splade' but fastembed is unavailable ({e}); "
                      f"falling back to keyword retrieval. Install with: pip install fastembed")
                enable_sparse = False

        self.store = make_store(cfg.vector_store, cfg.embedding.dims, enable_sparse=enable_sparse)
        # The retriever gets the sparse embedder so it can use native hybrid where supported.
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval,
                                   sparse_embedder=self.sparse_embedder)

        # Conversational memory store (per-session history) — see memory.py.
        from .memory import make_session_store
        self.sessions = make_session_store(cfg.memory) if cfg.memory.enabled else None

        from anthropic import Anthropic  # local import: only needed to actually run the agent
        # Pass the (optional) runtime Anthropic key explicitly; falls back to env when None.
        self._llm = Anthropic(api_key=cfg.anthropic_api_key) if cfg.anthropic_api_key else Anthropic()

    @classmethod
    def from_config(cls, path: str) -> "AgenticRAG":
        """Convenience constructor: load AppConfig from a YAML path, then build the system."""
        return cls(AppConfig.from_yaml(path))

    def ingest(self, paths: list[str]) -> int:
        """Ingest documents into the index, REUSING this pipeline's already-warm components.

        The API's /upload endpoint calls this so each upload reuses the loaded embedder, store, and
        sparse model instead of rebuilding them (which would reload the SPLADE/BM42 model every time).

        Parameters
        ----------
        paths : list[str]
            File paths (or globs) to extract → chunk → embed → store.

        Returns
        -------
        int
            Number of chunks stored.
        """
        from .ingest import ingest_paths  # local import to avoid a circular import at module load
        return ingest_paths(self.cfg, paths, embedder=self.embedder, store=self.store,
                            sparse_embedder=self.sparse_embedder)

    # ---- request-shaping helpers (prompt caching + conversational memory) --------------------
    def _system_param(self):
        """Return the `system` argument, marked for PROMPT CACHING when enabled.

        With caching on, we send `system` as a list of content blocks and tag the (large, static)
        prompt with `cache_control: ephemeral`. Anthropic then caches those tokens; subsequent
        calls in the same loop/turn re-read them at ~10% of the input price and lower latency.
        With caching off, we send the plain string.
        """
        if self.cfg.generator.use_prompt_caching:
            return [{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}]
        return SYSTEM_PROMPT

    def _tools_param(self):
        """Return the `tools` argument, with a cache breakpoint on the last tool when enabled.

        Tool definitions are static too; caching them (cache_control on the final tool) avoids
        re-billing the full schema on every step of the agent loop.
        """
        if self.cfg.generator.use_prompt_caching:
            tool = dict(SEARCH_TOOL)
            tool["cache_control"] = {"type": "ephemeral"}
            return [tool]
        return [SEARCH_TOOL]

    def _initial_messages(self, question: str, session_id: str | None) -> list[dict]:
        """Build the starting message list: prior session turns (if any) + the new question."""
        messages: list[dict] = []
        if session_id and self.sessions is not None:
            # Replay the clean transcript so follow-ups ("and the year before?") have context.
            messages.extend(self.sessions.history(session_id))
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _citations_from(ledger: dict) -> list[dict]:
        """Turn the provenance ledger into a citation list, sorted by score (best first)."""
        return [{"chunk_id": h.id, "source": h.source, "page": h.page, "score": round(h.score, 4)}
                for h in sorted(ledger.values(), key=lambda h: h.score, reverse=True)]

    def _run_tool_blocks(self, resp, ledger: dict) -> tuple[list, int]:
        """Service every tool_use block in a model response: search, update ledger, build results.

        Returns (tool_result_blocks, n_searches). Shared by answer() and answer_stream().
        """
        results, n = [], 0
        for block in resp.content:
            if block.type != "tool_use":
                continue  # skip text blocks the model may include alongside the tool call
            hits = self.retriever.search(**block.input)  # block.input == {"query": ...}
            n += 1
            for h in hits:  # merge into ledger, keeping the highest score per chunk id
                prev = ledger.get(h.id)
                if prev is None or h.score > prev.score:
                    ledger[h.id] = h
            # The API requires a tool_result referencing the tool_use id, with string content.
            results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps([
                    {"chunk_id": h.id, "source": h.source, "page": h.page, "text": h.text}
                    for h in hits]),
            })
        return results, n

    # ------------------------------------------------------------------------------------------
    def answer(self, question: str, session_id: str | None = None) -> AgentAnswer:
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
        session_id : str | None
            When given (and memory enabled), prior turns for this session are prepended so
            follow-up questions resolve in context, and this exchange is saved on completion.

        Returns
        -------
        AgentAnswer
            text + citations + steps + usage.
        """
        # Running conversation: prior session turns (if any) + this question.
        messages = self._initial_messages(question, session_id)
        # The provenance ledger: chunk id -> the best (highest-scoring) Hit seen for that chunk.
        ledger: dict[int, Hit] = {}
        # Counters: number of search calls, and cumulative input/output tokens.
        steps = in_tok = out_tok = 0

        # Loop until the model answers or we hit the safety cap.
        while steps <= self.cfg.agent.max_steps:
            # --- Ask the model. System prompt + tools are cache-tagged when caching is enabled.
            resp = self._llm.messages.create(
                model=self.cfg.generator.model,
                max_tokens=self.cfg.generator.max_tokens,
                temperature=self.cfg.generator.temperature,
                system=self._system_param(),
                tools=self._tools_param(),
                messages=messages,
            )
            # Accumulate token usage for cost reporting.
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens

            # --- CASE A: the model did NOT ask for a tool → it produced the final answer.
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                # Persist the clean (question, answer) exchange to conversational memory.
                if session_id and self.sessions is not None:
                    self.sessions.append(session_id, question, text)
                return AgentAnswer(text=text, citations=self._citations_from(ledger), steps=steps,
                                   usage={"input_tokens": in_tok, "output_tokens": out_tok})

            # --- CASE B: the model asked to search. Echo its turn, run the tools, feed results back.
            messages.append({"role": "assistant", "content": resp.content})
            results, n = self._run_tool_blocks(resp, ledger)
            steps += n
            messages.append({"role": "user", "content": results})

        # --- SAFETY STOP: we exhausted max_steps without the model committing to an answer.
        return AgentAnswer(text="Stopped after max_steps without a final answer.",
                           citations=self._citations_from(ledger), steps=steps,
                           usage={"input_tokens": in_tok, "output_tokens": out_tok})

    # ------------------------------------------------------------------------------------------
    def answer_stream(self, question: str, session_id: str | None = None):
        """Generator version of answer() that STREAMS the final answer token-by-token.

        Yields dict EVENTS so a web layer (see api.py /ask/stream) can forward them over SSE:
          {"type": "step",     "query": "...", "n_hits": 5}   # each retrieval the agent runs
          {"type": "token",    "text": "..."}                 # incremental answer text
          {"type": "final",    "answer": "...", "citations": [...], "steps": N, "usage": {...}}

        WHY STREAM: the tool-using steps still happen up front (you can't stream a decision to
        search), but once the model starts writing the ANSWER we stream those tokens so the UI
        shows text immediately instead of waiting for the whole response — a big UX win.
        """
        messages = self._initial_messages(question, session_id)
        ledger: dict[int, Hit] = {}
        steps = in_tok = out_tok = 0

        while steps <= self.cfg.agent.max_steps:
            # First, a NON-streamed call to discover whether the model wants to search or answer.
            resp = self._llm.messages.create(
                model=self.cfg.generator.model,
                max_tokens=self.cfg.generator.max_tokens,
                temperature=self.cfg.generator.temperature,
                system=self._system_param(),
                tools=self._tools_param(),
                messages=messages,
            )
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens

            if resp.stop_reason == "tool_use":
                # Announce each search as a step event, then feed results back and continue.
                messages.append({"role": "assistant", "content": resp.content})
                for block in resp.content:
                    if block.type == "tool_use":
                        yield {"type": "step", "query": block.input.get("query")}
                results, n = self._run_tool_blocks(resp, ledger)
                steps += n
                messages.append({"role": "user", "content": results})
                continue

            # FINAL turn: re-issue WITHOUT tools and STREAM the answer text token-by-token.
            text_parts = []
            with self._llm.messages.stream(
                model=self.cfg.generator.model,
                max_tokens=self.cfg.generator.max_tokens,
                temperature=self.cfg.generator.temperature,
                system=self._system_param(),
                messages=messages,
            ) as stream:
                for delta in stream.text_stream:   # yields the incremental answer text
                    text_parts.append(delta)
                    yield {"type": "token", "text": delta}
                final = stream.get_final_message()
                in_tok += final.usage.input_tokens
                out_tok += final.usage.output_tokens

            answer_text = "".join(text_parts)
            if session_id and self.sessions is not None:
                self.sessions.append(session_id, question, answer_text)
            yield {"type": "final", "answer": answer_text,
                   "citations": self._citations_from(ledger), "steps": steps,
                   "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
            return

        # Safety stop as a final event.
        yield {"type": "final", "answer": "Stopped after max_steps without a final answer.",
               "citations": self._citations_from(ledger), "steps": steps,
               "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
