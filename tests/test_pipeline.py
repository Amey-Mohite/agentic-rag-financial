"""
Unit tests for the PURE logic (no external services). Run with: pytest

DESIGN: TEST WHAT'S DETERMINISTIC, STUB WHAT'S NOT
--------------------------------------------------
We deliberately test only the parts that don't need a network or a database: the RRF fusion
math, the chunking behavior, config loading/validation, and the agent's loop control flow. The
two non-deterministic dependencies — the Anthropic client and the retriever — are replaced with
hand-written STUBS so the agent loop can be exercised offline and its bookkeeping asserted.
"""

import os
import sys
import types  # used to fabricate a fake `tiktoken` module

import numpy as np
import pytest


# ----------------------------------------------------------------------------------------------
# tiktoken downloads its vocabulary on first use. In offline CI that download would fail, so we
# stub tiktoken with a trivial whitespace tokenizer. This still exercises the chunking LOGIC
# (counting/packing "tokens") without the real vocab. Delete this fixture where vocab is available.
@pytest.fixture(autouse=True)  # autouse=True → applied automatically to every test in this file
def _stub_tiktoken(monkeypatch):
    fake = types.ModuleType("tiktoken")  # an empty module object we'll attach attributes to

    class _Enc:
        # "encode" = split on whitespace into word-tokens; "decode" = join back with spaces.
        def encode(self, s): return s.split()
        def decode(self, toks): return " ".join(toks)

    # get_encoding(name) ignores the name and returns our fake encoder.
    fake.get_encoding = lambda name: _Enc()
    # Insert the fake into sys.modules so any `import tiktoken` picks it up for this test.
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    yield  # the test runs here; monkeypatch auto-undoes the patch afterward


# ----------------------------------------------------------------------------------------------
def test_rrf_fuses_and_dedupes():
    """RRF should produce the de-duplicated UNION of inputs, with shared hits ranked highest."""
    from agentic_rag.stores import Hit
    from agentic_rag.retrieval import reciprocal_rank_fusion
    # Two ranked lists. ids 1 and 3 appear in BOTH (so they should win); 2 and 9 appear once.
    dense = [Hit(1, "a", "s", 1, 0.9), Hit(2, "b", "s", 1, 0.8), Hit(3, "c", "s", 1, 0.7)]
    sparse = [Hit(3, "c", "s", 1, 5.0), Hit(1, "a", "s", 1, 4.0), Hit(9, "z", "s", 1, 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    ids = [h.id for h in fused]
    assert set(ids) == {1, 2, 3, 9}               # de-duplicated union of all ids
    assert ids[0] in (1, 3) and ids[1] in (1, 3)  # the two shared hits rank at the top
    # id 1 = rank 0 in dense (1/60) + rank 1 in sparse (1/61). Check one of the top two matches.
    assert fused[0].score == pytest.approx(1 / 60 + 1 / 61) or \
           fused[1].score == pytest.approx(1 / 60 + 1 / 61)


def test_chunking_strategies_produce_provenance():
    """Every strategy must produce non-empty chunks that all carry the source + page provenance."""
    from agentic_rag.chunking import chunk_document
    from agentic_rag.config import ChunkingConfig
    # Repeat the text x10 so it's big enough to force multiple chunks at a 40-token budget.
    text = ("Revenue grew in 2023. New markets opened.\n\nRisk Factors\n"
            "We depend on one supplier. FX moves affect results. " * 10)
    for strat in ["fixed", "recursive", "semantic"]:
        cfg = ChunkingConfig(strategy=strat, chunk_tokens=40, overlap_tokens=5)
        chunks = chunk_document(text, "doc.pdf", cfg, page=3)
        assert len(chunks) > 0                                          # produced something
        assert all(c.source == "doc.pdf" and c.page == 3 for c in chunks)  # provenance preserved


def test_config_loads_and_validates(tmp_path, monkeypatch):
    """from_yaml should parse YAML, expand ${ENV}, and validate() should pass for a good config."""
    from agentic_rag.config import AppConfig
    monkeypatch.setenv("QDRANT_API_KEY", "secret")  # set the env var the YAML references
    yaml_text = """
embedding: { model: text-embedding-3-large, dims: 3072 }
vector_store: { backend: qdrant, url: 'http://localhost:6333', api_key: '${QDRANT_API_KEY}' }
agent: { max_steps: 4 }
"""
    p = tmp_path / "c.yaml"      # tmp_path is a pytest-provided temp directory
    p.write_text(yaml_text)
    cfg = AppConfig.from_yaml(str(p))
    assert cfg.vector_store.api_key == "secret"  # ${QDRANT_API_KEY} was expanded from the env
    assert cfg.agent.max_steps == 4              # YAML value overrode the default
    cfg.validate()                               # a valid qdrant config should NOT raise


def test_config_validation_catches_missing_dsn():
    """validate() must RAISE when a backend is missing its required connection field."""
    from agentic_rag.config import AppConfig, VectorStoreConfig
    cfg = AppConfig(vector_store=VectorStoreConfig(backend="pgvector", dsn=None))
    with pytest.raises(ValueError):  # pgvector with no dsn → ValueError expected
        cfg.validate()


def test_agent_loop_multi_hop_and_ledger(monkeypatch):
    """The agent loop should do multi-call retrieval, dedupe the ledger (keep max score), and sort.

    This is the most important test: it verifies the CONTROL FLOW of answer() using stubs for
    both the retriever and the Anthropic client, so no network/keys are needed.
    """
    from agentic_rag import agent as agent_mod
    from agentic_rag.stores import Hit

    # --- Stub the retriever: two searches, with OVERLAPPING ids at DIFFERENT scores ---
    calls = {"n": 0}  # mutable counter shared across calls (dict so the closure can mutate it)

    class FakeRetriever:
        def search(self, query):
            calls["n"] += 1
            if calls["n"] == 1:
                # First search: chunk 1 @5.0, chunk 2 @4.0.
                return [Hit(1, "rev23", "A.pdf", 57, 5.0), Hit(2, "acq", "A.pdf", 12, 4.0)]
            # Second search: chunk 1 again @9.0 (higher!), plus chunk 3 @6.0.
            return [Hit(1, "rev23", "A.pdf", 57, 9.0), Hit(3, "rev22", "B.pdf", 55, 6.0)]

    # --- Stub the Anthropic client: respond tool_use twice, then a final text answer ---
    class Usage:
        input_tokens = 100; output_tokens = 20  # fixed usage per call so totals are predictable

    class Block:
        # A flexible stand-in for a content block; we just set whatever attrs we pass in.
        def __init__(self, **kw): self.__dict__.update(kw)

    class Resp:
        # Mimics the API response object: .content, .stop_reason, .usage.
        def __init__(self, content, stop): self.content = content; self.stop_reason = stop; self.usage = Usage()

    class FakeMessages:
        def __init__(self): self.turn = 0
        def create(self, **kw):
            self.turn += 1
            if self.turn == 1:  # turn 1: ask to search FY2023 revenue
                return Resp([Block(type="tool_use", name="search_documents", id="t1",
                                   input={"query": "FY2023 revenue"})], "tool_use")
            if self.turn == 2:  # turn 2: ask to search FY2022 revenue
                return Resp([Block(type="tool_use", name="search_documents", id="t2",
                                   input={"query": "FY2022 revenue"})], "tool_use")
            # turn 3: final answer (stop_reason != "tool_use" ends the loop).
            return Resp([Block(type="text", text="Revenue rose (A.pdf p57; B.pdf p55).")], "end_turn")

    class FakeAnthropic:
        def __init__(self): self.messages = FakeMessages()

    # Build an AgenticRAG WITHOUT running __init__ (which would create real clients). `__new__`
    # makes a bare instance; we then inject our stubs onto it directly.
    rag = agent_mod.AgenticRAG.__new__(agent_mod.AgenticRAG)
    from agentic_rag.config import AppConfig
    rag.cfg = AppConfig()          # default config (max_steps=6, enough for 2 hops)
    rag.retriever = FakeRetriever()
    rag._llm = FakeAnthropic()

    out = rag.answer("How did revenue change?")
    assert out.steps == 2                          # two tool calls were made
    assert out.usage["input_tokens"] == 300        # 3 model calls x 100 input tokens each
    assert {c["chunk_id"] for c in out.citations} == {1, 2, 3}  # union of all chunks seen
    # Chunk 1 was seen at 5.0 then 9.0 — the ledger must keep the HIGHER score.
    score1 = next(c["score"] for c in out.citations if c["chunk_id"] == 1)
    assert score1 == 9.0
    assert out.citations[0]["score"] == 9.0        # citations are sorted by score descending


# ============================================================================================
# Tests for the production upgrades: memory, table extraction, prompt caching, hybrid routing.
# ============================================================================================
def test_memory_store_roundtrip_and_bound():
    """InMemorySessionStore should store turns per session, isolate sessions, and bound length."""
    from agentic_rag.config import MemoryConfig
    from agentic_rag.memory import InMemorySessionStore
    store = InMemorySessionStore(MemoryConfig(max_turns=2))  # keep last 2 exchanges
    store.append("s1", "q1", "a1")
    store.append("s1", "q2", "a2")
    store.append("s1", "q3", "a3")                 # this should evict the oldest (q1/a1)
    hist = store.history("s1")
    assert [m["content"] for m in hist] == ["q2", "a2", "q3", "a3"]  # bounded to 2 exchanges
    assert store.history("s2") == []               # sessions are isolated
    store.reset("s1")
    assert store.history("s1") == []               # reset clears history


def test_table_to_markdown_preserves_structure():
    """_table_to_markdown should render a 2-D table as an aligned Markdown pipe-table."""
    from agentic_rag.ingest import _table_to_markdown
    md = _table_to_markdown([["Metric", "FY25", "FY24"], ["Revenue", "294,866", "298,085"]])
    lines = md.splitlines()
    assert lines[0] == "| Metric | FY25 | FY24 |"  # header row
    assert set(lines[1].replace(" ", "")) <= set("|-")  # separator row is just | and -
    assert "| Revenue | 294,866 | 298,085 |" in md  # numbers stay aligned to their columns


def test_prompt_caching_param_shaping():
    """_system_param/_tools_param add cache_control when enabled, and are plain when disabled."""
    from agentic_rag import agent as agent_mod
    from agentic_rag.config import AppConfig

    rag = agent_mod.AgenticRAG.__new__(agent_mod.AgenticRAG)  # bare instance, no real clients
    rag.cfg = AppConfig()
    rag.cfg.generator.use_prompt_caching = True
    sys_p, tools_p = rag._system_param(), rag._tools_param()
    assert isinstance(sys_p, list) and sys_p[0]["cache_control"] == {"type": "ephemeral"}
    assert tools_p[0]["cache_control"] == {"type": "ephemeral"}

    rag.cfg.generator.use_prompt_caching = False
    assert isinstance(rag._system_param(), str)        # plain string when caching off
    assert "cache_control" not in rag._tools_param()[0]


def test_agent_memory_replays_history(monkeypatch):
    """With a session_id, the agent should prepend prior turns and persist the new exchange."""
    from agentic_rag import agent as agent_mod
    from agentic_rag.config import AppConfig, MemoryConfig
    from agentic_rag.memory import InMemorySessionStore

    seen_messages = {}

    class Usage:
        input_tokens = 10; output_tokens = 2

    class Block:
        def __init__(self, **kw): self.__dict__.update(kw)

    class Resp:
        def __init__(self, content, stop): self.content = content; self.stop_reason = stop; self.usage = Usage()

    class FakeMessages:
        def create(self, **kw):
            seen_messages["messages"] = kw["messages"]   # capture what the agent sent
            return Resp([Block(type="text", text="It was $390B.")], "end_turn")

    class FakeAnthropic:
        def __init__(self): self.messages = FakeMessages()

    rag = agent_mod.AgenticRAG.__new__(agent_mod.AgenticRAG)
    rag.cfg = AppConfig()
    rag.sessions = InMemorySessionStore(MemoryConfig())
    rag._llm = FakeAnthropic()
    rag.sessions.append("sess", "What was FY25 revenue?", "It was $391B.")  # a prior turn exists

    out = rag.answer("And the year before?", session_id="sess")
    # The request must have replayed the prior 2 messages BEFORE the new question (3 total).
    assert len(seen_messages["messages"]) == 3
    assert seen_messages["messages"][0]["content"] == "What was FY25 revenue?"
    assert seen_messages["messages"][-1]["content"] == "And the year before?"
    # The new exchange is now persisted (prior 1 + new 1 = 2 exchanges = 4 messages).
    assert len(rag.sessions.history("sess")) == 4
    assert out.text == "It was $390B."


def test_retriever_prefers_native_hybrid():
    """When the store supports native hybrid AND a sparse embedder is present, use hybrid_search."""
    from agentic_rag.config import RetrievalConfig
    from agentic_rag.stores import Hit
    from agentic_rag.retrieval import Retriever

    calls = {"hybrid": 0, "dense": 0}

    class FakeStore:
        supports_native_hybrid = True
        def hybrid_search(self, dvec, svec, k):
            calls["hybrid"] += 1
            return [Hit(1, "x", "A", 1, 0.5)]
        def dense_search(self, qvec, k):
            calls["dense"] += 1
            return [Hit(2, "y", "B", 1, 0.4)]

    class FakeDense:
        def embed_query(self, q): return [0.0]
    class FakeSparse:
        def embed_query(self, q): return object()   # sparse vector stand-in

    cfg = RetrievalConfig(use_hybrid=True, use_reranker=False, top_k=1, candidate_pool=5)
    r = Retriever(FakeStore(), FakeDense(), cfg, sparse_embedder=FakeSparse())
    hits = r.search("revenue")
    assert calls["hybrid"] == 1 and calls["dense"] == 0   # took the native hybrid path
    assert hits[0].id == 1
