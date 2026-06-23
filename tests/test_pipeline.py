"""
Unit tests for the pure logic (no external services). Run with: pytest

These cover the parts worth testing in isolation: RRF fusion math, chunking behavior, config
loading/validation, and the agent loop control flow (with a stubbed Anthropic client + retriever).
"""

import os
import sys
import types

import numpy as np
import pytest


# ----------------------------------------------------------------------------------------------
# tiktoken downloads its vocab on first use; in offline CI we stub it with a whitespace tokenizer so
# chunking logic is still exercised. Remove this fixture where the vocab is available.
@pytest.fixture(autouse=True)
def _stub_tiktoken(monkeypatch):
    fake = types.ModuleType("tiktoken")

    class _Enc:
        def encode(self, s): return s.split()
        def decode(self, toks): return " ".join(toks)

    fake.get_encoding = lambda name: _Enc()
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    yield


# ----------------------------------------------------------------------------------------------
def test_rrf_fuses_and_dedupes():
    from agentic_rag.stores import Hit
    from agentic_rag.retrieval import reciprocal_rank_fusion
    dense = [Hit(1, "a", "s", 1, 0.9), Hit(2, "b", "s", 1, 0.8), Hit(3, "c", "s", 1, 0.7)]
    sparse = [Hit(3, "c", "s", 1, 5.0), Hit(1, "a", "s", 1, 4.0), Hit(9, "z", "s", 1, 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    ids = [h.id for h in fused]
    assert set(ids) == {1, 2, 3, 9}              # de-duplicated union
    assert ids[0] in (1, 3) and ids[1] in (1, 3)  # shared hits rank top
    # id 1: rank 0 in dense + rank 1 in sparse
    assert fused[0].score == pytest.approx(1 / 60 + 1 / 61) or \
           fused[1].score == pytest.approx(1 / 60 + 1 / 61)


def test_chunking_strategies_produce_provenance():
    from agentic_rag.chunking import chunk_document
    from agentic_rag.config import ChunkingConfig
    text = ("Revenue grew in 2023. New markets opened.\n\nRisk Factors\n"
            "We depend on one supplier. FX moves affect results. " * 10)
    for strat in ["fixed", "recursive", "semantic"]:
        cfg = ChunkingConfig(strategy=strat, chunk_tokens=40, overlap_tokens=5)
        chunks = chunk_document(text, "doc.pdf", cfg, page=3)
        assert len(chunks) > 0
        assert all(c.source == "doc.pdf" and c.page == 3 for c in chunks)


def test_config_loads_and_validates(tmp_path, monkeypatch):
    from agentic_rag.config import AppConfig
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    yaml_text = """
embedding: { model: text-embedding-3-large, dims: 3072 }
vector_store: { backend: qdrant, url: 'http://localhost:6333', api_key: '${QDRANT_API_KEY}' }
agent: { max_steps: 4 }
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = AppConfig.from_yaml(str(p))
    assert cfg.vector_store.api_key == "secret"   # ${ENV} expanded
    assert cfg.agent.max_steps == 4
    cfg.validate()                                 # should not raise


def test_config_validation_catches_missing_dsn():
    from agentic_rag.config import AppConfig, VectorStoreConfig
    cfg = AppConfig(vector_store=VectorStoreConfig(backend="pgvector", dsn=None))
    with pytest.raises(ValueError):
        cfg.validate()


def test_agent_loop_multi_hop_and_ledger(monkeypatch):
    """The agent loop should do multi-call retrieval, dedupe the ledger (keep max score), and sort."""
    from agentic_rag import agent as agent_mod
    from agentic_rag.stores import Hit

    # --- stub the retriever: two calls, overlapping ids with different scores ---
    calls = {"n": 0}

    class FakeRetriever:
        def search(self, query):
            calls["n"] += 1
            if calls["n"] == 1:
                return [Hit(1, "rev23", "A.pdf", 57, 5.0), Hit(2, "acq", "A.pdf", 12, 4.0)]
            return [Hit(1, "rev23", "A.pdf", 57, 9.0), Hit(3, "rev22", "B.pdf", 55, 6.0)]

    # --- stub Anthropic: tool_use twice, then final answer ---
    class Usage:
        input_tokens = 100; output_tokens = 20

    class Block:
        def __init__(self, **kw): self.__dict__.update(kw)

    class Resp:
        def __init__(self, content, stop): self.content = content; self.stop_reason = stop; self.usage = Usage()

    class FakeMessages:
        def __init__(self): self.turn = 0
        def create(self, **kw):
            self.turn += 1
            if self.turn == 1:
                return Resp([Block(type="tool_use", name="search_documents", id="t1",
                                   input={"query": "FY2023 revenue"})], "tool_use")
            if self.turn == 2:
                return Resp([Block(type="tool_use", name="search_documents", id="t2",
                                   input={"query": "FY2022 revenue"})], "tool_use")
            return Resp([Block(type="text", text="Revenue rose (A.pdf p57; B.pdf p55).")], "end_turn")

    class FakeAnthropic:
        def __init__(self): self.messages = FakeMessages()

    # build an AgenticRAG without running __init__ (avoids real clients), inject stubs
    rag = agent_mod.AgenticRAG.__new__(agent_mod.AgenticRAG)
    from agentic_rag.config import AppConfig
    rag.cfg = AppConfig()
    rag.retriever = FakeRetriever()
    rag._llm = FakeAnthropic()

    out = rag.answer("How did revenue change?")
    assert out.steps == 2
    assert out.usage["input_tokens"] == 300                  # 3 model calls x 100
    assert {c["chunk_id"] for c in out.citations} == {1, 2, 3}
    score1 = next(c["score"] for c in out.citations if c["chunk_id"] == 1)
    assert score1 == 9.0                                      # dedup kept the higher score
    assert out.citations[0]["score"] == 9.0                  # sorted desc
