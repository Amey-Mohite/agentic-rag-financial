"""
config.py — typed application config loaded from config.yaml (with ${ENV} expansion).

One config object drives the whole system: which vector store, which chunking strategy, retrieval
toggles, and agent limits. Swap backends or tune retrieval by editing config.yaml, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import os

import yaml


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-large"
    dims: int = 3072


@dataclass
class GeneratorConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class VectorStoreConfig:
    backend: Literal["pgvector", "qdrant"] = "qdrant"
    # pgvector
    dsn: Optional[str] = None
    # qdrant
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection: str = "filings"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64


@dataclass
class ChunkingConfig:
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_tokens: int = 500
    overlap_tokens: int = 50


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_pool: int = 20
    use_hybrid: bool = True
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rrf_k: int = 60


@dataclass
class AgentConfig:
    max_steps: int = 6


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @staticmethod
    def from_yaml(path: str) -> "AppConfig":
        """Load config from YAML, expanding ${ENV_VAR} placeholders. RETURNS AppConfig."""
        with open(path) as f:
            raw = _expand_env(yaml.safe_load(f) or {})
        return AppConfig(
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            generator=GeneratorConfig(**raw.get("generator", {})),
            vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
            chunking=ChunkingConfig(**raw.get("chunking", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            agent=AgentConfig(**raw.get("agent", {})),
        )

    def validate(self) -> None:
        """Fail fast on misconfig (missing connection string, dim mismatch)."""
        vs = self.vector_store
        if vs.backend == "pgvector" and not vs.dsn:
            raise ValueError("pgvector backend requires vector_store.dsn")
        if vs.backend == "qdrant" and not vs.url:
            raise ValueError("qdrant backend requires vector_store.url")


def _expand_env(obj):
    """Recursively replace '${VAR}' strings with os.environ values."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    return obj
