"""
config.py — typed application config loaded from config.yaml (with ${ENV} expansion).

WHAT THIS MODULE IS
-------------------
A single, strongly-typed configuration object (`AppConfig`) drives the entire system:
which vector store to use, which chunking strategy to apply, which retrieval features
are turned on, and how many steps the agent may take. The guiding principle is:
"tune behavior by editing config.yaml, never by editing code."

WHY DATACLASSES
---------------
Each config section is a `@dataclass`. Dataclasses give us (a) named fields with type
hints, (b) sensible defaults, and (c) free `__init__`/`__repr__`. Because every field has
a default, you can build a fully-valid config with zero arguments — handy in tests.

WHY ${ENV} EXPANSION
--------------------
Secrets (API keys, database URLs) must NEVER be committed to YAML. Instead config.yaml
holds placeholders like `${QDRANT_API_KEY}` and we substitute the real value from the
process environment at load time. See `_expand_env` at the bottom of this file.
"""

# `from __future__ import annotations` makes ALL type annotations lazy (stored as strings).
# This lets us write `-> "AppConfig"` style forward references and use `list[Chunk]` syntax
# even on older Python versions, with no runtime cost.
from __future__ import annotations

# `dataclass` is the decorator that turns a plain class into a typed record.
# `field` lets us give a *mutable* default (here, a factory function) safely — you cannot
# write `x: Foo = Foo()` as a dataclass default because that single instance would be
# shared across every AppConfig; `field(default_factory=Foo)` builds a fresh one each time.
from dataclasses import dataclass, field

# `Literal` restricts a string field to an exact set of allowed values (e.g. only
# "qdrant" or "pgvector"). `Optional[X]` is shorthand for "X or None".
from typing import Literal, Optional

# `os` is used to read environment variables (os.environ).
import os

# `load_dotenv` reads a local .env file and pushes its KEY=VALUE pairs into os.environ.
# We call it at import time so that any module importing config gets env vars populated.
from dotenv import load_dotenv
load_dotenv()  # side effect: populate os.environ from the .env file if one exists

# PyYAML parser used to turn config.yaml text into nested Python dicts/lists.
import yaml


@dataclass
class EmbeddingConfig:
    """Settings for the text-embedding model that turns text into vectors.

    Fields
    ------
    model : str
        The embedding model name. Must match what was used at INGEST time — query and
        document vectors have to come from the same model or similarity is meaningless.
    dims : int
        Output dimensionality of each embedding vector. 3072 for text-embedding-3-large.
        This MUST match the vector size declared in the vector store's schema.
    """
    model: str = "text-embedding-3-large"  # OpenAI's largest embedding model
    dims: int = 3072                        # vector length produced by that model


@dataclass
class GeneratorConfig:
    """Settings for the LLM that writes the final answer (the 'generator' in RAG)."""
    model: str = "claude-sonnet-4-6"  # the chat model that reasons + answers
    max_tokens: int = 1024            # cap on the model's output length per call
    temperature: float = 0.0          # 0.0 = deterministic; we want factual, repeatable answers
    # PROMPT CACHING (cost optimization). When True, the agent marks the system prompt + tool
    # definitions with cache_control so Anthropic caches those tokens. In a multi-step agent loop
    # (and across conversation turns) the same large preamble is re-sent every call; caching it
    # makes repeat reads ~10% of the price and lower latency. See agent.py.
    use_prompt_caching: bool = True


@dataclass
class VectorStoreConfig:
    """Which vector database to use and how to connect to it.

    Two backends are supported and chosen by the `backend` field. Only the fields for the
    selected backend need to be filled in (validate() enforces this).
    """
    # `Literal[...]` means backend can ONLY be one of these two exact strings.
    backend: Literal["pgvector", "qdrant"] = "qdrant"

    # --- pgvector (Postgres) settings ---
    # DSN = the Postgres connection string, e.g. postgresql://user:pass@host:5432/ragdb.
    dsn: Optional[str] = None

    # --- qdrant settings ---
    # These default to environment variables so they can be set without touching YAML.
    # NOTE: defaults are evaluated ONCE at class-definition time, so os.environ is read at
    # import. config.yaml values (loaded later) override these via from_yaml().
    url: Optional[str] = os.environ.get("QDRANT_URL")       # e.g. http://localhost:6333
    api_key: Optional[str] = os.environ.get("QDRANT_API_KEY")  # blank for local Qdrant

    collection: str = "filings"      # name of the table/collection holding the vectors
    hnsw_m: int = 16                 # HNSW graph degree — higher = better recall, more memory
    hnsw_ef_construction: int = 64   # HNSW build-time search width — higher = better index, slower build


@dataclass
class ChunkingConfig:
    """How documents get split into retrievable pieces. See chunking.py for the algorithms."""
    # "fixed" = blunt every-N-tokens; "recursive" = paragraph/sentence aware (default);
    # "semantic" = structural topic-shift approximation.
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_tokens: int = 500    # target maximum size of each chunk, measured in tokens
    overlap_tokens: int = 50   # how many tokens consecutive chunks share (preserves context across boundaries)


@dataclass
class RetrievalConfig:
    """Knobs for the retrieval stage (see retrieval.py)."""
    top_k: int = 5                 # how many chunks to finally hand to the LLM
    candidate_pool: int = 20       # how many to pull from the store BEFORE reranking/trimming
    use_hybrid: bool = True        # combine dense (vector) + sparse search?
    use_reranker: bool = True      # apply a cross-encoder to re-score the candidate pool?
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # the cross-encoder model id
    rrf_k: int = 60                # constant in Reciprocal Rank Fusion (60 is the standard value)
    # SPARSE BACKEND — how the "sparse" half of hybrid retrieval is produced:
    #   "keyword" : the original simple full-text/keyword fallback (no extra deps).
    #   "splade"  : a learned sparse vector (SPLADE / BM42) via fastembed, fused INSIDE Qdrant
    #               using query_points prefetch + server-side RRF. This is "true" hybrid search.
    sparse_backend: Literal["keyword", "splade"] = "splade"
    # The fastembed sparse model id used when sparse_backend == "splade". BM42 is fast and strong
    # for short-to-medium passages; "prithivida/Splade_PP_en_v1" is the classic SPLADE++ option.
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"


@dataclass
class AgentConfig:
    """Limits on the agentic loop (see agent.py)."""
    max_steps: int = 6  # max number of tool/search calls before we force-stop (prevents runaway loops/cost)


@dataclass
class MemoryConfig:
    """Conversational memory settings (see memory.py).

    When enabled, the API/agent keep a short rolling history of prior user/assistant turns keyed by
    a session id, so follow-up questions like "and the year before?" resolve against earlier context
    instead of being treated as standalone queries.
    """
    enabled: bool = True       # turn conversational memory on/off
    max_turns: int = 6         # how many of the most recent (user,assistant) turns to retain
    backend: Literal["memory"] = "memory"  # "memory" = in-process dict; swap for Redis in prod


@dataclass
class IngestionConfig:
    """Document-ingestion settings (see ingest.py)."""
    # TABLE-AWARE EXTRACTION. Financial filings are full of numeric tables that naive text
    # extraction flattens into unreadable "soup", destroying row/column relationships. When True,
    # PDFs are parsed with pdfplumber and tables are rendered as Markdown pipe-tables so the
    # structure (and therefore the numbers) survives into the embeddings. Falls back to plain
    # text extraction if pdfplumber isn't installed.
    extract_tables: bool = True


@dataclass
class AppConfig:
    """The root config object — one of each sub-config, assembled together.

    Use `AppConfig()` for all-defaults (tests), or `AppConfig.from_yaml(path)` to load from
    config.yaml. Always call `.validate()` before using it in a real pipeline.
    """
    # `field(default_factory=...)` ensures each AppConfig gets its OWN fresh sub-config
    # instance rather than sharing one mutable object across instances.
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)

    # RUNTIME API KEYS (optional). Normally the OpenAI/Anthropic SDKs read keys from the
    # environment. But the "bring your own keys" web UI lets a tester paste keys at runtime; when
    # set here they're passed EXPLICITLY to the clients (so multiple sessions with different keys
    # don't clash via global env). Left None → the SDKs fall back to the environment.
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    @staticmethod
    def from_yaml(path: str) -> "AppConfig":
        """Load configuration from a YAML file, expanding ${ENV_VAR} placeholders.

        Parameters
        ----------
        path : str
            Filesystem path to config.yaml.

        Returns
        -------
        AppConfig
            A fully-populated config object. Any section missing from the YAML falls back
            to that sub-config's dataclass defaults.

        How it works
        ------------
        1. Read + parse the YAML into a nested dict (`yaml.safe_load`).
        2. Recursively replace any "${VAR}" string with its environment value (`_expand_env`).
        3. Splat each top-level section dict into the matching dataclass via `**`.
        """
        # Open the YAML file and parse it. `yaml.safe_load` returns None for an empty file,
        # so `or {}` guarantees we always have a dict to work with.
        with open(path) as f:
            raw = _expand_env(yaml.safe_load(f) or {})

        # Build each sub-config by unpacking its section dict as keyword arguments.
        # `raw.get("embedding", {})` returns {} if that section is absent → all defaults.
        return AppConfig(
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            generator=GeneratorConfig(**raw.get("generator", {})),
            vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
            chunking=ChunkingConfig(**raw.get("chunking", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            agent=AgentConfig(**raw.get("agent", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            ingestion=IngestionConfig(**raw.get("ingestion", {})),
        )

    def validate(self) -> None:
        """Fail fast on misconfiguration BEFORE doing any expensive work.

        Raising here (at startup) is far better than a cryptic connection error deep inside
        a retrieval call. Currently checks that the selected backend has its required
        connection field set.

        Raises
        ------
        ValueError
            If the chosen backend is missing its connection setting.
        """
        vs = self.vector_store  # local alias for readability
        # pgvector needs a Postgres DSN to connect.
        if vs.backend == "pgvector" and not vs.dsn:
            raise ValueError("pgvector backend requires vector_store.dsn")
        # qdrant needs a URL to reach the Qdrant server.
        if vs.backend == "qdrant" and not vs.url:
            raise ValueError("qdrant backend requires vector_store.url")


def _expand_env(obj):
    """Recursively replace any '${VAR}' string in a parsed-YAML structure with os.environ['VAR'].

    The YAML parser hands us a tree of dicts, lists, and scalars. We walk that tree and,
    wherever we find a string shaped exactly like "${SOMETHING}", swap in the environment
    variable's value (or "" if it isn't set). Everything else is returned unchanged.

    Parameters
    ----------
    obj : Any
        A node from the parsed YAML tree (dict, list, str, int, etc.).

    Returns
    -------
    Any
        The same structure with all ${VAR} placeholders resolved.
    """
    # Case 1: a dict → recurse into every value, keep the keys.
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    # Case 2: a list → recurse into every element.
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    # Case 3: a string that looks exactly like "${VAR}" → look up VAR in the environment.
    # obj[2:-1] strips the leading "${" and trailing "}" to get the bare variable name.
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    # Case 4: anything else (plain string, number, bool, None) → return as-is.
    return obj
