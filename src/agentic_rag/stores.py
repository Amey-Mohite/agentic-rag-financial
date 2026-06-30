"""
stores.py — two vector-database backends (pgvector & Qdrant) behind ONE common interface.

THE BIG IDEA: ONE INTERFACE, SWAPPABLE BACKENDS
-----------------------------------------------
The rest of the system never talks to Postgres or Qdrant directly. It talks to the
`VectorStore` "Protocol" (a structural interface) which declares four methods:
    setup()          — create the table/collection + indexes (idempotent)
    upsert()         — insert chunks + their vectors
    dense_search()   — semantic search by vector similarity
    sparse_search()  — keyword search (lexical / full-text)
Both `PgVectorStore` and `QdrantStore` implement those four methods. `make_store(cfg, dims)`
picks the right one based on config. Swapping databases is a one-line config change.

DENSE vs SPARSE (a core RAG concept)
------------------------------------
- DENSE search compares embedding vectors → finds chunks that MEAN the same thing even if
  they share no words ("revenue" ~ "net sales").
- SPARSE search matches actual words/tokens → great for exact terms, numbers, tickers,
  acronyms that embeddings sometimes blur.
Hybrid retrieval (see retrieval.py) fuses both. Provenance (source/page/text) is stored
ALONGSIDE every vector so any hit can be cited directly.
"""

from __future__ import annotations

# `dataclass` for the small Hit record; `Protocol` to declare the structural interface.
from dataclasses import dataclass
from typing import Protocol
import numpy as np

# Config + the Chunk type produced by chunking.py.
from .config import VectorStoreConfig
from .chunking import Chunk


@dataclass
class Hit:
    """One retrieval result: the chunk text, its provenance, and a relevance score.

    Fields
    ------
    id : int
        Stable identifier of the stored chunk (DB primary key / point id). Used to de-dupe.
    text : str
        The chunk text — what the LLM will read.
    source : str
        Originating document name (for citations).
    page : int
        Page number within that document (for citations).
    score : float
        Relevance score. Its SCALE depends on who produced it: cosine (~0..1) from dense
        search, ts_rank from sparse, a fused score from RRF, or a cross-encoder logit after
        reranking. Always treat it as "higher = better", not as an absolute probability.
    """
    id: int
    text: str
    source: str
    page: int
    score: float


class VectorStore(Protocol):
    """Structural interface every backend must satisfy.

    `Protocol` means: any class that HAS these four methods (with these signatures) counts
    as a VectorStore — no explicit subclassing required. The `...` bodies are just stubs;
    the real implementations live in the concrete classes below.
    `supports_native_hybrid` advertises whether the store can fuse dense + learned-sparse vectors
    server-side (Qdrant with SPLADE/BM42 enabled). The Retriever reads it to choose between a
    single native hybrid call and the older app-side dense+sparse+RRF path.
    """
    supports_native_hybrid: bool                                              # capability flag

    def setup(self) -> None: ...                                              # create schema/indexes
    # `sparse_vectors` is optional — only Qdrant-with-sparse uses it; other backends ignore it.
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, sparse_vectors=None) -> None: ...
    def dense_search(self, query_vec: np.ndarray, k: int) -> list[Hit]: ...   # vector similarity search
    def sparse_search(self, query_text: str, k: int) -> list[Hit]: ...        # keyword/full-text search


# ----------------------------------------------------------------------------------------------
class PgVectorStore:
    """Backend 1 — PostgreSQL + the `pgvector` extension.

    Uses `halfvec` (16-bit float vectors, half the storage) with an HNSW index for fast
    approximate dense search, and Postgres' native full-text search (`tsvector` + GIN index)
    for the sparse side. Everything lives in one table, one database — operationally simple.
    """

    supports_native_hybrid = False  # pgvector fuses dense+sparse app-side (RRF in retrieval.py)

    def __init__(self, cfg: VectorStoreConfig, dims: int):
        """Remember the connection config and the vector dimensionality (needed for the schema)."""
        self._cfg = cfg     # holds dsn, collection name, HNSW params
        self._dims = dims   # vector length — must match the embedder's output

    def setup(self) -> None:
        """Create the extension, table, and indexes. Idempotent (safe to run repeatedly).

        Idempotency comes from `IF NOT EXISTS` everywhere, so re-running on an existing DB
        is a no-op rather than an error.
        """
        import psycopg  # local import: only needed when actually using pgvector
        # `with` opens a connection AND a cursor, auto-closing both at block end.
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            # Enable the pgvector extension (provides the vector/halfvec types + operators).
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Create the table. Note `fts` is a GENERATED column: Postgres auto-derives a
            # full-text search vector from `text` on every insert/update — no app code needed.
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._cfg.collection} (
                    id SERIAL PRIMARY KEY, source TEXT, page INT, text TEXT,
                    embedding halfvec({self._dims}),
                    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)""")
            # HNSW index on the embedding for fast approximate-nearest-neighbor dense search.
            # `halfvec_cosine_ops` tells it to use cosine distance.
            cur.execute(f"""CREATE INDEX IF NOT EXISTS {self._cfg.collection}_hnsw
                ON {self._cfg.collection} USING hnsw (embedding halfvec_cosine_ops)
                WITH (m={self._cfg.hnsw_m}, ef_construction={self._cfg.hnsw_ef_construction})""")
            # GIN index on the full-text column makes sparse/keyword queries fast.
            cur.execute(f"""CREATE INDEX IF NOT EXISTS {self._cfg.collection}_fts
                ON {self._cfg.collection} USING GIN (fts)""")
            conn.commit()  # persist all the DDL above

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, sparse_vectors=None) -> None:
        """Insert each (chunk, vector) pair as a row. `vectors[i]` is the embedding of `chunks[i]`.

        `sparse_vectors` is accepted for interface compatibility but IGNORED here: pgvector's
        sparse side is the generated full-text column, not a learned sparse vector.
        """
        import psycopg
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            # Pair up chunks with their vectors and insert row by row.
            for ch, vec in zip(chunks, vectors):
                cur.execute(
                    f"INSERT INTO {self._cfg.collection} (source,page,text,embedding) VALUES (%s,%s,%s,%s)",
                    # `%s` placeholders are parameterized (prevents SQL injection). `vec.tolist()`
                    # converts the NumPy array to a plain Python list pgvector understands.
                    (ch.source, ch.page, ch.text, vec.tolist()))
            conn.commit()  # commit the whole batch at once

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        """Return the k chunks whose embeddings are nearest the query vector (cosine).

        The `<=>` operator is pgvector's cosine DISTANCE (0 = identical, 2 = opposite). We
        ORDER BY it ascending (closest first) and report `1 - distance` as the SCORE so that
        higher = more similar, matching the Hit convention.
        """
        import psycopg
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id,text,source,page,1-(embedding <=> %s::halfvec) FROM {self._cfg.collection} "
                f"ORDER BY embedding <=> %s::halfvec LIMIT %s",
                (query_vec.tolist(), query_vec.tolist(), k))
            # Each DB row is (id, text, source, page, score) — exactly Hit's field order, so
            # `Hit(*row)` splats the tuple straight into the dataclass.
            return [Hit(*row) for row in cur.fetchall()]

    def sparse_search(self, query_text: str, k: int) -> list[Hit]:
        """Return the k chunks that best match the query's WORDS via Postgres full-text search.

        `plainto_tsquery` parses the raw query into a tsquery; `@@` tests a match; and
        `ts_rank_cd` scores how well each row matches (used both to filter and to order).
        """
        import psycopg
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id,text,source,page,ts_rank_cd(fts,plainto_tsquery('english',%s)) "
                f"FROM {self._cfg.collection} WHERE fts @@ plainto_tsquery('english',%s) "
                f"ORDER BY ts_rank_cd(fts,plainto_tsquery('english',%s)) DESC LIMIT %s",
                (query_text, query_text, query_text, k))
            return [Hit(*row) for row in cur.fetchall()]


# ----------------------------------------------------------------------------------------------
class QdrantStore:
    """Backend 2 — Qdrant, a purpose-built vector database.

    Uses Qdrant's current `query_points` API with NAMED vectors. We always have a "dense" vector;
    when `enable_sparse` is set we ALSO store a named "sparse" vector (SPLADE/BM42) and fuse the
    two server-side via query_points prefetch + RRF — this is "true" hybrid search. Payload
    (text/source/page) is stored next to each point so hits are self-describing.
    """

    # Class-level default; the instance overrides it in __init__ based on enable_sparse.
    supports_native_hybrid = False

    def __init__(self, cfg: VectorStoreConfig, dims: int, enable_sparse: bool = False):
        """Remember connection config + vector dimensionality, and whether sparse hybrid is on.

        Parameters
        ----------
        cfg, dims : as before.
        enable_sparse : bool
            When True, the collection carries a named "sparse" vector and `hybrid_search` is
            available — the Retriever will prefer it over the app-side RRF path.
        """
        self._cfg = cfg
        self._dims = dims
        self._enable_sparse = enable_sparse
        # Advertise the capability on the instance so Retriever can branch on it.
        self.supports_native_hybrid = enable_sparse
        self._client_instance = None   # cached client (see _client)

    def _client(self):
        """Return a single, cached QdrantClient for this store.

        WHY CACHE: an in-memory Qdrant lives INSIDE the client object — a fresh client each call
        would be a brand-new empty database, so upsert and search would never see the same data.
        Caching one client makes in-memory mode work and is more efficient for remote mode too.

        MODES (chosen by the configured url):
          - ":memory:"  → an embedded, in-process Qdrant (zero setup; data is per-process, ephemeral).
                          Perfect for the "bring your own keys" demo: testers need no Qdrant account.
          - a URL       → a real Qdrant server / Qdrant Cloud (persistent, shared).
        """
        if self._client_instance is None:
            from qdrant_client import QdrantClient
            url = (self._cfg.url or "").strip()
            if url in (":memory:", "memory", ""):
                # Embedded in-process vector DB — no server, no key, nothing to deploy.
                self._client_instance = QdrantClient(location=":memory:")
            else:
                self._client_instance = QdrantClient(url=url, api_key=self._cfg.api_key or None)
        return self._client_instance

    def setup(self) -> None:
        """Create the collection if it doesn't already exist (idempotent).

        When sparse is enabled we additionally declare a named "sparse" vector. Sparse vectors
        live in their own config map (`sparse_vectors_config`) separate from dense vectors.
        """
        from qdrant_client import models
        client = self._client()
        # List existing collection names so we don't recreate (which would error / wipe).
        existing = [c.name for c in client.get_collections().collections]
        if self._cfg.collection not in existing:
            # Sparse vectors are only declared when enabled; otherwise pass None.
            sparse_cfg = ({"sparse": models.SparseVectorParams()}
                          if self._enable_sparse else None)
            client.create_collection(
                collection_name=self._cfg.collection,
                # Named DENSE vector of our dimensionality, scored by cosine.
                vectors_config={"dense": models.VectorParams(
                    size=self._dims, distance=models.Distance.COSINE)},
                # Named SPARSE vector (SPLADE/BM42). Qdrant scores sparse by dot product.
                sparse_vectors_config=sparse_cfg)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, sparse_vectors=None) -> None:
        """Write chunks + vectors as Qdrant points, with provenance stored in each payload.

        Parameters
        ----------
        chunks, vectors : the chunks and their DENSE embeddings (vectors[i] ↔ chunks[i]).
        sparse_vectors : list[SparseVector] | None
            Optional learned sparse embeddings (one per chunk). When provided AND sparse is
            enabled, each point also gets a named "sparse" vector for true hybrid search.
        """
        from qdrant_client import models
        client = self._client()
        points = []
        for i, (ch, vec) in enumerate(zip(chunks, vectors)):
            # Every point has the dense vector under the "dense" name.
            vector = {"dense": vec.tolist()}
            # If we computed sparse vectors, attach the matching one as the "sparse" named vector.
            if self._enable_sparse and sparse_vectors is not None:
                sv = sparse_vectors[i]
                vector["sparse"] = models.SparseVector(indices=sv.indices, values=sv.values)
            points.append(models.PointStruct(
                id=i, vector=vector,
                payload={"text": ch.text, "source": ch.source, "page": ch.page}))
        # `wait=True` blocks until the write is durably applied — important before searching.
        client.upsert(collection_name=self._cfg.collection, points=points, wait=True)

    def hybrid_search(self, dense_vec: np.ndarray, sparse_vec, k: int) -> list[Hit]:
        """TRUE hybrid search: fuse dense + sparse INSIDE Qdrant with prefetch + server-side RRF.

        How it works
        ------------
        `query_points` runs two PREFETCH sub-queries — one over the "dense" vector, one over the
        "sparse" vector — each pulling a candidate pool, then fuses them with Reciprocal Rank
        Fusion (`FusionQuery(fusion=RRF)`) on the server. One round-trip, no app-side fusion, and
        the sparse side is a learned SPLADE/BM42 vector (term importance), not a keyword filter.

        Parameters
        ----------
        dense_vec : np.ndarray
            The dense query embedding.
        sparse_vec : SparseVector
            The learned sparse query embedding (indices + values).
        k : int
            How many fused results to return.

        Returns
        -------
        list[Hit]
            Fused, ranked results (score is the RRF fused score from Qdrant).
        """
        from qdrant_client import models
        # Pull a wider candidate pool per arm than k so fusion has material to work with.
        prefetch_limit = max(k * 4, 20)
        res = self._client().query_points(
            collection_name=self._cfg.collection,
            prefetch=[
                # Arm 1: dense nearest neighbors.
                models.Prefetch(query=dense_vec.tolist(), using="dense", limit=prefetch_limit),
                # Arm 2: sparse (SPLADE/BM42) matches.
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                    using="sparse", limit=prefetch_limit),
            ],
            # Fuse the two prefetch arms with Reciprocal Rank Fusion on the server.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True)
        return [Hit(int(p.id), p.payload["text"], p.payload["source"], p.payload["page"], float(p.score))
                for p in res.points]

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        """Vector similarity search: return the k nearest points to the query vector."""
        res = self._client().query_points(
            collection_name=self._cfg.collection, query=query_vec.tolist(),
            using="dense", limit=k, with_payload=True)  # using="dense" selects our named vector
        # Re-pack Qdrant's scored points into our uniform Hit type. `p.score` is cosine sim.
        return [Hit(int(p.id), p.payload["text"], p.payload["source"], p.payload["page"], float(p.score))
                for p in res.points]

    def sparse_search(self, query_text: str, k: int) -> list[Hit]:
        """Keyword fallback over the payload `text` field (a simplified 'sparse' search).

        CONCEPT/CAVEAT: this is NOT a true sparse-vector search. It uses Qdrant's payload
        full-text filter (MatchText) to find points whose text contains the query terms, via
        `scroll` (which filters but does NOT rank). For real hybrid search you'd add a sparse
        vector (SPLADE/BM42) and fuse scores. Hence score is hard-coded 0.0 here — RRF in
        retrieval.py only uses RANK/position, so a flat score is acceptable.
        """
        from qdrant_client import models
        res = self._client().scroll(
            collection_name=self._cfg.collection,
            # Keep only points whose `text` payload matches the query terms.
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="text", match=models.MatchText(text=query_text))]),
            limit=k, with_payload=True)
        # `scroll` returns a (points, next_page_offset) tuple — res[0] is the points list.
        return [Hit(int(p.id), p.payload["text"], p.payload["source"], p.payload["page"], 0.0)
                for p in res[0]]


def make_store(cfg: VectorStoreConfig, dims: int, enable_sparse: bool = False) -> VectorStore:
    """Factory: return the concrete store implementation named in the config.

    Parameters
    ----------
    cfg : VectorStoreConfig
        Provides `backend` ("qdrant" | "pgvector") plus connection details.
    dims : int
        Vector dimensionality (forwarded to the store so it builds a matching schema).
    enable_sparse : bool
        Whether to enable native dense+sparse hybrid (only honored by the Qdrant backend; the
        caller sets this from RetrievalConfig.sparse_backend == "splade").

    Returns
    -------
    VectorStore
        A `QdrantStore` or `PgVectorStore` (both satisfy the VectorStore protocol).

    Raises
    ------
    ValueError
        On an unrecognized backend name.
    """
    if cfg.backend == "pgvector":
        return PgVectorStore(cfg, dims)  # pgvector ignores enable_sparse (uses Postgres FTS)
    if cfg.backend == "qdrant":
        return QdrantStore(cfg, dims, enable_sparse=enable_sparse)
    raise ValueError(f"unknown vector store backend: {cfg.backend}")
