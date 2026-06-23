"""
stores.py — pgvector and Qdrant backends behind one VectorStore interface.

Both implement setup / upsert / dense_search / sparse_search. Provenance (source/page/text) travels
with every vector so a hit can be cited directly. Backend is chosen by config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .config import VectorStoreConfig
from .chunking import Chunk


@dataclass
class Hit:
    """One retrieval result with provenance + score."""
    id: int
    text: str
    source: str
    page: int
    score: float


class VectorStore(Protocol):
    def setup(self) -> None: ...
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def dense_search(self, query_vec: np.ndarray, k: int) -> list[Hit]: ...
    def sparse_search(self, query_text: str, k: int) -> list[Hit]: ...


# ----------------------------------------------------------------------------------------------
class PgVectorStore:
    """Postgres + pgvector: halfvec + HNSW for dense, native full-text search for sparse."""

    def __init__(self, cfg: VectorStoreConfig, dims: int):
        self._cfg = cfg
        self._dims = dims

    def setup(self) -> None:
        """Create extension, table, HNSW + GIN indexes (idempotent)."""
        import psycopg
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._cfg.collection} (
                    id SERIAL PRIMARY KEY, source TEXT, page INT, text TEXT,
                    embedding halfvec({self._dims}),
                    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)""")
            cur.execute(f"""CREATE INDEX IF NOT EXISTS {self._cfg.collection}_hnsw
                ON {self._cfg.collection} USING hnsw (embedding halfvec_cosine_ops)
                WITH (m={self._cfg.hnsw_m}, ef_construction={self._cfg.hnsw_ef_construction})""")
            cur.execute(f"""CREATE INDEX IF NOT EXISTS {self._cfg.collection}_fts
                ON {self._cfg.collection} USING GIN (fts)""")
            conn.commit()

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        import psycopg
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            for ch, vec in zip(chunks, vectors):
                cur.execute(
                    f"INSERT INTO {self._cfg.collection} (source,page,text,embedding) VALUES (%s,%s,%s,%s)",
                    (ch.source, ch.page, ch.text, vec.tolist()))
            conn.commit()

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        import psycopg
        with psycopg.connect(self._cfg.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id,text,source,page,1-(embedding <=> %s::halfvec) FROM {self._cfg.collection} "
                f"ORDER BY embedding <=> %s::halfvec LIMIT %s",
                (query_vec.tolist(), query_vec.tolist(), k))
            return [Hit(*row) for row in cur.fetchall()]

    def sparse_search(self, query_text: str, k: int) -> list[Hit]:
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
    """Qdrant: dense ANN via the current query_points API + named vectors."""

    def __init__(self, cfg: VectorStoreConfig, dims: int):
        self._cfg = cfg
        self._dims = dims

    def _client(self):
        from qdrant_client import QdrantClient
        print(f"connecting to Qdrant at {self._cfg.url} with api_key={self._cfg.api_key}")
        return QdrantClient(url=self._cfg.url, api_key=self._cfg.api_key or None)

    def setup(self) -> None:
        from qdrant_client import models
        client = self._client()
        existing = [c.name for c in client.get_collections().collections]
        if self._cfg.collection not in existing:
            client.create_collection(
                collection_name=self._cfg.collection,
                vectors_config={"dense": models.VectorParams(
                    size=self._dims, distance=models.Distance.COSINE)})

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models
        client = self._client()
        points = [models.PointStruct(
            id=i, vector={"dense": vec.tolist()},
            payload={"text": ch.text, "source": ch.source, "page": ch.page})
            for i, (ch, vec) in enumerate(zip(chunks, vectors))]
        client.upsert(collection_name=self._cfg.collection, points=points, wait=True)

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        res = self._client().query_points(
            collection_name=self._cfg.collection, query=query_vec.tolist(),
            using="dense", limit=k, with_payload=True)
        return [Hit(int(p.id), p.payload["text"], p.payload["source"], p.payload["page"], float(p.score))
                for p in res.points]

    def sparse_search(self, query_text: str, k: int) -> list[Hit]:
        """Keyword fallback over payload text. (Add a SPLADE/BM42 sparse vector for true hybrid.)"""
        from qdrant_client import models
        res = self._client().scroll(
            collection_name=self._cfg.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="text", match=models.MatchText(text=query_text))]),
            limit=k, with_payload=True)
        return [Hit(int(p.id), p.payload["text"], p.payload["source"], p.payload["page"], 0.0)
                for p in res[0]]


def make_store(cfg: VectorStoreConfig, dims: int) -> VectorStore:
    """Return the configured store. RAISES on unknown backend."""
    if cfg.backend == "pgvector":
        return PgVectorStore(cfg, dims)
    if cfg.backend == "qdrant":
        return QdrantStore(cfg, dims)
    raise ValueError(f"unknown vector store backend: {cfg.backend}")
