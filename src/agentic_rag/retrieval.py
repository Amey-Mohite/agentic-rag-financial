"""
retrieval.py — hybrid retrieval: dense + sparse fused with Reciprocal Rank Fusion, then a
cross-encoder rerank. Toggled by config (use_hybrid / use_reranker).
"""

from __future__ import annotations

from .config import RetrievalConfig
from .stores import VectorStore, Hit
from .embeddings import Embedder


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Fuse ranked lists via RRF: score = sum 1/(rrf_k + rank). Uses rank, not raw scores.

    WHY: dense (cosine ~0-1) and sparse (ts_rank, unbounded) scores aren't comparable; RRF sidesteps
    that by using only position. Items high in EITHER list rise. De-dupes by hit id.
    PARAM ranked_lists: e.g. [dense_hits, sparse_hits], each best-first.
    PARAM rrf_k       : RRF constant (60 standard).
    RETURNS: fused list[Hit] sorted by fused score desc.
    """
    fused: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            by_id[hit.id] = hit
    out = []
    for i in sorted(fused, key=lambda i: fused[i], reverse=True):
        h = by_id[i]
        out.append(Hit(h.id, h.text, h.source, h.page, fused[i]))
    return out


class Retriever:
    """Query embedding + store search + fusion + reranking -> top-k Hits."""

    def __init__(self, store: VectorStore, embedder: Embedder, cfg: RetrievalConfig):
        self._store = store
        self._embedder = embedder
        self._cfg = cfg
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self._cfg.reranker_model)
        return self._reranker

    def search(self, query: str) -> list[Hit]:
        """Retrieve best chunks for a query (honors hybrid/rerank toggles). RETURNS list[Hit]."""
        qvec = self._embedder.embed_query(query)
        dense = self._store.dense_search(qvec, self._cfg.candidate_pool)
        if self._cfg.use_hybrid:
            sparse = self._store.sparse_search(query, self._cfg.candidate_pool)
            pool = reciprocal_rank_fusion([dense, sparse], self._cfg.rrf_k)
        else:
            pool = dense
        pool = pool[: self._cfg.candidate_pool]
        if not pool:
            return []
        if self._cfg.use_reranker:
            scores = self._get_reranker().predict([(query, h.text) for h in pool])
            for h, s in zip(pool, scores):
                h.score = float(s)
            pool.sort(key=lambda h: h.score, reverse=True)
        return pool[: self._cfg.top_k]
