"""
retrieval.py — the retrieval stage: find the best chunks for a query.

THE THREE-STEP RETRIEVAL PIPELINE
---------------------------------
1. SEARCH      — embed the query and pull a CANDIDATE POOL from the store (dense). If hybrid
                 is on, also run sparse/keyword search and FUSE the two ranked lists.
2. FUSE (RRF)  — Reciprocal Rank Fusion merges the dense and sparse lists using each item's
                 RANK (position), not its raw score. Items ranked high in EITHER list rise.
3. RERANK      — optionally re-score the pool with a cross-encoder (a model that reads the
                 query and chunk TOGETHER and outputs a precise relevance score), then keep
                 only the final top_k.

WHY EACH STEP EXISTS
--------------------
- Dense catches meaning; sparse catches exact terms/numbers. Together they cover each
  other's blind spots.
- RRF avoids the apples-to-oranges problem: dense cosine (~0..1) and sparse ts_rank
  (unbounded) aren't comparable, so we fuse by position instead.
- Reranking is slow but precise. We run cheap retrieval to get ~20 candidates, then spend
  the expensive cross-encoder only on those 20 to pick the best 5.
All of this is toggled by config: use_hybrid and use_reranker.
"""

from __future__ import annotations

# Config knobs (top_k, candidate_pool, toggles, rrf_k, reranker model).
from .config import RetrievalConfig
# The store interface + the Hit result type.
from .stores import VectorStore, Hit
# The query embedder.
from .embeddings import Embedder

# Module-level cache of loaded cross-encoder rerankers, shared across all Retriever instances
# (keyed by model name) so a multi-session server loads each reranker only once.
_RERANKERS: dict = {}


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Fuse several ranked lists into one, using Reciprocal Rank Fusion (RRF).

    THE FORMULA: each item's fused score = sum over the lists it appears in of
    1 / (rrf_k + rank), where rank is its 0-based position in that list. An item near the
    top of a list (small rank) contributes a large term; appearing in MULTIPLE lists stacks
    those terms, so items found by both dense AND sparse bubble to the top.

    WHY RANK INSTEAD OF RAW SCORES: dense cosine scores (~0..1) and sparse ts_rank scores
    (unbounded) live on different scales and can't be added directly. RRF sidesteps this
    entirely by only using POSITION, which is comparable across any retrievers.

    Parameters
    ----------
    ranked_lists : list[list[Hit]]
        Each inner list is one retriever's results, ordered best-first
        (e.g. [dense_hits, sparse_hits]).
    rrf_k : int, optional
        The smoothing constant (60 is the literature standard). Larger values flatten the
        contribution differences between ranks.

    Returns
    -------
    list[Hit]
        A single de-duplicated list (by Hit.id) sorted by fused score, highest first. Each
        returned Hit's `score` field is overwritten with its fused score.
    """
    fused: dict[int, float] = {}   # hit id -> accumulated fused score
    by_id: dict[int, Hit] = {}     # hit id -> a representative Hit object (to rebuild later)
    # Walk every list and every item; `enumerate` gives the 0-based rank (position).
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            # Add this list's contribution for this id. `.get(id, 0.0)` starts new ids at 0.
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            by_id[hit.id] = hit  # remember the Hit so we can return text/source/page later
    out = []
    # Sort ids by their fused score, descending (best first), and rebuild Hit objects with
    # the fused score in the `score` slot.
    for i in sorted(fused, key=lambda i: fused[i], reverse=True):
        h = by_id[i]
        out.append(Hit(h.id, h.text, h.source, h.page, fused[i]))
    return out


class Retriever:
    """Orchestrates embed → store search → fuse → rerank → top-k for a single query.

    Holds references to the store, the embedder, and the retrieval config. The cross-encoder
    reranker is loaded LAZILY (only on first use) because it's a heavy model download.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, cfg: RetrievalConfig,
                 sparse_embedder=None):
        """Wire in the collaborators; defer loading the reranker until it's actually needed.

        Parameters
        ----------
        store, embedder, cfg : as before.
        sparse_embedder : SparseEmbedder | None
            When provided AND the store supports native hybrid, search() fuses dense + learned
            sparse vectors inside the store (true hybrid). Otherwise we fall back to the app-side
            dense + keyword + RRF path.
        """
        self._store = store                  # where vectors live (Qdrant/pgvector)
        self._embedder = embedder            # turns the query into a DENSE vector
        self._sparse_embedder = sparse_embedder  # turns the query into a SPARSE vector (optional)
        self._cfg = cfg                      # retrieval knobs
        self._reranker = None                # lazy: filled in by _get_reranker() on first search
        # Use native hybrid only if BOTH the store supports it and we have a sparse embedder.
        self._native_hybrid = bool(getattr(store, "supports_native_hybrid", False)
                                    and sparse_embedder is not None)

    def _get_reranker(self):
        """Lazily construct and cache the cross-encoder reranker.

        Cached in a MODULE-LEVEL dict keyed by model name so every Retriever / web session SHARES
        one loaded cross-encoder (it's the same model regardless of API keys), keeping memory
        bounded on a multi-session server.
        """
        if self._reranker is None:
            name = self._cfg.reranker_model
            if name not in _RERANKERS:
                from sentence_transformers import CrossEncoder  # local import: heavy dependency
                _RERANKERS[name] = CrossEncoder(name)
            self._reranker = _RERANKERS[name]
        return self._reranker

    def search(self, query: str) -> list[Hit]:
        """Retrieve the best chunks for a query, honoring the hybrid/rerank toggles.

        Parameters
        ----------
        query : str
            The natural-language search query (often produced by the agent, not the end user).

        Returns
        -------
        list[Hit]
            Up to `top_k` chunks, best first.
        """
        # 1. Embed the query into the same space as the stored chunk vectors.
        qvec = self._embedder.embed_query(query)
        # 2. Build the candidate POOL (wider than top_k so the reranker has options).
        if self._cfg.use_hybrid and self._native_hybrid:
            # BEST PATH — true hybrid fused inside the store (dense + learned SPLADE/BM42 sparse,
            # one round-trip, server-side RRF). See QdrantStore.hybrid_search.
            sparse_qvec = self._sparse_embedder.embed_query(query)
            pool = self._store.hybrid_search(qvec, sparse_qvec, self._cfg.candidate_pool)
        elif self._cfg.use_hybrid:
            # FALLBACK PATH — dense + keyword sparse fused app-side with RRF (no sparse model, or
            # a backend without native hybrid such as pgvector).
            dense = self._store.dense_search(qvec, self._cfg.candidate_pool)
            sparse = self._store.sparse_search(query, self._cfg.candidate_pool)
            pool = reciprocal_rank_fusion([dense, sparse], self._cfg.rrf_k)
        else:
            # Dense-only retrieval.
            pool = self._store.dense_search(qvec, self._cfg.candidate_pool)
        # 4. Trim the fused pool back to candidate_pool size before the (costly) rerank.
        pool = pool[: self._cfg.candidate_pool]
        if not pool:
            return []  # nothing found — return empty rather than erroring
        # 5. Optional precise rerank: score each (query, chunk) pair with the cross-encoder.
        if self._cfg.use_reranker:
            # The cross-encoder reads query+chunk together and returns a relevance score per pair.
            scores = self._get_reranker().predict([(query, h.text) for h in pool])
            for h, s in zip(pool, scores):
                h.score = float(s)  # overwrite each Hit's score with the cross-encoder's verdict
            pool.sort(key=lambda h: h.score, reverse=True)  # best first
        # 6. Hand back only the final top_k chunks.
        return pool[: self._cfg.top_k]
