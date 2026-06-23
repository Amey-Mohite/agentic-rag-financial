"""
embeddings.py — OpenAI text-embedding-3-large, L2-normalized so cosine == dot product.
Query and document must use the same model + dims or similarity is meaningless.
"""

from __future__ import annotations

import numpy as np

from .config import EmbeddingConfig


class Embedder:
    """Wraps the OpenAI embeddings API behind embed_texts(texts) -> [N, dims]."""

    def __init__(self, cfg: EmbeddingConfig):
        from openai import OpenAI
        self._client = OpenAI()          # reads OPENAI_API_KEY
        self._model = cfg.model
        self._dims = cfg.dims

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batch-embed (up to 2048 inputs/call) and L2-normalize.

        PARAM texts: chunk or query strings.
        RETURNS: [N, dims] float32, unit-length rows.
        """
        resp = self._client.embeddings.create(model=self._model, input=texts, dimensions=self._dims)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)
        # EXAMPLE OUTPUT: np.ndarray shape (len(texts), 3072), each row unit length

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. RETURNS [dims] float32 normalized vector."""
        return self.embed_texts([text])[0]
