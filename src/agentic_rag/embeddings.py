"""
embeddings.py — turn text into vectors with OpenAI's text-embedding-3-large.

WHAT AN EMBEDDING IS
--------------------
An embedding is a fixed-length list of numbers (here, 3072 of them) that captures the
*meaning* of a piece of text. Two texts about the same thing land close together in this
3072-dimensional space; unrelated texts land far apart. That is what makes "search by
meaning" (semantic / dense retrieval) possible.

WHY WE L2-NORMALIZE
-------------------
We scale every vector to unit length (L2 norm = 1). After that, the cosine similarity
between two vectors equals their dot product — which is exactly what vector databases
compute fastest. Normalizing here means the store can use plain dot-product/cosine math
and all scores live in a consistent ~[-1, 1] range.

THE GOLDEN RULE
---------------
Query text and document text MUST be embedded with the SAME model and SAME dimensions.
Mixing models (or dims) produces vectors that live in different spaces, so their distances
are meaningless and retrieval silently returns garbage.
"""

from __future__ import annotations

# NumPy gives us fast vector math (the array, the norms, the division).
import numpy as np

# We only need the EmbeddingConfig (model name + dims) from config.
from .config import EmbeddingConfig
# Load .env so OPENAI_API_KEY is present in the environment when the client is built.
from dotenv import load_dotenv
load_dotenv()


class Embedder:
    """Thin wrapper around the OpenAI embeddings API.

    The public surface is deliberately tiny: `embed_texts(list_of_strings)` for batches
    (used at ingest time) and `embed_query(string)` for a single query (used at search
    time). Both return L2-normalized float32 NumPy arrays.
    """

    def __init__(self, cfg: EmbeddingConfig):
        """Construct the embedder and its underlying OpenAI client.

        Parameters
        ----------
        cfg : EmbeddingConfig
            Provides the model name and output dimensionality.
        """
        # Import locally (not at module top) so that merely importing this module doesn't
        # require the openai package — only constructing an Embedder does.
        from openai import OpenAI
        self._client = OpenAI()   # reads OPENAI_API_KEY from the environment automatically
        self._model = cfg.model   # remember which model to call
        self._dims = cfg.dims     # remember the requested output dimensionality

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a BATCH of strings and L2-normalize each resulting vector.

        Parameters
        ----------
        texts : list[str]
            Chunk texts (ingest) or, occasionally, several queries. The OpenAI API accepts
            up to ~2048 inputs per call, so batch generously to cut round-trips and cost.

        Returns
        -------
        np.ndarray
            A 2-D array of shape [N, dims], dtype float32, where every ROW is a unit-length
            vector. Row i is the embedding of texts[i].
        """
        # One API call embeds the whole batch. `dimensions=` asks the API to return vectors
        # of exactly the size we configured (text-embedding-3 supports truncated dims).
        resp = self._client.embeddings.create(model=self._model, input=texts, dimensions=self._dims)
        # `resp.data` is a list of objects each holding an `.embedding` list. Stack them into
        # a single [N, dims] float32 matrix.
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Compute the L2 norm (length) of each row. keepdims=True keeps shape [N, 1] so it
        # broadcasts cleanly against [N, dims] in the division below.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Divide each row by its length → unit vectors. `np.clip(norms, 1e-8, None)` floors
        # the divisor at a tiny positive number to avoid division-by-zero on an all-zero vector.
        return vecs / np.clip(norms, 1e-8, None)
        # EXAMPLE OUTPUT: np.ndarray of shape (len(texts), 3072); each row has length ~1.0.

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a SINGLE query string.

        Parameters
        ----------
        text : str
            The user's query (or any single string).

        Returns
        -------
        np.ndarray
            A 1-D array of shape [dims], float32, unit length.
        """
        # Reuse the batch path with a one-element list, then take row 0 to get a 1-D vector.
        return self.embed_texts([text])[0]
