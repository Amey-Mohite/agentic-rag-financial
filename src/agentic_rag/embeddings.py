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

from dataclasses import dataclass

# NumPy gives us fast vector math (the array, the norms, the division).
import numpy as np

# We need the embedding + retrieval configs (model names) from config.
from .config import EmbeddingConfig, RetrievalConfig
# Load .env so OPENAI_API_KEY is present in the environment when the client is built.
from dotenv import load_dotenv
load_dotenv()

# Module-level cache of loaded sparse models, shared across all SparseEmbedder/pipeline instances
# (keyed by model name) so a multi-session server loads each heavy model only once.
_SPARSE_MODELS: dict = {}


class Embedder:
    """Thin wrapper around the OpenAI embeddings API.

    The public surface is deliberately tiny: `embed_texts(list_of_strings)` for batches
    (used at ingest time) and `embed_query(string)` for a single query (used at search
    time). Both return L2-normalized float32 NumPy arrays.
    """

    def __init__(self, cfg: EmbeddingConfig, api_key: str | None = None):
        """Construct the embedder and its underlying OpenAI client.

        Parameters
        ----------
        cfg : EmbeddingConfig
            Provides the model name and output dimensionality.
        api_key : str | None
            Explicit OpenAI key (used by the bring-your-own-keys web UI). When None, the OpenAI
            SDK reads OPENAI_API_KEY from the environment.
        """
        # Import locally (not at module top) so that merely importing this module doesn't
        # require the openai package — only constructing an Embedder does.
        from openai import OpenAI
        # Pass the key explicitly when provided; otherwise let the SDK use the environment.
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
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


@dataclass
class SparseVector:
    """A learned SPARSE embedding: parallel lists of token indices and their weights.

    Unlike a DENSE vector (3072 numbers, mostly non-zero), a sparse vector is mostly zeros — it
    only stores the handful of vocabulary tokens the model thinks matter, each with a weight.
    SPLADE/BM42 produce these. Qdrant stores them natively and scores them by dot product, giving
    "smart keyword" matching that understands term importance (unlike plain BM25/full-text).

    Fields
    ------
    indices : list[int]
        Vocabulary token ids that are present (the non-zero positions).
    values : list[float]
        The weight for each corresponding index (same length as `indices`).
    """
    indices: list[int]
    values: list[float]


class SparseEmbedder:
    """Wraps fastembed's SparseTextEmbedding (SPLADE / BM42) behind a tiny interface.

    This is the engine behind "true" hybrid search: it turns text into learned sparse vectors that
    Qdrant fuses with the dense vectors server-side (see stores.QdrantStore). The model is loaded
    LAZILY on first use because it's a heavy download.

    Why this beats the old keyword fallback: the previous `sparse_search` just filtered payload
    text for the query words (no ranking, no notion of which words matter). SPLADE/BM42 assign
    learned weights — so "depreciation" in a query outranks "the" — and the result is a properly
    ranked sparse signal that complements dense retrieval.
    """

    def __init__(self, cfg: RetrievalConfig):
        """Remember the configured sparse model id; defer loading until first embed call."""
        self._model_name = cfg.sparse_model
        self._model = None  # lazy: built by _get_model() on first use

    def _get_model(self):
        """Lazily construct and cache the fastembed sparse model.

        The model is cached in a MODULE-LEVEL dict keyed by model name, so multiple pipelines
        (e.g. different web sessions) SHARE one loaded model instead of each loading its own —
        keeping memory bounded on a multi-session server.
        """
        if self._model is None:
            if self._model_name not in _SPARSE_MODELS:
                from fastembed import SparseTextEmbedding  # heavy/optional dependency
                _SPARSE_MODELS[self._model_name] = SparseTextEmbedding(model_name=self._model_name)
            self._model = _SPARSE_MODELS[self._model_name]
        return self._model

    def embed_texts(self, texts: list[str]) -> list["SparseVector"]:
        """Sparse-embed a BATCH of strings (documents at ingest time).

        Returns
        -------
        list[SparseVector]
            One SparseVector per input text, in order.
        """
        # fastembed yields objects with `.indices` and `.values` numpy arrays; convert to lists
        # (Qdrant's client wants plain Python lists/ints/floats).
        return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in self._get_model().embed(texts)]

    def embed_query(self, text: str) -> "SparseVector":
        """Sparse-embed a SINGLE query string. Returns one SparseVector."""
        # `query_embed` is fastembed's query-side variant (some sparse models weight queries
        # differently from documents); it yields a generator, so take the first item.
        return next(SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                    for e in self._get_model().query_embed([text]))
