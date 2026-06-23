"""
ingest.py — build the index: extract text from PDFs → chunk → embed → store.
Run once per corpus (or when documents change).
"""

from __future__ import annotations

import os
import glob

from .config import AppConfig
from .chunking import chunk_document, Chunk
from .embeddings import Embedder
from .stores import make_store


def extract(path: str) -> list[tuple[int, str]]:
    """Extract text as (page_number, text) pairs. .txt is a single page-0 doc. RETURNS list."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return [(i + 1, (page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    with open(path, errors="ignore") as f:
        return [(0, f.read())]


def ingest_paths(cfg: AppConfig, paths: list[str]) -> int:
    """Extract → chunk → embed → store for all files/globs in `paths`.

    PARAM cfg  : AppConfig.
    PARAM paths: file paths or glob patterns.
    RETURNS: number of chunks stored.
    """
    cfg.validate()
    embedder = Embedder(cfg.embedding)
    store = make_store(cfg.vector_store, cfg.embedding.dims)
    store.setup()

    files: list[str] = []
    for p in paths:
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])

    chunks: list[Chunk] = []
    for path in files:
        source = os.path.basename(path)
        for page, text in extract(path):
            chunks.extend(chunk_document(text, source, cfg.chunking, page=page))
    if not chunks:
        return 0

    BATCH = 128
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vectors = embedder.embed_texts([c.text for c in batch])
        store.upsert(batch, vectors)
    return len(chunks)
