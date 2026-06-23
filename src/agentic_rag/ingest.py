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
    """Extract text as (page_number, text) pairs. RETURNS list.

    - .pdf  : per-page text via pypdf (page numbers preserved for citations).
    - .htm/.html : strip tags/scripts to clean text (EDGAR filings are HTML). One page-0 doc.
    - other : read as plain text. One page-0 doc.
    WHY parse HTML rather than read raw: raw HTML dumps tags/JS into the text, which pollutes
    embeddings and retrieval. We strip to readable text first.
    """
    low = path.lower()
    if low.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return [(i + 1, (page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    if low.endswith((".htm", ".html")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        return [(0, _html_to_text(html))]
    with open(path, errors="ignore") as f:
        return [(0, f.read())]


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text.

    Uses BeautifulSoup if available (best quality); otherwise falls back to a regex tag-stripper so
    the pipeline still works with no extra dependency. RETURNS the cleaned text.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception:
        import re
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)  # drop script/style blocks
        text = re.sub(r"(?s)<[^>]+>", " ", text)                   # strip remaining tags
        import html as _h
        text = _h.unescape(text)                                   # &amp; -> &, etc.
    # collapse the blank-line soup HTML extraction tends to produce
    import re
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


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