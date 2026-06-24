"""
ingest.py — build the search index. This is the OFFLINE half of RAG.

THE INGEST PIPELINE
-------------------
    files → extract text → chunk → embed → store
You run this ONCE per corpus (or whenever the documents change). After ingest, the vector
store holds every chunk + its embedding + provenance, ready for the agent to query at runtime.

Supported inputs: PDFs (per-page text, real page numbers preserved for citations) and HTML
filings from SEC EDGAR (tags stripped to clean text). Anything else is read as plain text.
"""

from __future__ import annotations

import os    # for basename (document name) and makedirs
import glob  # for expanding patterns like "data/*.htm" into file lists

# The building blocks: config, the chunker, the embedder, and the store factory.
from .config import AppConfig
from .chunking import chunk_document, Chunk
from .embeddings import Embedder
from .stores import make_store


def extract(path: str) -> list[tuple[int, str]]:
    """Read a file and return its text as a list of (page_number, text) pairs.

    Returning PAGES (not one blob) lets us stamp real page numbers onto chunks for citations.

    Behavior by file type
    ----------------------
    - .pdf         : one (page_number, text) pair per page via pypdf (1-based page numbers).
    - .htm / .html : strip tags/scripts to readable text (EDGAR filings are HTML). Returns a
                     single (0, text) pair since HTML has no intrinsic page numbers.
    - other        : read as plain text → single (0, text) pair.

    WHY PARSE HTML rather than read it raw: raw HTML dumps `<div>`, `<script>`, CSS, and JS
    into the text stream. That noise pollutes the embeddings and wrecks retrieval. We strip to
    clean prose first.

    Parameters
    ----------
    path : str
        Path to the source file.

    Returns
    -------
    list[tuple[int, str]]
        (page_number, page_text) pairs.
    """
    low = path.lower()  # lowercase once so extension checks are case-insensitive
    if low.endswith(".pdf"):
        from pypdf import PdfReader  # local import: only needed for PDFs
        reader = PdfReader(path)
        # enumerate pages from 0, but report 1-based page numbers (i+1) as humans expect.
        # `page.extract_text() or ""` guards against pages that yield None (e.g. scanned images).
        return [(i + 1, (page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    if low.endswith((".htm", ".html")):
        # Read the raw HTML (ignore undecodable bytes) then convert to clean text.
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        return [(0, _html_to_text(html))]
    # Fallback: treat the file as plain text.
    with open(path, errors="ignore") as f:
        return [(0, f.read())]


def _html_to_text(html: str) -> str:
    """Convert an HTML string into readable plain text.

    Prefers BeautifulSoup (best quality) but FALLS BACK to a pure-regex stripper if bs4 isn't
    installed — so the pipeline keeps working with zero extra dependencies.

    Parameters
    ----------
    html : str
        Raw HTML markup.

    Returns
    -------
    str
        Cleaned, de-tagged text with runaway blank lines collapsed.
    """
    try:
        # --- Preferred path: BeautifulSoup parses the DOM properly.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # Remove <script> and <style> elements entirely — their contents are code, not prose.
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Extract visible text, using newlines between elements so structure survives a bit.
        text = soup.get_text(separator="\n")
    except Exception:
        # --- Fallback path: no bs4 available → strip with regexes.
        import re
        # Drop whole <script>/<style> blocks (including their contents). (?is): dot-all + ignorecase.
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
        # Remove every remaining tag (anything between < and >).
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        import html as _h
        # Turn HTML entities (&amp;, &lt;, …) back into real characters.
        text = _h.unescape(text)
    # Collapse 3+ consecutive blank lines into one blank line (HTML extraction makes lots).
    import re
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def ingest_paths(cfg: AppConfig, paths: list[str]) -> int:
    """Run the full ingest pipeline for the given files/globs and return how many chunks were stored.

    Steps: validate config → build embedder + store → create schema → expand globs →
    extract+chunk every file → embed in batches → upsert.

    Parameters
    ----------
    cfg : AppConfig
        The full application config.
    paths : list[str]
        File paths and/or glob patterns (e.g. ["data/*.htm", "report.pdf"]).

    Returns
    -------
    int
        Total number of chunks stored (0 if nothing was found).
    """
    cfg.validate()                                       # fail fast on misconfig
    embedder = Embedder(cfg.embedding)                   # the embedding model wrapper
    store = make_store(cfg.vector_store, cfg.embedding.dims)  # the chosen vector DB
    store.setup()                                        # create table/collection + indexes

    # --- Expand any glob patterns into a concrete file list. A path containing *, ?, or [ is
    # treated as a glob; otherwise it's used literally.
    files: list[str] = []
    for p in paths:
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])

    # --- Extract + chunk every page of every file into one big list of Chunks.
    chunks: list[Chunk] = []
    for path in files:
        source = os.path.basename(path)  # the bare filename, used as the citation source
        for page, text in extract(path):
            chunks.extend(chunk_document(text, source, cfg.chunking, page=page))
    if not chunks:
        return 0  # nothing to embed/store

    # --- Embed + upsert in BATCHES (embedding APIs are far cheaper per item in bulk, and we
    # avoid building one enormous request). 128 chunks per batch is a reasonable default.
    BATCH = 128
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vectors = embedder.embed_texts([c.text for c in batch])  # [batch, dims] matrix
        store.upsert(batch, vectors)                              # write this batch
    return len(chunks)
