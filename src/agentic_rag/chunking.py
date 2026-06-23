"""
chunking.py — split document text into token-sized chunks with provenance.

Chunking is the #1 silent failure point in RAG: if the answer is split across a boundary, retrieval
can't surface it. Three strategies, chosen by config. Sizes are measured in tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import tiktoken

from .config import ChunkingConfig

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A retrievable slice of a document plus where it came from (for citations)."""
    text: str
    source: str
    page: int = 0
    ordinal: int = 0


def count_tokens(text: str) -> int:
    """Token count via cl100k_base. RETURNS int."""
    return len(_ENC.encode(text))


def chunk_document(text: str, source: str, cfg: ChunkingConfig, page: int = 0) -> list[Chunk]:
    """Dispatch to the configured chunking strategy.

    PARAM text  : raw text (one page or whole doc).
    PARAM source: document name for provenance.
    PARAM cfg   : ChunkingConfig.
    PARAM page  : page number (for citations).
    RETURNS: list[Chunk].
    """
    if cfg.strategy == "fixed":
        return _fixed(text, source, cfg, page)
    if cfg.strategy == "recursive":
        return _recursive(text, source, cfg, page)
    if cfg.strategy == "semantic":
        return _semantic(text, source, cfg, page)
    raise ValueError(f"unknown chunking strategy: {cfg.strategy}")


def _fixed(text: str, source: str, cfg: ChunkingConfig, page: int) -> list[Chunk]:
    """Cut every N tokens with overlap. Crude, predictable baseline."""
    toks = _ENC.encode(text)
    step = cfg.chunk_tokens - cfg.overlap_tokens
    out, start, ordinal = [], 0, 0
    while start < len(toks):
        piece = _ENC.decode(toks[start:start + cfg.chunk_tokens]).strip()
        if piece:
            out.append(Chunk(piece, source, page, ordinal)); ordinal += 1
        start += step
    return out


def _recursive(text: str, source: str, cfg: ChunkingConfig, page: int) -> list[Chunk]:
    """Split on paragraphs then sentences, packing to the token budget. Recommended default."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if count_tokens(p) <= cfg.chunk_tokens:
            units.append(p)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())
    out, buf, ordinal = [], "", 0
    for u in units:
        cand = (buf + " " + u).strip() if buf else u
        if count_tokens(cand) <= cfg.chunk_tokens:
            buf = cand
        else:
            if buf:
                out.append(Chunk(buf, source, page, ordinal)); ordinal += 1
            buf = u
    if buf:
        out.append(Chunk(buf, source, page, ordinal))
    return out


def _semantic(text: str, source: str, cfg: ChunkingConfig, page: int) -> list[Chunk]:
    """Approximate topic-shift splitting via structural breaks, then size-bound each block.

    NOTE: a full semantic chunker embeds successive sentences and splits where similarity drops;
    that needs the embedder injected. This structural approximation keeps the module dependency-free.
    """
    blocks = [b.strip() for b in re.split(r"\n(?=[A-Z][^\n]{0,60}\n)|\n\s*\n", text) if b.strip()]
    out, ordinal = [], 0
    for b in blocks:
        if count_tokens(b) <= cfg.chunk_tokens:
            out.append(Chunk(b, source, page, ordinal)); ordinal += 1
        else:
            for sub in _fixed(b, source, cfg, page):
                sub.ordinal = ordinal; ordinal += 1
                out.append(sub)
    return out
