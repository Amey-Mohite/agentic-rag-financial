"""
chunking.py — split document text into token-sized chunks, each carrying its provenance.

WHY CHUNKING MATTERS (read this!)
---------------------------------
Chunking is the #1 silent failure point in a RAG system. The retriever can only return
WHOLE chunks; it never returns half a chunk. So if the sentence that answers the user's
question lands on a boundary — split across two chunks — neither chunk contains the full
fact, the embedding of each half is "diluted", and retrieval quietly fails. You get a
plausible-but-wrong answer and no error message. Good chunking keeps semantically-related
text together and adds a little OVERLAP so a fact straddling a boundary still appears
intact in at least one chunk.

UNITS ARE TOKENS, NOT CHARACTERS
--------------------------------
Everything here is measured in *tokens* (the sub-word units the model actually consumes),
counted with tiktoken's `cl100k_base` encoding — the same tokenizer used by OpenAI's
embedding/GPT models. We size chunks in tokens because that's what both the embedder and
the LLM context window are budgeted in.

THREE STRATEGIES (chosen by config.chunking.strategy)
-----------------------------------------------------
- "fixed"     : cut every N tokens. Crude but predictable. Good baseline / fallback.
- "recursive" : split on paragraphs, then sentences, packing up to the budget. DEFAULT.
- "semantic"  : approximate topic-shift splitting using structural cues (headings/blank lines).
"""

from __future__ import annotations

# `dataclass` gives us the lightweight `Chunk` record below.
from dataclasses import dataclass
# `re` (regular expressions) is used to split on paragraph/sentence/heading boundaries.
import re

# tiktoken is OpenAI's fast tokenizer. We use it to count and slice text by token.
import tiktoken

# We only need the ChunkingConfig type from config (strategy + sizes).
from .config import ChunkingConfig

# Build the encoder ONCE at import time (it's relatively expensive to construct) and reuse
# it everywhere. cl100k_base is the encoding behind text-embedding-3-* and GPT-4-class models.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A retrievable slice of a document PLUS where it came from (so answers can be cited).

    Fields
    ------
    text : str
        The actual chunk text that will be embedded and, if retrieved, shown to the LLM.
    source : str
        The originating document name (e.g. "AAPL_10-K.htm"). Used in citations.
    page : int
        The page the chunk came from (PDFs preserve real page numbers; HTML uses 0).
    ordinal : int
        The chunk's running index within its page/document. Useful for ordering and debugging.
    """
    text: str
    source: str
    page: int = 0     # default 0 for non-paged formats (HTML, plain text)
    ordinal: int = 0  # default 0; set by the splitters as they emit chunks in order


def count_tokens(text: str) -> int:
    """Return how many tokens `text` encodes to under cl100k_base.

    Parameters
    ----------
    text : str
        Any string.

    Returns
    -------
    int
        The number of tokens. We call this constantly to decide whether adding the next
        unit would overflow the chunk budget.
    """
    # `.encode` turns the string into a list of integer token ids; its length is the count.
    return len(_ENC.encode(text))


def chunk_document(text: str, source: str, cfg: ChunkingConfig, page: int = 0) -> list[Chunk]:
    """Public entry point: dispatch to the chunking strategy named in the config.

    This is the ONE function the rest of the codebase calls. It looks at
    `cfg.strategy` and forwards to the matching private splitter.

    Parameters
    ----------
    text : str
        Raw text to split — typically one page (PDF) or a whole document (HTML/text).
    source : str
        Document name, copied into every produced Chunk for provenance/citations.
    cfg : ChunkingConfig
        Holds the strategy name and the token budgets (chunk_tokens, overlap_tokens).
    page : int, optional
        Page number to stamp onto each chunk (default 0).

    Returns
    -------
    list[Chunk]
        The document split into provenance-tagged chunks.

    Raises
    ------
    ValueError
        If cfg.strategy is not one of the three known strategies.
    """
    # Route to the requested algorithm. Each returns a list[Chunk].
    if cfg.strategy == "fixed":
        return _fixed(text, source, cfg, page)
    if cfg.strategy == "recursive":
        return _recursive(text, source, cfg, page)
    if cfg.strategy == "semantic":
        return _semantic(text, source, cfg, page)
    # Defensive: an unknown strategy is a config bug — fail loudly rather than silently.
    raise ValueError(f"unknown chunking strategy: {cfg.strategy}")


def _fixed(text: str, source: str, cfg: ChunkingConfig, page: int) -> list[Chunk]:
    """Strategy 1 — cut every N tokens with a fixed overlap. Crude, predictable baseline.

    It completely ignores sentence/paragraph structure, so it can slice mid-sentence. Its
    virtue is determinism and simplicity; it's also the fallback the other strategies use
    when they meet a single unit that's larger than the whole budget.
    """
    # Encode the entire text to a flat list of token ids so we can slice by token position.
    toks = _ENC.encode(text)
    # The window advances by (chunk_tokens - overlap_tokens) each step, so consecutive
    # chunks share `overlap_tokens` tokens. e.g. 500 budget, 50 overlap → step of 450.
    step = cfg.chunk_tokens - cfg.overlap_tokens
    out, start, ordinal = [], 0, 0  # output list, current token offset, running chunk index
    # Walk the token list in windows until we've consumed all tokens.
    while start < len(toks):
        # Slice out one window and decode it back to text. `.strip()` trims edge whitespace.
        piece = _ENC.decode(toks[start:start + cfg.chunk_tokens]).strip()
        if piece:  # skip empty/whitespace-only windows
            out.append(Chunk(piece, source, page, ordinal)); ordinal += 1
        start += step  # advance the window (note: overlap means step < chunk_tokens)
    return out


def _recursive(text: str, source: str, cfg: ChunkingConfig, page: int) -> list[Chunk]:
    """Strategy 2 (DEFAULT) — respect structure: split on paragraphs, then sentences, then pack.

    The idea: never break inside a sentence if we can help it. We first break the text into
    "units" (whole paragraphs, or individual sentences when a paragraph is too big), then
    greedily PACK consecutive units into a chunk until adding one more would exceed the
    token budget. This keeps related text together far better than blind fixed slicing.
    """
    # Step A: split into paragraphs on blank lines. Keep only non-empty, trimmed paragraphs.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Step B: turn paragraphs into "units". A paragraph that already fits stays whole;
    # an oversized paragraph is broken into sentences so the packer has finer-grained pieces.
    units: list[str] = []
    for p in paragraphs:
        if count_tokens(p) <= cfg.chunk_tokens:
            units.append(p)  # paragraph fits — keep it intact
        else:
            # Split on sentence enders (. ! ?) followed by whitespace. The lookbehind
            # `(?<=[.!?])` keeps the punctuation attached to the sentence it ends.
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())

    # Step C: greedily pack units into chunks up to the token budget.
    out, buf, ordinal = [], "", 0  # output list, current accumulating buffer, running index
    for u in units:
        # Tentatively append the next unit to the buffer (with a space if buffer non-empty).
        cand = (buf + " " + u).strip() if buf else u
        if count_tokens(cand) <= cfg.chunk_tokens:
            buf = cand  # still fits — accept the candidate and keep going
        else:
            # Adding `u` would overflow. First, flush whatever is currently buffered.
            if buf:
                out.append(Chunk(buf, source, page, ordinal)); ordinal += 1
            # Edge case: a single unit that is itself bigger than the whole budget. We can't
            # pack it, so fall back to fixed-size slicing for just that unit.
            if count_tokens(u) > cfg.chunk_tokens:
                for sub in _fixed(u, source, cfg, page):
                    sub.ordinal = ordinal; ordinal += 1  # renumber to keep ordinals sequential
                    out.append(sub)
                buf = ""  # buffer consumed
            else:
                buf = u  # start a new buffer with this unit
    # Don't forget the final buffered chunk after the loop ends.
    if buf:
        out.append(Chunk(buf, source, page, ordinal))
    return out


def _semantic(text: str, source: str, cfg: ChunkingConfig, page: int) -> list[Chunk]:
    """Strategy 3 — approximate topic-shift splitting using STRUCTURAL cues, then size-bound.

    NOTE / CONCEPT: a "true" semantic chunker embeds successive sentences and starts a new
    chunk wherever the cosine similarity between neighbors drops (a topic change). That
    requires injecting the embedder into the chunker and is more expensive. To keep this
    module dependency-free and fast, we APPROXIMATE topic boundaries with structure:
    blank lines and short Title-Case lines that look like section headings. Each resulting
    block is then capped to the token budget (falling back to fixed slicing if too big).
    """
    # Split on EITHER:
    #   `\n(?=[A-Z][^\n]{0,60}\n)` — a newline right before a short line starting with a
    #     capital letter (a likely heading like "Risk Factors"), OR
    #   `\n\s*\n` — a blank line (paragraph break).
    blocks = [b.strip() for b in re.split(r"\n(?=[A-Z][^\n]{0,60}\n)|\n\s*\n", text) if b.strip()]
    out, ordinal = [], 0
    for b in blocks:
        if count_tokens(b) <= cfg.chunk_tokens:
            out.append(Chunk(b, source, page, ordinal)); ordinal += 1  # block fits — keep whole
        else:
            # Block too large — slice it with the fixed splitter and renumber ordinals.
            for sub in _fixed(b, source, cfg, page):
                sub.ordinal = ordinal; ordinal += 1
                out.append(sub)
    return out
