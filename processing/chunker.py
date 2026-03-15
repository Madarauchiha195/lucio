"""
processing/chunker.py
Dual chunker strategy for legal documents.

  page_chunker   – one Chunk per input page (preserves page number exactly)
  token_chunker  – sliding-window tiktoken split (400–600 tokens, 12% overlap)

Chunk schema:
    id           : str  "<doc>_<page>_<type>_<idx>"
    doc          : str  original filename
    page         : int  physical/heuristic page number
    start_offset : int  character offset within page text (for token chunker)
    end_offset   : int
    text         : str  chunk text
    chunk_type   : str  "page" | "token"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    TOKEN_CHUNK_MIN, TOKEN_CHUNK_MAX, TOKEN_CHUNK_OVERLAP_PCT,
)

Page = Dict[str, Any]


# ── Chunk dataclass ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id:           str
    doc:          str
    page:         int
    start_offset: int
    end_offset:   int
    text:         str
    chunk_type:   str          # "page" | "token"
    token_count:  int = field(init=False)

    def __post_init__(self):
        self.token_count = len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "doc":          self.doc,
            "page":         self.page,
            "start_offset": self.start_offset,
            "end_offset":   self.end_offset,
            "text":         self.text,
            "chunk_type":   self.chunk_type,
            "token_count":  self.token_count,
        }

    # Backward-compat properties so old code that reads .doc_name still works
    @property
    def doc_name(self) -> str:
        return self.doc

    @property
    def page_number(self) -> int:
        return self.page


# ── Page chunker ───────────────────────────────────────────────────────────────

def page_chunker(pages: List[Page]) -> List[Chunk]:
    """
    One Chunk per page — the simplest and highest-fidelity option.
    Preserves the exact page number from the extractor.
    """
    chunks: List[Chunk] = []
    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue
        doc   = page["doc_name"]
        pg    = page["page_number"]
        cid   = f"{doc}_p{pg}_page_0"
        chunks.append(Chunk(
            id=cid, doc=doc, page=pg,
            start_offset=0, end_offset=len(text),
            text=text, chunk_type="page",
        ))
    return chunks


# ── Token chunker ─────────────────────────────────────────────────────────────

def _get_encoder():
    """Return a tiktoken encoder (cl100k_base works for most modern models)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def token_chunker(pages: List[Page],
                  min_tokens: int = TOKEN_CHUNK_MIN,
                  max_tokens: int = TOKEN_CHUNK_MAX,
                  overlap_pct: float = TOKEN_CHUNK_OVERLAP_PCT) -> List[Chunk]:
    """
    Sliding-window token-level chunker using tiktoken.
    Falls back to character splitting if tiktoken is not installed.
    """
    enc = _get_encoder()
    chunks: List[Chunk] = []

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue
        doc = page["doc_name"]
        pg  = page["page_number"]

        if enc:
            token_ids = enc.encode(text)
            target    = (min_tokens + max_tokens) // 2       # ~500
            overlap   = max(1, int(target * overlap_pct))    # ~60

            start = 0
            idx   = 0
            while start < len(token_ids):
                end        = min(start + max_tokens, len(token_ids))
                chunk_ids  = token_ids[start:end]
                chunk_text = enc.decode(chunk_ids).strip()

                if chunk_text:
                    char_start = text.find(chunk_text[:40])   # approx offset
                    char_end   = char_start + len(chunk_text)
                    cid = f"{doc}_p{pg}_token_{idx}"
                    chunks.append(Chunk(
                        id=cid, doc=doc, page=pg,
                        start_offset=max(0, char_start),
                        end_offset=max(0, char_end),
                        text=chunk_text, chunk_type="token",
                    ))
                    idx += 1

                next_start = end - overlap
                if next_start <= start:
                    next_start = start + 1
                start = next_start
        else:
            # Fallback: character splitting (same as old chunker.py)
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for idx, s in enumerate(range(0, len(text), max(1, step))):
                chunk_text = text[s: s + CHUNK_SIZE].strip()
                if chunk_text:
                    cid = f"{doc}_p{pg}_token_{idx}"
                    chunks.append(Chunk(
                        id=cid, doc=doc, page=pg,
                        start_offset=s, end_offset=s + len(chunk_text),
                        text=chunk_text, chunk_type="token",
                    ))

    return chunks


# ── Combined chunker (default) ────────────────────────────────────────────────

def chunk_all(pages: List[Page], strategy: str = "token") -> List[Chunk]:
    """
    chunk_all(pages, strategy='token' | 'page')
    Deduplicates by chunk id.
    """
    if strategy == "page":
        raw = page_chunker(pages)
    else:
        raw = token_chunker(pages)

    seen: set = set()
    unique: List[Chunk] = []
    for c in raw:
        if c.id not in seen:
            seen.add(c.id)
            unique.append(c)
    return unique
