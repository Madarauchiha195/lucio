"""
chunker.py
Splits Page dicts into overlapping text chunks and attaches metadata.

Chunk dataclass:
    id           – unique string: "<doc_name>_p<page>_c<idx>"
    doc_name     – source file name
    page_number  – physical page (or heuristic for DOCX)
    text         – chunk text
    token_count  – approximate word count

Entry point:
    chunk_all(pages) -> List[Chunk]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from config import CHUNK_SIZE, CHUNK_OVERLAP

Page = Dict[str, Any]


@dataclass
class Chunk:
    id: str
    doc_name: str
    page_number: int
    text: str
    token_count: int = field(init=False)

    def __post_init__(self):
        self.token_count = len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count,
        }


def _sentence_aware_split(text: str, size: int, overlap: int) -> List[str]:
    """
    Slide a window of *size* chars over *text* with *overlap*.
    Tries to break at sentence boundaries ('. ', '! ', '? ', '\n') rather
    than mid-word for cleaner chunks.
    """
    BREAK_PATTERN = re.compile(r'(?<=[.!?\n])\s+')

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)

        if end < length:
            # try to find a clean sentence break near the end
            window = text[start:end]
            # look for last break in the last 20% of the window
            search_from = max(0, len(window) - size // 5)
            breaks = list(BREAK_PATTERN.finditer(window, search_from))
            if breaks:
                end = start + breaks[-1].start() + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # advance with overlap
        next_start = end - overlap
        if next_start <= start:          # safety: always advance
            next_start = start + 1
        start = next_start

    return chunks


def chunk_page(page: Page, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Split a single Page dict into Chunk objects."""
    raw_chunks = _sentence_aware_split(page["text"], size, overlap)
    result: List[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        chunk_id = f"{page['doc_name']}_p{page['page_number']}_c{idx}"
        result.append(Chunk(
            id=chunk_id,
            doc_name=page["doc_name"],
            page_number=page["page_number"],
            text=text,
        ))
    return result


def chunk_all(pages: List[Page], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Chunk every page in the list and return a flat list of Chunk objects.
    Deduplicates by chunk ID in case the same file is ingested twice.
    """
    seen: set[str] = set()
    all_chunks: List[Chunk] = []
    for page in pages:
        for chunk in chunk_page(page, size, overlap):
            if chunk.id not in seen:
                seen.add(chunk.id)
                all_chunks.append(chunk)
    return all_chunks


if __name__ == "__main__":
    sample_page: Page = {
        "doc_name": "sample.pdf",
        "page_number": 1,
        "text": " ".join(["This is a legal contract."] * 100),
    }
    chunks = chunk_page(sample_page)
    print(f"Produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  [{c.id}] {c.text[:60]!r} … ({c.token_count} tokens)")
