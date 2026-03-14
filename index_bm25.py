"""
index_bm25.py
Build, persist, and query a BM25 index over document chunks.

Uses rank_bm25.BM25Okapi for efficient keyword-based retrieval.

Public API:
    build_bm25(chunks)   -> BM25Index
    save_bm25(index)
    load_bm25()          -> BM25Index
    search_bm25(index, query, top_k) -> List[Tuple[Chunk, float]]
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from config import BM25_INDEX_PATH, CHUNKS_CACHE_PATH, TOP_K_BM25
from chunker import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


@dataclass
class BM25Index:
    bm25: object                        # BM25Okapi instance
    chunks: List[Chunk]                 # parallel list – bm25 doc i ↔ chunks[i]


def build_bm25(chunks: List[Chunk]) -> BM25Index:
    """Tokenise all chunks and build a BM25Okapi index."""
    from rank_bm25 import BM25Okapi

    logger.info("Building BM25 index over %d chunks …", len(chunks))
    corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    logger.info("BM25 index ready.")
    return BM25Index(bm25=bm25, chunks=chunks)


def save_bm25(index: BM25Index, path: Path = BM25_INDEX_PATH) -> None:
    """Pickle the BM25 index to disk."""
    with open(path, "wb") as fh:
        pickle.dump(index, fh, protocol=5)
    logger.info("BM25 index saved → %s", path)


def load_bm25(path: Path = BM25_INDEX_PATH) -> BM25Index:
    """Load a previously saved BM25 index from disk."""
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found at {path}. Run --build first.")
    logger.info("Loading BM25 index from %s …", path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def search_bm25(
    index: BM25Index,
    query: str,
    top_k: int = TOP_K_BM25,
) -> List[Tuple[Chunk, float]]:
    """
    Score all chunks against *query* and return the top-k (chunk, score) pairs
    in descending score order.
    """
    tokens = _tokenize(query)
    scores = index.bm25.get_scores(tokens)           # numpy array

    # argsort descending
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Tuple[Chunk, float]] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0.0:                              # skip zero-scoring docs
            results.append((index.chunks[idx], score))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from chunker import Chunk

    # Quick smoke test
    sample_chunks = [
        Chunk(id="a_p1_c0", doc_name="a.pdf", page_number=1,
              text="The tenant shall pay rent on the first of each month."),
        Chunk(id="b_p1_c0", doc_name="b.pdf", page_number=1,
              text="Force majeure clause: neither party is liable for acts of nature."),
        Chunk(id="c_p1_c0", doc_name="c.pdf", page_number=1,
              text="Payment terms: net 30 days from invoice date."),
    ]
    idx = build_bm25(sample_chunks)
    results = search_bm25(idx, "rent payment due date", top_k=3)
    for chunk, score in results:
        print(f"  [{chunk.doc_name}] score={score:.3f} | {chunk.text[:60]!r}")
