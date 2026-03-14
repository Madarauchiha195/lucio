"""
index_faiss.py
Build, persist, and query a FAISS (vector) index over document chunks.

Embeddings are generated with sentence-transformers (all-MiniLM-L6-v2, 384-d)
and cached to disk so re-indexing is fast on subsequent runs.

Public API:
    build_faiss(chunks)           -> FaissIndex
    save_faiss(index)
    load_faiss()                  -> FaissIndex
    embed_query(model, text)      -> np.ndarray  (shape: (1, dim))
    search_faiss(index, vec, top_k) -> List[Tuple[Chunk, float]]
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import (
    EMBED_MODEL, EMBED_BATCH, EMBED_DIM,
    FAISS_INDEX_PATH, CHUNKS_CACHE_PATH, EMBED_CACHE_PATH,
    TOP_K_FAISS,
)
from chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class FaissIndex:
    index: object           # faiss.Index
    chunks: List[Chunk]     # parallel list – faiss result id i ↔ chunks[i]
    model_name: str


def _load_model(model_name: str = EMBED_MODEL):
    """Load a sentence-transformers model (cached by HuggingFace locally)."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s …", model_name)
    return SentenceTransformer(model_name)


def _embed_chunks(model, chunks: List[Chunk]) -> np.ndarray:
    """
    Compute embeddings for all chunks, using a disk cache.
    Returns a float32 array of shape (N, dim).
    """
    texts = [c.text for c in chunks]
    total = len(texts)

    # ── cache check ────────────────────────────────────────────────────────────
    if EMBED_CACHE_PATH.exists() and CHUNKS_CACHE_PATH.exists():
        logger.info("Embedding cache found. Loading …")
        embeddings = np.load(str(EMBED_CACHE_PATH))
        with open(CHUNKS_CACHE_PATH, "rb") as fh:
            cached_chunks: List[Chunk] = pickle.load(fh)
        if len(embeddings) == len(chunks) and len(cached_chunks) == len(chunks):
            # Verify IDs match to guard against stale cache
            if all(a.id == b.id for a, b in zip(cached_chunks, chunks)):
                logger.info("Cache hit: %d embeddings loaded.", len(embeddings))
                return embeddings.astype(np.float32)
        logger.info("Cache mismatch – recomputing embeddings.")

    # ── compute ────────────────────────────────────────────────────────────────
    logger.info("Computing embeddings for %d chunks (batch=%d) …", total, EMBED_BATCH)
    all_vecs: List[np.ndarray] = []
    for start in range(0, total, EMBED_BATCH):
        batch = texts[start: start + EMBED_BATCH]
        vecs = model.encode(batch, batch_size=EMBED_BATCH,
                            show_progress_bar=False,
                            normalize_embeddings=True)
        all_vecs.append(vecs)
        if (start // EMBED_BATCH) % 4 == 0:
            logger.debug("  %d / %d chunks embedded …", min(start + EMBED_BATCH, total), total)

    embeddings = np.vstack(all_vecs).astype(np.float32)

    # ── persist cache ─────────────────────────────────────────────────────────
    np.save(str(EMBED_CACHE_PATH), embeddings)
    with open(CHUNKS_CACHE_PATH, "wb") as fh:
        pickle.dump(chunks, fh, protocol=5)
    logger.info("Embeddings saved → %s", EMBED_CACHE_PATH)

    return embeddings


def build_faiss(chunks: List[Chunk], model_name: str = EMBED_MODEL) -> FaissIndex:
    """
    Embed all chunks and build a FAISS IndexFlatIP (inner-product / cosine after
    L2-normalisation by sentence-transformers).
    """
    import faiss

    model = _load_model(model_name)
    embeddings = _embed_chunks(model, chunks)

    dim = embeddings.shape[1]
    logger.info("Building FAISS index (dim=%d, vectors=%d) …", dim, len(chunks))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index ready: %d vectors.", index.ntotal)

    return FaissIndex(index=index, chunks=chunks, model_name=model_name)


def save_faiss(fi: FaissIndex, path: Path = FAISS_INDEX_PATH, build_duration: float = 0.0) -> None:
    """Save the FAISS binary index and the chunk list to disk."""
    import faiss
    faiss.write_index(fi.index, str(path))
    # chunks already cached by _embed_chunks; save model_name alongside
    meta_path = path.with_suffix(".meta.pkl")
    with open(meta_path, "wb") as fh:
        pickle.dump({"model_name": fi.model_name, "num_chunks": len(fi.chunks), "build_time": build_duration}, fh)
    logger.info("FAISS index saved -> %s", path)


def load_faiss(path: Path = FAISS_INDEX_PATH) -> FaissIndex:
    """Load the FAISS index and chunks from disk."""
    import faiss
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at {path}. Run --build first.")
    if not CHUNKS_CACHE_PATH.exists():
        raise FileNotFoundError(f"Chunk cache not found at {CHUNKS_CACHE_PATH}.")

    logger.info("Loading FAISS index from %s …", path)
    index = faiss.read_index(str(path))

    with open(CHUNKS_CACHE_PATH, "rb") as fh:
        chunks: List[Chunk] = pickle.load(fh)

    meta_path = path.with_suffix(".meta.pkl")
    model_name = EMBED_MODEL
    if meta_path.exists():
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        model_name = meta.get("model_name", EMBED_MODEL)

    logger.info("FAISS index loaded: %d vectors, %d chunks.", index.ntotal, len(chunks))
    return FaissIndex(index=index, chunks=chunks, model_name=model_name)


def embed_query(model, text: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a float32 array of shape (1, dim), L2-normalised.
    """
    vec = model.encode([text], normalize_embeddings=True)
    return vec.astype(np.float32)


def search_faiss(
    fi: FaissIndex,
    query_vec: np.ndarray,
    top_k: int = TOP_K_FAISS,
) -> List[Tuple[Chunk, float]]:
    """
    Search the FAISS index for the top-k nearest neighbours of *query_vec*.
    Returns list of (Chunk, cosine_similarity) in descending order.
    """
    k = min(top_k, fi.index.ntotal)
    scores, indices = fi.index.search(query_vec, k)

    results: List[Tuple[Chunk, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((fi.chunks[int(idx)], float(score)))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from chunker import Chunk

    sample_chunks = [
        Chunk(id="a_p1_c0", doc_name="a.pdf", page_number=1,
              text="The tenant shall pay rent on the first of each month."),
        Chunk(id="b_p1_c0", doc_name="b.pdf", page_number=1,
              text="Force majeure clause: neither party is liable for acts of nature."),
        Chunk(id="c_p1_c0", doc_name="c.pdf", page_number=1,
              text="Payment terms: net 30 days from invoice date."),
    ]
    fi = build_faiss(sample_chunks)
    model = _load_model()
    qvec = embed_query(model, "when is rent due?")
    results = search_faiss(fi, qvec, top_k=3)
    for chunk, score in results:
        print(f"  [{chunk.doc_name}] cosine={score:.4f} | {chunk.text[:60]!r}")
