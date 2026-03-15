"""
indexing/vector_index.py
Dense embedding + FAISS ANN index with persistent checkpointing.

Supports:
  - sentence-transformers (local, default)
  - OpenAI text-embedding-3-small (when USE_OPENAI_API=true)
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import (
    EMBEDDING_MODEL, USE_OPENAI_API, EMBED_BATCH, EMBED_DIM,
    FAISS_INDEX_PATH, CHUNKS_CACHE_PATH, EMBED_CACHE_PATH,
    ANN_TOPK, GPU_DEVICE, OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_with_st(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """sentence-transformers local embeddings."""
    from sentence_transformers import SentenceTransformer
    device = GPU_DEVICE if GPU_DEVICE != "cpu" else None
    model  = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)


def _embed_with_openai(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """OpenAI embeddings."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp  = client.embeddings.create(input=batch, model=model_name)
        vecs  = [np.array(item.embedding, dtype="float32") for item in resp.data]
        all_embeddings.extend(vecs)
    return np.vstack(all_embeddings)


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Route to the correct embedding backend based on config."""
    if USE_OPENAI_API and "text-embedding" in model_name:
        return _embed_with_openai(texts, model_name, EMBED_BATCH)
    return _embed_with_st(texts, model_name, EMBED_BATCH)


# ── VectorIndex ───────────────────────────────────────────────────────────────

@dataclass
class VectorIndex:
    index: object          # faiss.Index
    chunk_ids: List[str]
    model_name: str
    dim: int

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "chunk_ids":  self.chunk_ids,
                "model_name": self.model_name,
                "dim":        self.dim,
            }, f)
        logger.info("FAISS index saved → %s (%d vectors)", path.name, len(self.chunk_ids))

    @staticmethod
    def load(path: Path) -> "VectorIndex":
        import faiss
        index = faiss.read_index(str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return VectorIndex(
            index=index,
            chunk_ids=meta["chunk_ids"],
            model_name=meta["model_name"],
            dim=meta["dim"],
        )


# ── Build with checkpoint caching ─────────────────────────────────────────────

def build_vector_index(chunks, model_name: str = EMBEDDING_MODEL) -> VectorIndex:
    """
    Builds a FAISS flat-IP index from chunk embeddings.
    Uses EMBED_CACHE_PATH as a checkpoint — only re-embeds new chunks.
    """
    import faiss

    texts     = [c.text for c in chunks]
    chunk_ids = [c.id   for c in chunks]

    # Load cached embeddings if available and same size
    embeddings: np.ndarray | None = None
    if EMBED_CACHE_PATH.exists():
        try:
            cached = np.load(str(EMBED_CACHE_PATH))
            if cached.shape[0] == len(texts):
                logger.info("Loaded cached embeddings for %d chunks.", len(texts))
                embeddings = cached
        except Exception:
            pass

    if embeddings is None:
        logger.info("Computing embeddings for %d chunks …", len(texts))
        embeddings = embed_texts(texts, model_name)
        np.save(str(EMBED_CACHE_PATH), embeddings)
        logger.info("Embeddings cached → %s", EMBED_CACHE_PATH.name)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product ≈ cosine for L2-normalised vectors
    if GPU_DEVICE != "cpu":
        try:
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            logger.warning("GPU FAISS not available — using CPU.")

    index.add(embeddings.astype("float32"))
    return VectorIndex(index=index, chunk_ids=chunk_ids, model_name=model_name, dim=dim)


# ── ANN search ────────────────────────────────────────────────────────────────

def vector_search(vi: VectorIndex, query: str, k: int = ANN_TOPK) -> List[Tuple[str, float]]:
    """Returns [(chunk_id, score)] sorted descending."""
    q_emb = embed_texts([query], vi.model_name)[0].reshape(1, -1).astype("float32")
    distances, indices = vi.index.search(q_emb, k)
    results: List[Tuple[str, float]] = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(vi.chunk_ids):
            results.append((vi.chunk_ids[idx], float(score)))
    return results
