"""
reranker/cross_encoder.py
Cross-encoder reranker using sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2.
Scores (query, chunk_text) pairs and re-ranks the merged candidate list.
Automatically uses GPU if CUDA is available.
"""
from __future__ import annotations

import logging
from typing import List

from config import RERANKER_MODEL, GPU_DEVICE, RERANKER_TOPK

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        device = GPU_DEVICE if GPU_DEVICE != "cpu" else "cpu"
        _model = CrossEncoder(RERANKER_MODEL, device=device)
        logger.info("Cross-encoder loaded: %s on %s", RERANKER_MODEL, device)
    except Exception as exc:
        logger.warning("Cross-encoder unavailable: %s — skipping reranking.", exc)
        _model = None
    return _model


def rerank(query: str, chunks: List, top_k: int = RERANKER_TOPK) -> List:
    """
    Re-rank *chunks* given *query* using a cross-encoder.
    Returns the top_k chunks sorted by descending cross-encoder score.
    Falls back to returning chunks unchanged if the model is not available.
    """
    model = _get_model()
    if model is None or not chunks:
        return chunks[:top_k]

    pairs  = [(query, c.text) for c in chunks]
    try:
        scores = model.predict(pairs)
    except Exception as exc:
        logger.warning("Cross-encoder predict failed: %s", exc)
        return chunks[:top_k]

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    result = [chunk for chunk, _ in ranked[:top_k]]
    logger.debug("Cross-encoder reranked %d → %d chunks", len(chunks), len(result))
    return result
