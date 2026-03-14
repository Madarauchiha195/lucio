"""
retrieval.py
Hybrid retrieval: parallel BM25 + FAISS search fused with Reciprocal Rank Fusion.

Entry point:
    HybridRetriever.search(query) -> List[Chunk]
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from config import TOP_K_BM25, TOP_K_FAISS, TOP_K_FINAL, RRF_K
from chunker import Chunk
from index_bm25 import BM25Index, search_bm25
from index_faiss import FaissIndex, search_faiss, embed_query

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Chunk, float]]],
    k: int = RRF_K,
) -> List[Tuple[Chunk, float]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.
    Score = sum(1 / (k + rank)) across all lists.
    Returns chunks sorted by fused score descending.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.id] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in fused]


class HybridRetriever:
    """
    Wraps a BM25 index and a FAISS index.
    Call .search(query) to get fused, deduplicated top-K chunks.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        faiss_index: FaissIndex,
        embed_model,
        top_k_bm25: int = TOP_K_BM25,
        top_k_faiss: int = TOP_K_FAISS,
        top_k_final: int = TOP_K_FINAL,
    ):
        self.bm25 = bm25_index
        self.faiss = faiss_index
        self.model = embed_model
        self.top_k_bm25 = top_k_bm25
        self.top_k_faiss = top_k_faiss
        self.top_k_final = top_k_final

    def search(self, query: str) -> List[Chunk]:
        """
        Fire BM25 and FAISS searches concurrently, fuse with RRF,
        and return the top-K unique chunks.
        """
        bm25_results: List[Tuple[Chunk, float]] = []
        faiss_results: List[Tuple[Chunk, float]] = []

        query_vec = embed_query(self.model, query)

        def _bm25():
            return search_bm25(self.bm25, query, self.top_k_bm25)

        def _faiss():
            return search_faiss(self.faiss, query_vec, self.top_k_faiss)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = pool.submit(_bm25)
            fut_faiss = pool.submit(_faiss)
            bm25_results = fut_bm25.result()
            faiss_results = fut_faiss.result()

        logger.debug("BM25: %d results, FAISS: %d results for query: %r",
                     len(bm25_results), len(faiss_results), query[:60])

        fused = _reciprocal_rank_fusion([bm25_results, faiss_results])
        top_chunks = [chunk for chunk, _ in fused[: self.top_k_final]]

        logger.debug("Hybrid retrieval → %d chunks after RRF fusion.", len(top_chunks))
        return top_chunks
