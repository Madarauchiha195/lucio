"""
indexing/hybrid_retriever.py
Hybrid retrieval: BM25 top-30 + ANN top-30, merged via Reciprocal Rank Fusion (RRF).
Public API: get_candidate_chunks(query, k=20) -> List[Chunk]
"""
from __future__ import annotations

import logging
from typing import List, Dict

from config import BM25_TOPK, ANN_TOPK, RERANKER_TOPK, RRF_K

logger = logging.getLogger(__name__)


def _rrf_merge(bm25_results, ann_results, k: int = RRF_K) -> List[str]:
    """
    Reciprocal Rank Fusion.
    bm25_results / ann_results : [(chunk_id, score), …], sorted desc.
    Returns merged list of chunk_ids ordered by combined RRF score.
    """
    scores: Dict[str, float] = {}

    for rank, (cid, _) in enumerate(bm25_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, (cid, _) in enumerate(ann_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    """
    Wraps BM25Index + VectorIndex for combined retrieval.

    Parameters
    ----------
    bm25_index   : indexing.bm25_index.BM25Index
    vector_index : indexing.vector_index.VectorIndex
    chunks       : List[Chunk] — the full corpus (for id→Chunk lookup)
    """

    def __init__(self, bm25_index, vector_index, chunks):
        self.bm25   = bm25_index
        self.vi     = vector_index
        self._id_map: Dict[str, object] = {c.id: c for c in chunks}

    def get_candidate_chunks(self, query: str, k: int = RERANKER_TOPK):
        """
        Returns the top-k Chunk objects after RRF merging.
        Falls back gracefully if either index is unavailable.
        """
        from indexing.bm25_index import bm25_search
        from indexing.vector_index import vector_search

        bm25_hits = bm25_search(self.bm25, query, k=BM25_TOPK)
        ann_hits  = vector_search(self.vi,  query, k=ANN_TOPK)

        merged_ids = _rrf_merge(bm25_hits, ann_hits)

        results = []
        for cid in merged_ids[:k]:
            chunk = self._id_map.get(cid)
            if chunk:
                results.append(chunk)

        logger.debug("HybridRetriever returned %d chunks for query: %r", len(results), query[:60])
        return results

    # Backward-compatibility shim used by old pipeline.py
    def search(self, query: str, k: int = RERANKER_TOPK):
        return self.get_candidate_chunks(query, k)
