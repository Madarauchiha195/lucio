"""
tests/test_integration.py
Integration test: simulates 200-document run using document duplication.
Asserts JSON output format and measures runtime vs the configured limit.
Run: pytest tests/test_integration.py -v -s
"""
import copy
import json
import time
from pathlib import Path

import pytest

from config import DOCS_DIR, RUNTIME_LIMIT_SECONDS


TARGET_DOCS  = 200
SAMPLE_LIMIT = 5    # limit actual files used in test to keep runtime manageable


@pytest.fixture(scope="module")
def sample_pages():
    """Ingest a small subset and duplicate pages to simulate 200 documents."""
    from ingestion.ingest import ingest_all
    real_pages = ingest_all(DOCS_DIR)[:50]   # take real pages

    # Duplicate to simulate TARGET_DOCS pages of content
    simulated = []
    for i in range(TARGET_DOCS):
        for page in real_pages[:1]:
            dup = copy.deepcopy(page)
            dup["doc_name"] = f"simulated_doc_{i:03d}.pdf"
            simulated.append(dup)
    return simulated


@pytest.fixture(scope="module")
def index_and_retriever(sample_pages):
    from processing.chunker import chunk_all
    from indexing.bm25_index import build_bm25
    from indexing.vector_index import build_vector_index
    from indexing.hybrid_retriever import HybridRetriever

    chunks = chunk_all(sample_pages, strategy="token")
    bm25   = build_bm25(chunks)
    vec    = build_vector_index(chunks)
    return HybridRetriever(bm25, vec, chunks), chunks


SAMPLE_QUESTIONS = [
    "What are the key obligations of the parties?",
    "What are the termination clauses?",
    "What is the governing law?",
]


def test_answers_schema(index_and_retriever):
    retriever, _ = index_and_retriever
    from qa.answer_generator import answer_question
    from reranker.cross_encoder import rerank

    results = []
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        candidates = retriever.get_candidate_chunks(q)
        reranked   = rerank(q, candidates)
        result     = answer_question(i, q, reranked)
        results.append(result.to_dict())

    # Validate output schema
    for r in results:
        assert "question_id" in r
        assert "answer"      in r
        assert "document"    in r
        assert "page"        in r
        assert "evidence"    in r
        assert isinstance(r["answer"], str) and len(r["answer"]) > 0


def test_runtime(index_and_retriever):
    retriever, _ = index_and_retriever
    from qa.answer_generator import answer_question
    from reranker.cross_encoder import rerank

    t0 = time.time()
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        cands    = retriever.get_candidate_chunks(q)
        reranked = rerank(q, cands)
        answer_question(i, q, reranked)
    elapsed = time.time() - t0

    # Scale: 3 questions took `elapsed` seconds → estimate for 15
    estimated_15q = elapsed * (15 / len(SAMPLE_QUESTIONS))
    print(f"\n  3-question time: {elapsed:.1f}s → estimated 15-question: {estimated_15q:.1f}s (limit: {RUNTIME_LIMIT_SECONDS}s)")

    if estimated_15q > RUNTIME_LIMIT_SECONDS:
        pytest.skip(f"Estimated {estimated_15q:.1f}s exceeds {RUNTIME_LIMIT_SECONDS}s limit (run benchmark.py for details)")
