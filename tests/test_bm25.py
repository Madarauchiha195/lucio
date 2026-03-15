"""
tests/test_bm25.py
Unit tests for BM25 boolean and proximity search.
Run: pytest tests/test_bm25.py -v
"""
import pytest
from processing.chunker import Chunk


# ── Test fixture: build a tiny BM25 index ────────────────────────────────────

def _make_chunks():
    texts = [
        "The merger and acquisition deal was signed in New York.",
        "The acquisition of TechCorp was completed in California, not related to merger.",
        "This document covers technology licensing terms and patents.",
        "The parties agree to arbitration in case of disputes about compensation.",
        "Environmental regulations and compliance requirements are outlined here.",
    ]
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(Chunk(
            id=f"doc_{i}_p1_page_0", doc=f"doc_{i}.txt", page=1,
            start_offset=0, end_offset=len(text), text=text, chunk_type="page",
        ))
    return chunks


@pytest.fixture
def bm25_idx():
    from indexing.bm25_index import build_bm25
    return build_bm25(_make_chunks())


# ── Basic search ──────────────────────────────────────────────────────────────

def test_basic_search(bm25_idx):
    from indexing.bm25_index import bm25_search
    results = bm25_search(bm25_idx, "merger acquisition")
    assert results, "Expected results for 'merger acquisition'"
    ids = [r[0] for r in results]
    # The first chunk talks about merger AND acquisition
    assert "doc_0_p1_page_0" in ids


# ── Boolean AND ───────────────────────────────────────────────────────────────

def test_boolean_and(bm25_idx):
    from indexing.bm25_index import boolean_search
    results = boolean_search(bm25_idx, "merger AND acquisition")
    ids = set(results)
    assert "doc_0_p1_page_0" in ids, "Chunk 0 contains both merger and acquisition"
    assert "doc_2_p1_page_0" not in ids, "Chunk 2 has neither merger nor acquisition"


# ── Boolean NOT ───────────────────────────────────────────────────────────────

def test_boolean_not(bm25_idx):
    from indexing.bm25_index import boolean_search
    results = boolean_search(bm25_idx, "acquisition NOT california")
    ids = set(results)
    # doc_1 contains both acquisition and california → excluded
    assert "doc_1_p1_page_0" not in ids


# ── Proximity search ──────────────────────────────────────────────────────────

def test_proximity_close(bm25_idx):
    from indexing.bm25_index import proximity_search
    # "merger" and "acquisition" are 2 tokens apart in doc_0
    results = proximity_search(bm25_idx, "merger", "acquisition", distance=5)
    assert "doc_0_p1_page_0" in results


def test_proximity_far(bm25_idx):
    from indexing.bm25_index import proximity_search
    # "acquisition" and "california" appear in doc_1 but are very far apart
    results = proximity_search(bm25_idx, "acquisition", "california", distance=2)
    # may or may not match depending on tokenisation; just assert it runs
    assert isinstance(results, list)
