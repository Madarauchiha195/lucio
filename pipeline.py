"""
pipeline.py
End-to-end orchestrator for the Document QA pipeline.

Two phases:
  1. build_index() – ingest documents, chunk, build BM25 & FAISS indexes.
  2. run_questions(questions) – load indexes, answer all questions concurrently.

All results are written to output/<Q_id>.json.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (
    DOCS_DIR, OUTPUT_DIR, CACHE_DIR,
    BM25_INDEX_PATH, FAISS_INDEX_PATH, CHUNKS_CACHE_PATH,
    LLM_CONCURRENCY, EMBED_MODEL,
)
from ingestion import ingest_all
from chunker import chunk_all, Chunk
from index_bm25 import build_bm25, save_bm25, load_bm25
from index_faiss import build_faiss, save_faiss, load_faiss, _load_model
from retrieval import HybridRetriever
from llm_answer import answer_question, AnswerResult

logger = logging.getLogger(__name__)


# ── Build phase ───────────────────────────────────────────────────────────────

def build_index(force: bool = False) -> None:
    """
    Ingest → chunk → build BM25 + FAISS indexes.
    Skips rebuild if indexes already exist (unless *force=True*).
    """
    if not force and BM25_INDEX_PATH.exists() and FAISS_INDEX_PATH.exists():
        logger.info("Indexes already exist. Skipping build. Use --force to rebuild.")
        return

    t0 = time.time()
    logger.info("=== INGESTION ===")
    pages = ingest_all(DOCS_DIR)
    if not pages:
        logger.warning("No pages ingested. Place documents in: %s", DOCS_DIR)
        return

    logger.info("=== CHUNKING ===")
    chunks: List[Chunk] = chunk_all(pages)
    logger.info("Total chunks: %d", len(chunks))

    logger.info("=== BM25 INDEX ===")
    bm25_idx = build_bm25(chunks)
    save_bm25(bm25_idx, BM25_INDEX_PATH)

    elapsed = time.time() - t0
    
    logger.info("=== FAISS INDEX ===")
    faiss_idx = build_faiss(chunks, model_name=EMBED_MODEL)
    save_faiss(faiss_idx, FAISS_INDEX_PATH, build_duration=elapsed)

    logger.info("Index build complete in %.1f s. Chunks: %d", elapsed, len(chunks))


# ── Query phase ───────────────────────────────────────────────────────────────

def _answer_one(
    retriever: HybridRetriever,
    question: str,
    q_id: int,
) -> AnswerResult:
    """Retrieve chunks and call the LLM for a single question."""
    chunks = retriever.search(question)
    result = answer_question(q_id, question, chunks)
    out_path = OUTPUT_DIR / f"q{q_id:02d}.json"
    out_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def run_questions(questions: List[str]) -> List[AnswerResult]:
    """
    Load indexes, then answer all *questions* concurrently.
    Returns a list of AnswerResult objects in the same order as *questions*.
    """
    logger.info("=== LOADING INDEXES ===")
    bm25_idx  = load_bm25(BM25_INDEX_PATH)
    faiss_idx = load_faiss(FAISS_INDEX_PATH)
    model     = _load_model(faiss_idx.model_name)

    retriever = HybridRetriever(
        bm25_index=bm25_idx,
        faiss_index=faiss_idx,
        embed_model=model,
    )

    logger.info("=== ANSWERING %d QUESTIONS (concurrency=%d) ===",
                len(questions), LLM_CONCURRENCY)
    t0 = time.time()

    results: List[AnswerResult | None] = [None] * len(questions)
    futures = {}

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        for q_id, question in enumerate(questions, start=1):
            fut = pool.submit(_answer_one, retriever, question, q_id)
            futures[fut] = q_id - 1   # store original index

        with tqdm(total=len(questions), desc="Answering", unit="q") as pbar:
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                    pbar.update(1)
                except Exception as exc:
                    question = questions[idx]
                    logger.error("Question %d failed: %s", idx + 1, exc)
                    results[idx] = AnswerResult(
                        question_id=idx + 1,
                        question=question,
                        answer=f"[ERROR] {exc}",
                        sources=[],
                    )
                    pbar.update(1)

    elapsed = time.time() - t0
    answered = [r for r in results if r is not None]
    logger.info("All questions answered in %.1f s.", elapsed)

    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"  Answered {len(answered)} / {len(questions)} questions in {elapsed:.1f}s")
    print(f"{'='*60}\n")
    for i, r in enumerate(answered, start=1):
        if r:
            print(f"Q{i:02d}: {r.question}")
            print(f"     {r.answer[:200]}{'…' if len(r.answer) > 200 else ''}")
            if r.sources:
                src_str = ", ".join(f"{s.get('document', '')} p.{s.get('page', '')}" for s in r.sources[:3])
                print(f"     ↳ Sources: {src_str}")
            print()

    return answered
