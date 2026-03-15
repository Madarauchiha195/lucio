"""
run_challenge.py — Lucio Challenge End-to-End Orchestrator

Phase 1: Download corpus (calls download_corpus.py logic)
Phase 2: Ingest → chunk → build BM25 + vector index
Phase 3: Load indexes → hybrid retrieve → cross-encoder rerank → async LLM answering
Phase 4: Save lucio_submission.json

Usage:
    python run_challenge.py                    # uses Testing Set Questions.xlsx
    python run_challenge.py --questions q.xlsx # custom question file
    python run_challenge.py --strategy page    # use page chunker instead of token
    python run_challenge.py --force            # rebuild index even if cached
    python run_challenge.py --benchmark        # run benchmark after answering
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import time
from pathlib import Path

# UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ("transformers", "sentence_transformers", "faiss", "httpx", "openai", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("run_challenge")


# ── Phase 1: Download corpus ──────────────────────────────────────────────────

def phase_download():
    logger.info("=== PHASE 1: CORPUS DOWNLOAD ===")
    try:
        from download_corpus import main as download_main
        asyncio.run(download_main())
    except Exception as exc:
        logger.warning("Corpus download skipped (offline/mocked): %s", exc)


# ── Phase 2: Build index ──────────────────────────────────────────────────────

def phase_build(strategy: str = "token", force: bool = False) -> tuple:
    from config import BM25_INDEX_PATH, FAISS_INDEX_PATH, CACHE_DIR

    if not force and BM25_INDEX_PATH.exists() and FAISS_INDEX_PATH.exists():
        logger.info("=== PHASE 2: LOADING CACHED INDEX ===")
        from indexing.bm25_index import BM25Index
        from indexing.vector_index import VectorIndex
        import pickle

        bm25_idx = BM25Index.load(BM25_INDEX_PATH)
        vec_idx  = VectorIndex.load(FAISS_INDEX_PATH)
        with open(CACHE_DIR / "chunks.pkl", "rb") as f:
            import pickle
            chunks = pickle.load(f)
        return chunks, bm25_idx, vec_idx

    logger.info("=== PHASE 2: INGESTION + CHUNKING + INDEXING ===")
    t0 = time.time()

    from ingestion.ingest import ingest_all
    from processing.chunker import chunk_all
    from indexing.bm25_index import build_bm25
    from indexing.vector_index import build_vector_index
    from config import DOCS_DIR, CACHE_DIR
    import pickle

    pages  = ingest_all(DOCS_DIR)
    chunks = chunk_all(pages, strategy=strategy)
    logger.info("Total chunks: %d", len(chunks))

    bm25_idx = build_bm25(chunks)
    bm25_idx.save(BM25_INDEX_PATH)

    vec_idx = build_vector_index(chunks)
    vec_idx.save(FAISS_INDEX_PATH)

    with open(CACHE_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Build complete in %.1f s.", time.time() - t0)
    return chunks, bm25_idx, vec_idx


# ── Phase 3: Answer questions ─────────────────────────────────────────────────

def phase_query(chunks, bm25_idx, vec_idx, questions: list[str]) -> list:
    logger.info("=== PHASE 3: RETRIEVAL + RERANKING + LLM ANSWERING ===")

    from indexing.hybrid_retriever import HybridRetriever
    from reranker.cross_encoder import rerank
    from qa.answer_generator import answer_all_async, AnswerResult

    retriever = HybridRetriever(bm25_idx, vec_idx, chunks)

    # Retrieve + rerank (sync, fast)
    reranked_per_q = []
    for q in questions:
        candidates = retriever.get_candidate_chunks(q)
        reranked   = rerank(q, candidates)
        reranked_per_q.append(reranked)

    # Build a retriever wrapper that returns pre-computed reranked results
    class _StaticRetriever:
        def get_candidate_chunks(self, q, k=10):
            idx = questions.index(q) if q in questions else 0
            return reranked_per_q[idx]

    results = asyncio.run(answer_all_async(questions, _StaticRetriever()))
    return results


# ── Phase 4: Save submission ──────────────────────────────────────────────────

def phase_save(results) -> Path:
    logger.info("=== PHASE 4: SAVING SUBMISSION ===")
    data = [r.to_dict() for r in results if r]
    out  = Path("lucio_submission.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Submission saved → %s (%d answers)", out, len(data))
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_questions(path: str | None = None) -> list[str]:
    """Load questions from Excel or fall back to default questions.py."""
    import pandas as pd

    excel_path = Path(path) if path else Path("d:/lucio/documents/Testing Set Questions.xlsx")
    if excel_path.exists():
        df = pd.read_excel(excel_path)
        qs = df["Question"].dropna().tolist()
        logger.info("Loaded %d questions from %s", len(qs), excel_path.name)
        return qs

    logger.warning("Questions file not found — using questions.py defaults")
    from questions import QUESTIONS
    return QUESTIONS


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lucio Challenge — Advanced Legal QA Pipeline")
    parser.add_argument("--questions", default=None,   help="Path to Excel questions file")
    parser.add_argument("--strategy",  default="token", choices=["token", "page"], help="Chunking strategy")
    parser.add_argument("--force",     action="store_true", help="Force full index rebuild")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after answering")
    parser.add_argument("--download",  action="store_true", help="Run corpus downloader first")
    parser.add_argument("--verbose",   action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    t_total = time.time()

    if args.download:
        phase_download()

    questions              = load_questions(args.questions)
    chunks, bm25_idx, vec_idx = phase_build(strategy=args.strategy, force=args.force)
    results                = phase_query(chunks, bm25_idx, vec_idx, questions)
    out_path               = phase_save(results)

    total = time.time() - t_total
    logger.info("Challenge complete in %.1f s. Output: %s", total, out_path)

    if args.benchmark:
        from performance.benchmark import simulate_lucio_challenge
        simulate_lucio_challenge(questions=questions[:5])


if __name__ == "__main__":
    main()
