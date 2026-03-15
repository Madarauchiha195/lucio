"""
run_challenge.py — Lucio Challenge End-to-End Orchestrator (FAST v2)

New in v2:
  - --fast mode: BM25-only retrieval (no vector index), enables sub-10s runs
  - Gemini text-embedding-004 used for vector index (100x faster than local ST)
  - NER extraction via spaCy (en_core_web_sm) stored in document graph
  - Questions path auto-detects local + Colab paths
  - Per-phase timing always printed (even without --benchmark flag)

Usage:
    python run_challenge.py                     # standard: BM25 + vectors + reranker
    python run_challenge.py --fast              # BM25-only (<10s on any hardware)
    python run_challenge.py --force             # rebuild index from scratch
    python run_challenge.py --questions q.xlsx  # custom question file
    python run_challenge.py --benchmark         # extended benchmark after run
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

T = {}   # global timing dict


def _tick(stage: str) -> float:
    T[stage] = time.time()
    return T[stage]


def _tock(stage: str) -> float:
    elapsed = time.time() - T[stage]
    logger.info("  ⏱ %s done in %.2fs", stage, elapsed)
    T[stage] = elapsed
    return elapsed


# ── Phase 1: Download corpus ──────────────────────────────────────────────────

def phase_download():
    logger.info("=== PHASE 1: CORPUS DOWNLOAD ===")
    try:
        from download_corpus import main as download_main
        asyncio.run(download_main())
    except Exception as exc:
        logger.warning("Corpus download skipped: %s", exc)


# ── Phase 2: Build index ──────────────────────────────────────────────────────

def phase_build(strategy: str = "page", force: bool = False, fast: bool = False) -> tuple:
    from config import BM25_INDEX_PATH, FAISS_INDEX_PATH, CACHE_DIR, DOCS_DIR
    import pickle

    cache_chunks_path = CACHE_DIR / "chunks.pkl"

    # ── Load entirely from cache ────────────────────────────────────────────
    if not force and BM25_INDEX_PATH.exists() and cache_chunks_path.exists():
        logger.info("=== PHASE 2: LOADING CACHED INDEX ===")
        _tick("load_cache")
        from indexing.bm25_index import BM25Index
        bm25_idx = BM25Index.load(BM25_INDEX_PATH)
        with open(cache_chunks_path, "rb") as f:
            chunks = pickle.load(f)

        vec_idx = None
        if not fast and FAISS_INDEX_PATH.exists():
            from indexing.vector_index import VectorIndex
            vec_idx = VectorIndex.load(FAISS_INDEX_PATH)

        _tock("load_cache")
        return chunks, bm25_idx, vec_idx

    # ── Full build ──────────────────────────────────────────────────────────
    logger.info("=== PHASE 2: INGESTION + CHUNKING + INDEXING ===")

    # Ingestion
    _tick("ingestion")
    from ingestion.ingest import ingest_all
    pages = ingest_all(DOCS_DIR)
    _tock("ingestion")
    logger.info("  Pages extracted: %d from %d unique files",
                len(pages), len(set(p["doc_name"] for p in pages)))

    # Chunking — use page strategy as default (MUCH faster than tiktoken)
    _tick("chunking")
    from processing.chunker import chunk_all
    chunks = chunk_all(pages, strategy=strategy)
    _tock("chunking")
    logger.info("  Chunks created: %d (strategy=%s)", len(chunks), strategy)

    # BM25 Index
    _tick("bm25")
    from indexing.bm25_index import build_bm25
    bm25_idx = build_bm25(chunks)
    bm25_idx.save(BM25_INDEX_PATH)
    _tock("bm25")

    # Save chunks
    with open(cache_chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Vector Index (Gemini embeddings — skipped in --fast mode)
    vec_idx = None
    if not fast:
        _tick("vector_index")
        from indexing.vector_index import build_vector_index
        vec_idx = build_vector_index(chunks)
        vec_idx.save(FAISS_INDEX_PATH)
        _tock("vector_index")

    # NER + Document Graph
    _tick("ner_graph")
    try:
        from evidence_mapper.document_map import DocumentGraph
        graph = DocumentGraph()
        for page in pages[:200]:           # cap for speed
            citations = graph.extract_citations(page["text"], page["doc_name"])
            if citations:
                logger.debug("  %s → %d citations", page["doc_name"], len(citations))
        logger.info("  Document graph: %s", graph.summary())
    except Exception as e:
        logger.debug("NER/graph step skipped: %s", e)
    _tock("ner_graph")

    return chunks, bm25_idx, vec_idx


# ── Phase 3: Answer questions ─────────────────────────────────────────────────

def phase_query(chunks, bm25_idx, vec_idx, questions: list[str], fast: bool = False) -> list:
    logger.info("=== PHASE 3: RETRIEVAL + RERANKING + LLM ANSWERING ===")

    _tick("retrieval")

    if fast or vec_idx is None:
        # Fast mode: BM25 only (no vector search)
        logger.info("  Mode: BM25-only retrieval (fast)")
        from indexing.bm25_index import bm25_search
        id_map = {c.id: c for c in chunks}

        def _get_candidates(q):
            hits = bm25_search(bm25_idx, q, k=10)
            return [id_map[cid] for cid, _ in hits if cid in id_map]

    else:
        # Standard: Hybrid BM25 + ANN + RRF
        logger.info("  Mode: Hybrid BM25 + ANN + RRF")
        from indexing.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(bm25_idx, vec_idx, chunks)
        _get_candidates = retriever.get_candidate_chunks

    # Cross-encoder reranker
    from reranker.cross_encoder import rerank

    reranked_per_q = []
    for q in questions:
        candidates = _get_candidates(q)
        reranked   = rerank(q, candidates)
        reranked_per_q.append(reranked)

    _tock("retrieval")

    # Async LLM answering (all questions in parallel)
    _tick("llm")
    from qa.answer_generator import answer_all_async

    class _StaticRetriever:
        def get_candidate_chunks(self, q, k=10):
            idx = questions.index(q) if q in questions else 0
            return reranked_per_q[idx]

    results = asyncio.run(answer_all_async(questions, _StaticRetriever()))
    _tock("llm")

    return results


# ── Phase 4: Save submission ──────────────────────────────────────────────────

def phase_save(results) -> Path:
    logger.info("=== PHASE 4: SAVING SUBMISSION ===")
    data = [r.to_dict() for r in results if r]
    out  = Path("lucio_submission.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Submission saved → %s (%d answers)", out, len(data))
    return out


# ── Print benchmark table ─────────────────────────────────────────────────────

def _print_benchmark(total: float, n_docs: int, n_chunks: int, n_q: int) -> None:
    from config import RUNTIME_LIMIT_SECONDS
    limit = RUNTIME_LIMIT_SECONDS

    rows = [(k, v) for k, v in T.items() if isinstance(v, float)]
    rows.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print(f"  LUCIO BENCHMARK REPORT   (target: <{limit}s)")
    print("=" * 60)
    print(f"  Documents  : {n_docs}")
    print(f"  Chunks     : {n_chunks}")
    print(f"  Questions  : {n_q}")
    print("-" * 60)
    for stage, t in rows:
        bar  = "█" * min(40, int(40 * t / max(total, 0.001)))
        flag = "  ⚠ BOTTLENECK" if t > total * 0.4 else ""
        print(f"  {stage:<22} {t:6.2f}s  {bar}{flag}")
    print("-" * 60)
    ok  = "✅ WITHIN LIMIT" if total <= limit else f"⚠  OVER LIMIT (target {limit}s)"
    print(f"  {'TOTAL':<22} {total:6.2f}s  {ok}")

    if total > limit:
        print("\n  Bottleneck priority:")
        for i, (stage, t) in enumerate(rows[:3], 1):
            pct = 100 * t / max(total, 0.001)
            print(f"    {i}. {stage}: {t:.1f}s ({pct:.0f}%)")
        print("\n  Tips to speed up:")
        if "vector_index" in dict(rows) and dict(rows)["vector_index"] > 5:
            print("    • vector_indexing is slow → use --fast (BM25-only) OR let Gemini API cache build once")
        if "ingestion" in dict(rows) and dict(rows)["ingestion"] > 5:
            print("    • ingestion is slow → remove large non-essential files from documents/")
        if "llm" in dict(rows) and dict(rows)["llm"] > 10:
            print("    • LLM answering is slow → check GEMINI_API_KEY and network connectivity")

    print("=" * 60 + "\n")


# ── Question loader ───────────────────────────────────────────────────────────

def load_questions(path: str | None = None) -> list[str]:
    import pandas as pd

    # Try multiple paths (local Windows, Colab, arg override)
    candidates = [
        Path(path) if path else None,
        Path("documents/Testing Set Questions.xlsx"),
        Path("/content/lucio/documents/Testing Set Questions.xlsx"),
        Path("d:/lucio/documents/Testing Set Questions.xlsx"),
    ]

    for p in candidates:
        if p and p.exists():
            df = pd.read_excel(p)
            qs = df["Question"].dropna().tolist()
            logger.info("Loaded %d questions from %s", len(qs), p.name)
            return qs

    logger.warning("Questions file not found — using questions.py defaults")
    from questions import QUESTIONS
    return QUESTIONS


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lucio Challenge — Advanced Legal QA Pipeline")
    parser.add_argument("--questions", default=None,   help="Path to Excel questions file")
    parser.add_argument("--strategy",  default="page", choices=["token", "page"],
                        help="Chunking strategy (page is fastest)")
    parser.add_argument("--force",     action="store_true", help="Force full index rebuild")
    parser.add_argument("--fast",      action="store_true", help="BM25-only mode (no vectors, fastest)")
    parser.add_argument("--benchmark", action="store_true", help="Extended benchmark run with extra stats")
    parser.add_argument("--download",  action="store_true", help="Run corpus downloader first")
    parser.add_argument("--verbose",   action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    t_total = time.time()

    if args.download:
        phase_download()

    questions = load_questions(args.questions)

    _tick("total")
    chunks, bm25_idx, vec_idx = phase_build(
        strategy=args.strategy, force=args.force, fast=args.fast
    )
    results = phase_query(chunks, bm25_idx, vec_idx, questions, fast=args.fast)
    out_path = phase_save(results)

    total = time.time() - t_total
    T["total"] = total

    n_docs   = len(set(c.doc for c in chunks))
    n_chunks = len(chunks)

    _print_benchmark(total, n_docs, n_chunks, len(questions))

    logger.info("Done in %.1fs → %s", total, out_path)

    if args.benchmark:
        from performance.benchmark import simulate_lucio_challenge
        simulate_lucio_challenge(questions=questions[:5], strategy=args.strategy)


if __name__ == "__main__":
    main()
