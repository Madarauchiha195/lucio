"""
performance/benchmark.py
End-to-end benchmark harness.

simulate_lucio_challenge(corpus_dir, questions) → BenchmarkReport
  - Measures per-stage latency: ingestion, chunking, indexing, retrieval, LLM
  - If total > RUNTIME_LIMIT_SECONDS: prints prioritised bottleneck list
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from config import RUNTIME_LIMIT_SECONDS, DOCS_DIR

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    timings: Dict[str, float] = field(default_factory=dict)
    total:   float = 0.0
    n_docs:  int   = 0
    n_chunks: int  = 0
    n_questions: int = 0
    exceeded_limit: bool = False

    def log(self) -> None:
        print(f"\n{'='*60}")
        print(f"  BENCHMARK REPORT  (limit: {RUNTIME_LIMIT_SECONDS}s)")
        print(f"{'='*60}")
        for stage, t in self.timings.items():
            flag = " ⚠" if t > self.total * 0.3 else ""
            print(f"  {stage:<25} {t:7.2f}s{flag}")
        print(f"  {'TOTAL':<25} {self.total:7.2f}s")
        if self.exceeded_limit:
            print(f"\n  ⚠  EXCEEDED {RUNTIME_LIMIT_SECONDS}s LIMIT!")
            # Ranked bottlenecks
            ranked = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
            print("\n  Bottleneck priority list:")
            for i, (stage, t) in enumerate(ranked, 1):
                pct = 100.0 * t / max(self.total, 0.001)
                print(f"    {i}. {stage}: {t:.2f}s ({pct:.1f}%)")
        print(f"{'='*60}\n")


def simulate_lucio_challenge(
    corpus_dir: Path = DOCS_DIR,
    questions: List[str] | None = None,
    strategy: str = "token",
) -> BenchmarkReport:
    """
    Full end-to-end benchmark run.
    corpus_dir: folder containing documents.
    questions : list of question strings (uses default 5 if None).
    strategy  : "token" | "page" chunking strategy.
    """
    if questions is None:
        questions = [
            "What are the key obligations of the parties?",
            "What are the termination clauses?",
            "What is the governing law?",
            "What are the confidentiality terms?",
            "What are the penalty clauses?",
        ]

    report  = BenchmarkReport(n_questions=len(questions))
    timings: Dict[str, float] = {}
    t_total = time.time()

    # ── 1. Ingestion ──────────────────────────────────────────────────────────
    t = time.time()
    from ingestion.ingest import ingest_all
    pages = ingest_all(corpus_dir, workers=6)
    timings["ingestion"]  = time.time() - t
    report.n_docs         = len(set(p["doc_name"] for p in pages))

    # ── 2. Chunking ───────────────────────────────────────────────────────────
    t = time.time()
    from processing.chunker import chunk_all
    chunks = chunk_all(pages, strategy=strategy)
    timings["chunking"]  = time.time() - t
    report.n_chunks      = len(chunks)

    # ── 3. Indexing ───────────────────────────────────────────────────────────
    t = time.time()
    from indexing.bm25_index import build_bm25
    bm25_idx = build_bm25(chunks)
    timings["bm25_indexing"] = time.time() - t

    t = time.time()
    from indexing.vector_index import build_vector_index
    vec_idx = build_vector_index(chunks)
    timings["vector_indexing"] = time.time() - t

    # ── 4. Retrieval + reranking ──────────────────────────────────────────────
    t = time.time()
    from indexing.hybrid_retriever import HybridRetriever
    from reranker.cross_encoder import rerank
    retriever = HybridRetriever(bm25_idx, vec_idx, chunks)
    candidates_per_q = [retriever.get_candidate_chunks(q) for q in questions]
    reranked_per_q   = [rerank(q, cands) for q, cands in zip(questions, candidates_per_q)]
    timings["retrieval_rerank"] = time.time() - t

    # ── 5. LLM answering ──────────────────────────────────────────────────────
    t = time.time()
    from qa.answer_generator import answer_question
    results = [answer_question(i+1, q, cands)
               for i, (q, cands) in enumerate(zip(questions, reranked_per_q))]
    timings["llm_answering"] = time.time() - t

    # ── Summary ───────────────────────────────────────────────────────────────
    report.total          = time.time() - t_total
    report.timings        = timings
    report.exceeded_limit = report.total > RUNTIME_LIMIT_SECONDS
    report.log()

    return report
