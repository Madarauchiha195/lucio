"""
main.py – CLI entry-point for the Document QA system.

Usage:
    python main.py --build              # ingest & index all documents
    python main.py --build --force      # re-index even if indexes exist
    python main.py --query              # answer all 15 questions
    python main.py --query --q "text"   # answer a single ad-hoc question
    python main.py --build --query      # build then query in one shot
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time

# Force UTF-8 output on Windows to avoid UnicodeEncodeError in logger
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler])
    # Silence noisy third-party loggers
    for name in ("transformers", "sentence_transformers", "faiss", "httpx", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Document QA System - hybrid BM25 + FAISS + GPT-4"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Ingest documents and build/update the search indexes."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force full rebuild even if indexes already exist."
    )
    parser.add_argument(
        "--query", action="store_true",
        help="Answer the 15 fixed questions (or a single --q question)."
    )
    parser.add_argument(
        "--q", metavar="QUESTION", default=None,
        help="Ad-hoc question string. Overrides the fixed question list."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging."
    )
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.print_help()
        return 1

    _setup_logging(args.verbose)
    logger = logging.getLogger("main")

    t_start = time.time()

    # Build phase
    if args.build:
        from pipeline import build_index
        logger.info("Starting index build ...")
        build_index(force=args.force)

    # Query phase
    if args.query:
        from pipeline import run_questions

        if args.q:
            questions = [args.q]
        else:
            from questions import QUESTIONS
            questions = QUESTIONS

        logger.info("Starting query phase (%d question(s)) ...", len(questions))
        run_questions(questions)

        from config import OUTPUT_DIR
        logger.info("Results written to: %s", OUTPUT_DIR)

    total = time.time() - t_start
    logger.info("Done in %.1f s.", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
