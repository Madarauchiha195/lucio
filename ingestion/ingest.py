"""
ingestion/ingest.py
Parallel ingestion orchestrator.
Calls file_detection + extractor for every file in a directory.
Returns a flat list of Page dicts sorted by (doc_name, page_number).
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from config import DOCS_DIR, INGESTION_WORKERS, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)
Page = Dict[str, Any]


def _dispatch(path_str: str) -> List[Page]:
    """Top-level worker — must be module-level for pickling."""
    from ingestion.file_detection import detect_type
    from ingestion.extractor import extract
    path = Path(path_str)
    return extract(path, detect_type(path))


def ingest_all(docs_dir: Path = DOCS_DIR, workers: int = INGESTION_WORKERS) -> List[Page]:
    """
    Ingest all supported documents in *docs_dir* using parallel workers.
    Returns list[Page] = [{doc_name, page_number, offset, text, tables}]
    """
    files = [
        str(f) for f in sorted(docs_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        logger.warning("No supported documents found in %s", docs_dir)
        return []

    logger.info("Ingesting %d files with %d workers …", len(files), workers)
    all_pages: List[Page] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_dispatch, f): f for f in files}
        for future in as_completed(futures):
            fname = Path(futures[future]).name
            try:
                pages = future.result()
                logger.debug("  %s → %d page(s)", fname, len(pages))
                all_pages.extend(pages)
            except Exception as exc:
                logger.error("  FAILED [%s]: %s", fname, exc)

    all_pages.sort(key=lambda p: (p["doc_name"], p["page_number"]))
    logger.info("Ingestion complete: %d pages from %d files.", len(all_pages), len(files))
    return all_pages
