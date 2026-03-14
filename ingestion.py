"""
ingestion.py
Parallel document ingestion for PDF, DOCX, HTML, and plain-text files.

Each parser returns a list of Page dicts:
    {"doc_name": str, "page_number": int, "text": str}

Entry point:
    ingest_all(docs_dir) -> List[Page]
"""
from __future__ import annotations

import re
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from config import DOCS_DIR, INGESTION_WORKERS, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

Page = Dict[str, Any]  # {"doc_name": str, "page_number": int, "text": str}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip excessive whitespace and control characters."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Per-format parsers ─────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> List[Page]:
    """Parse a PDF using PyMuPDF, one Page per physical page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    pages: List[Page] = []
    try:
        doc = fitz.open(str(path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned = _clean(text)
            if cleaned:
                pages.append({
                    "doc_name": path.name,
                    "page_number": page_num + 1,
                    "text": cleaned,
                })
        doc.close()
    except Exception as exc:
        logger.warning("PDF parse error [%s]: %s", path.name, exc)
    return pages


def _parse_docx(path: Path) -> List[Page]:
    """
    Parse a DOCX using python-docx.
    Word files have no hard page boundaries; we approximate pages by
    grouping every PARAGRAPHS_PER_PAGE paragraphs.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    PARAGRAPHS_PER_PAGE = 30          # heuristic – tune if needed
    pages: List[Page] = []
    try:
        document = Document(str(path))
        bucket: List[str] = []
        page_num = 1
        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                bucket.append(text)
            if len(bucket) >= PARAGRAPHS_PER_PAGE:
                pages.append({
                    "doc_name": path.name,
                    "page_number": page_num,
                    "text": _clean("\n".join(bucket)),
                })
                bucket = []
                page_num += 1
        if bucket:                     # final partial page
            pages.append({
                "doc_name": path.name,
                "page_number": page_num,
                "text": _clean("\n".join(bucket)),
            })
    except Exception as exc:
        logger.warning("DOCX parse error [%s]: %s", path.name, exc)
    return pages


def _parse_html(path: Path) -> List[Page]:
    """Parse an HTML file using BeautifulSoup, returning a single 'page'."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

    pages: List[Page] = []
    try:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        cleaned = _clean(text)
        if cleaned:
            pages.append({
                "doc_name": path.name,
                "page_number": 1,
                "text": cleaned,
            })
    except Exception as exc:
        logger.warning("HTML parse error [%s]: %s", path.name, exc)
    return pages


def _parse_txt(path: Path) -> List[Page]:
    """Parse a plain-text file; returns a single page."""
    pages: List[Page] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = _clean(text)
        if cleaned:
            pages.append({
                "doc_name": path.name,
                "page_number": 1,
                "text": cleaned,
            })
    except Exception as exc:
        logger.warning("TXT parse error [%s]: %s", path.name, exc)
    return pages


# ── Dispatcher (must be module-level for pickling) ────────────────────────────

def _dispatch(path_str: str) -> List[Page]:
    """Top-level worker function: dispatch to the correct parser."""
    path = Path(path_str)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    elif ext == ".docx":
        return _parse_docx(path)
    elif ext in {".html", ".htm"}:
        return _parse_html(path)
    elif ext == ".txt":
        return _parse_txt(path)
    else:
        logger.debug("Skipping unsupported file: %s", path.name)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_all(docs_dir: Path = DOCS_DIR, workers: int = INGESTION_WORKERS) -> List[Page]:
    """
    Ingest all documents in *docs_dir* using a ProcessPoolExecutor.

    Returns a flat list of Page dicts, sorted by (doc_name, page_number).
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pages = ingest_all()
    for p in pages[:5]:
        print(f"[{p['doc_name']} | p{p['page_number']}] {p['text'][:80]!r}")
