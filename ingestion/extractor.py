"""
ingestion/extractor.py
Specialized extractors for each file format.
Output: list of Page dicts:
    {doc_name, page_number, offset, text, tables: list[str]}
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

Page = Dict[str, Any]  # typed alias


def _clean(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF ───────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> List[Page]:
    """
    Primary: PyMuPDF page-level text + offset.
    Table layer: pdfplumber on same page.
    Fallback: if page text is empty → pytesseract OCR.
    """
    pages: List[Page] = []
    try:
        import fitz  # PyMuPDF
        import pdfplumber

        plumber_doc = pdfplumber.open(str(path))
        fitz_doc = fitz.open(str(path))

        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            # Full text with character offsets
            blocks = fitz_page.get_text("blocks", sort=True)
            raw_texts = [b[4] for b in blocks if b[4].strip()]
            text = _clean(" ".join(raw_texts))

            # OCR fallback for scanned pages
            if not text:
                try:
                    import pytesseract
                    from PIL import Image
                    pix = fitz_page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = _clean(pytesseract.image_to_string(img))
                except Exception as e:
                    logger.debug("OCR fallback failed p%d: %s", page_num + 1, e)

            # Table extraction
            tables: List[str] = []
            try:
                pl_page = plumber_doc.pages[page_num]
                for table in pl_page.extract_tables():
                    rows = ["\t".join(str(c) for c in row if c) for row in table if any(row)]
                    if rows:
                        tables.append("\n".join(rows))
            except Exception:
                pass

            if text or tables:
                pages.append({
                    "doc_name":    path.name,
                    "page_number": page_num + 1,
                    "offset":      page_num,   # byte offset approximation
                    "text":        text,
                    "tables":      tables,
                })

        fitz_doc.close()
        plumber_doc.close()
    except Exception as exc:
        logger.warning("PDF extract error [%s]: %s", path.name, exc)
    return pages


# ── DOCX ──────────────────────────────────────────────────────────────────────

def extract_docx(path: Path) -> List[Page]:
    PARAS_PER_PAGE = 30
    pages: List[Page] = []
    try:
        from docx import Document
        document = Document(str(path))
        bucket: List[str] = []
        page_num = 1
        for para in document.paragraphs:
            t = para.text.strip()
            if t:
                bucket.append(t)
            if len(bucket) >= PARAS_PER_PAGE:
                pages.append({
                    "doc_name":    path.name,
                    "page_number": page_num,
                    "offset":      page_num - 1,
                    "text":        _clean("\n".join(bucket)),
                    "tables":      [],
                })
                bucket = []
                page_num += 1
        if bucket:
            pages.append({
                "doc_name":    path.name,
                "page_number": page_num,
                "offset":      page_num - 1,
                "text":        _clean("\n".join(bucket)),
                "tables":      [],
            })
    except Exception as exc:
        logger.warning("DOCX extract error [%s]: %s", path.name, exc)
    return pages


# ── HTML ──────────────────────────────────────────────────────────────────────

def extract_html(path: Path) -> List[Page]:
    pages: List[Page] = []
    try:
        from bs4 import BeautifulSoup
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = _clean(soup.get_text(separator="\n"))
        if text:
            pages.append({"doc_name": path.name, "page_number": 1, "offset": 0, "text": text, "tables": []})
    except Exception as exc:
        logger.warning("HTML extract error [%s]: %s", path.name, exc)
    return pages


# ── TXT ───────────────────────────────────────────────────────────────────────

def extract_txt(path: Path) -> List[Page]:
    pages: List[Page] = []
    try:
        text = _clean(path.read_text(encoding="utf-8", errors="replace"))
        if text:
            pages.append({"doc_name": path.name, "page_number": 1, "offset": 0, "text": text, "tables": []})
    except Exception as exc:
        logger.warning("TXT extract error [%s]: %s", path.name, exc)
    return pages


# ── CSV / XLSX ────────────────────────────────────────────────────────────────

def extract_csv(path: Path) -> List[Page]:
    pages: List[Page] = []
    try:
        import pandas as pd
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        text = _clean(df.to_string(index=False))
        if text:
            pages.append({"doc_name": path.name, "page_number": 1, "offset": 0,
                          "text": text, "tables": [text]})
    except Exception as exc:
        logger.warning("CSV/XLSX extract error [%s]: %s", path.name, exc)
    return pages


# ── Dispatcher ────────────────────────────────────────────────────────────────

_EXTRACTORS = {
    "pdf":  extract_pdf,
    "docx": extract_docx,
    "html": extract_html,
    "txt":  extract_txt,
    "csv":  extract_csv,
    "xlsx": extract_csv,
}


def extract(path: Path, file_type: str | None = None) -> List[Page]:
    """
    Extract pages from *path*. If *file_type* is None it is auto-detected.
    Returns list[Page] = [{doc_name, page_number, offset, text, tables}]
    """
    if file_type is None:
        from ingestion.file_detection import detect_type
        file_type = detect_type(path)

    extractor = _EXTRACTORS.get(file_type)
    if extractor is None:
        logger.debug("No extractor for type '%s': %s", file_type, path.name)
        return []

    try:
        return extractor(path)
    except Exception as exc:
        logger.error("Extractor crashed [%s/%s]: %s", file_type, path.name, exc)
        return []
