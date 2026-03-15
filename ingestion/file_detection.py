"""
ingestion/file_detection.py
Detects the type of a document file and routes it to the correct parser.
Uses python-magic for MIME-type detection with a suffix-based fallback.
"""
from __future__ import annotations

from pathlib import Path

# MIME-type → parser tag
_MIME_MAP = {
    "application/pdf":                                       "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword":                                    "docx",
    "text/html":                                             "html",
    "text/plain":                                            "txt",
    "text/csv":                                              "csv",
    "application/vnd.ms-excel":                              "csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}

_EXT_MAP = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".doc":  "docx",
    ".html": "html",
    ".htm":  "html",
    ".txt":  "txt",
    ".csv":  "csv",
    ".xlsx": "xlsx",
}


def detect_type(path: Path) -> str:
    """
    Return a parser tag for *path*: 'pdf' | 'docx' | 'html' | 'txt' | 'csv' | 'xlsx' | 'unknown'.

    Tries python-magic first (reliable for mis-named files), falls back to suffix.
    """
    # 1. Try python-magic
    try:
        import magic
        mime = magic.from_file(str(path), mime=True)
        tag = _MIME_MAP.get(mime)
        if tag:
            return tag
    except Exception:
        pass  # magic not available or file unreadable — fall through

    # 2. Suffix-based fallback
    return _EXT_MAP.get(path.suffix.lower(), "unknown")
