"""
tests/test_extractor.py
Unit tests for per-format document extraction.
Run: pytest tests/test_extractor.py -v
"""
import os
import textwrap
from pathlib import Path

import pytest


# ── Helper: create temporary test files ───────────────────────────────────────

def _tmp_txt(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(content, encoding="utf-8")
    return p


def _tmp_html(tmp_path: Path) -> Path:
    p = tmp_path / "sample.html"
    p.write_text("<html><body><p>Hello legal world</p></body></html>", encoding="utf-8")
    return p


def _tmp_csv(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text("col1,col2\nvalue1,value2\nvalue3,value4", encoding="utf-8")
    return p


# ── TXT extractor ─────────────────────────────────────────────────────────────

def test_txt_extraction(tmp_path):
    content = "This is a legal contract about parties."
    p = _tmp_txt(tmp_path, content)
    from ingestion.extractor import extract_txt
    pages = extract_txt(p)
    assert pages, "TXT extractor returned empty list"
    assert pages[0]["doc_name"] == "sample.txt"
    assert pages[0]["page_number"] == 1
    assert "legal contract" in pages[0]["text"]


# ── HTML extractor ────────────────────────────────────────────────────────────

def test_html_extraction(tmp_path):
    p = _tmp_html(tmp_path)
    from ingestion.extractor import extract_html
    pages = extract_html(p)
    assert pages
    assert pages[0]["page_number"] == 1
    assert "Hello legal world" in pages[0]["text"]


# ── CSV extractor ─────────────────────────────────────────────────────────────

def test_csv_extraction(tmp_path):
    p = _tmp_csv(tmp_path)
    from ingestion.extractor import extract_csv
    pages = extract_csv(p)
    assert pages
    assert "value1" in pages[0]["text"]


# ── File detection ────────────────────────────────────────────────────────────

def test_file_detection_txt(tmp_path):
    p = _tmp_txt(tmp_path, "hello")
    from ingestion.file_detection import detect_type
    assert detect_type(p) == "txt"


def test_file_detection_html(tmp_path):
    p = _tmp_html(tmp_path)
    from ingestion.file_detection import detect_type
    assert detect_type(p) in ("html", "txt")   # magic may read as text/html or text/plain


def test_file_detection_csv(tmp_path):
    p = _tmp_csv(tmp_path)
    from ingestion.file_detection import detect_type
    assert detect_type(p) in ("csv", "txt")
