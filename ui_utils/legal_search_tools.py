"""
ui_utils/legal_search_tools.py
Advanced keyword search utilities for legal documents.

Functions:
  boolean_search(index, query)             – AND/OR/NOT token logic
  proximity_search(index, t1, t2, dist)    – within-N-tokens match
  clause_extractor(text)                   – detect/extract legal headings & clauses
  citation_extractor(text)                 – extract case/statute citations via regex
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple


# ── Boolean search ────────────────────────────────────────────────────────────

def boolean_search(bm25_index, query: str) -> List[str]:
    """
    Wraps the BM25 index boolean_search.
    Supports AND / OR / NOT.  E.g. "merger AND acquisition NOT tech".
    Returns a list of matching chunk_ids.
    """
    from indexing.bm25_index import boolean_search as _bs
    return _bs(bm25_index, query)


# ── Proximity search ──────────────────────────────────────────────────────────

def proximity_search(bm25_index, term1: str, term2: str, distance: int = 10) -> List[str]:
    """
    Wraps the BM25 index proximity_search.
    Returns chunk_ids where term1 and term2 appear within *distance* tokens.
    """
    from indexing.bm25_index import proximity_search as _ps
    return _ps(bm25_index, term1, term2, distance)


# ── Clause extractor ──────────────────────────────────────────────────────────

_HEADING_PATTERNS = [
    # ALL CAPS heading: ARTICLE XII — TERMINATION
    re.compile(r"^([A-Z][A-Z\s\d\.]+[:—\-]?\s*[A-Z\s]*)$", re.MULTILINE),
    # Numbered heading: 12.3 Payment Terms
    re.compile(r"^(\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z\s]{3,})$", re.MULTILINE),
    # Section heading: Section 4: Background
    re.compile(r"^(Section\s+\d+[a-z]?[:.\s]+[A-Z][A-Za-z\s]{2,})$", re.MULTILINE | re.IGNORECASE),
]


def clause_extractor(text: str) -> List[Dict]:
    """
    Detect legal headings / clauses in *text* using heading-detection regex patterns.
    Returns a list of {heading, start, end, snippet} dicts.
    """
    found_positions = []
    for pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            found_positions.append((m.start(), m.group(0).strip()))

    # Sort by position and de-duplicate nearby matches
    found_positions.sort(key=lambda x: x[0])

    clauses = []
    for i, (start, heading) in enumerate(found_positions):
        end = found_positions[i + 1][0] if i + 1 < len(found_positions) else len(text)
        snippet = text[start:end][:500].strip()
        clauses.append({"heading": heading, "start": start, "end": end, "snippet": snippet})

    return clauses


# ── Citation extractor ────────────────────────────────────────────────────────

_CITATION_RE = [
    re.compile(r"([A-Z][a-z][\w\s]+)\s+v\.?\s+([\w\s]+),\s+\d+\s+[\w\.]+\s+\d+"),   # case
    re.compile(r"\d+\s+U\.S\.C\.?\s+§+\s*\d+[a-z]?"),                                # statute
    re.compile(r"§+\s*\d+(?:\.\d+)*(?:\([a-z]\))?"),                                  # section ref
    re.compile(r"\bRule\s+\d+(?:\.\d+)?(?:\([a-z]\))?\b"),                            # court rule
]


def citation_extractor(text: str) -> List[str]:
    """
    Extract all legal citations from *text* using regex patterns.
    Returns deduplicated list of citation strings.
    """
    results = set()
    for pattern in _CITATION_RE:
        for match in pattern.finditer(text):
            results.add(match.group(0).strip())
    return sorted(results)
