"""
indexing/bm25_index.py
BM25 index with stopword filtering AND boolean / proximity query support.
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from config import BM25_TOPK

logger = logging.getLogger(__name__)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _get_stopwords() -> set:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        import nltk, ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return set()


def tokenize(text: str, stop_words: set | None = None) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    return tokens


# ── Index dataclass ───────────────────────────────────────────────────────────

@dataclass
class BM25Index:
    bm25: object          # rank_bm25.BM25Okapi
    chunk_ids: List[str]
    tokenized_corpus: List[List[str]]
    stop_words: set

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Build ─────────────────────────────────────────────────────────────────────

def build_bm25(chunks) -> BM25Index:
    """Build a BM25 index from a list of Chunk objects."""
    from rank_bm25 import BM25Okapi
    stop_words = _get_stopwords()
    tokenized  = [tokenize(c.text, stop_words) for c in chunks]
    bm25       = BM25Okapi(tokenized)
    return BM25Index(
        bm25=bm25,
        chunk_ids=[c.id for c in chunks],
        tokenized_corpus=tokenized,
        stop_words=stop_words,
    )


# ── Basic search ──────────────────────────────────────────────────────────────

def bm25_search(index: BM25Index, query: str, k: int = BM25_TOPK) -> List[Tuple[str, float]]:
    """Returns [(chunk_id, score)] sorted descending."""
    tokens = tokenize(query, index.stop_words)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    ranked = sorted(zip(index.chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [(cid, sc) for cid, sc in ranked[:k] if sc > 0]


# ── Boolean search ────────────────────────────────────────────────────────────

def boolean_search(index: BM25Index, query: str) -> List[str]:
    """
    Very lightweight boolean parser supporting AND / OR / NOT clauses.
    E.g. "merger AND acquisition NOT technology"
    Returns chunk_ids that satisfy the boolean expression.
    """
    # Tokenise corpus into sets once
    corpus_sets = [set(toks) for toks in index.tokenized_corpus]

    # Split into AND groups
    query = query.strip()
    # Tokenise at the operator level
    parts = re.split(r"\bAND\b", query, flags=re.IGNORECASE)
    # Each part may contain OR / NOT
    required_sets = []
    for part in parts:
        or_parts = re.split(r"\bOR\b", part, flags=re.IGNORECASE)
        or_groups = []
        for op in or_parts:
            not_split = re.split(r"\bNOT\b", op, flags=re.IGNORECASE)
            must     = set(tokenize(not_split[0], index.stop_words))
            must_not = set(tokenize(not_split[1], index.stop_words)) if len(not_split) > 1 else set()
            or_groups.append((must, must_not))
        required_sets.append(or_groups)

    results: List[str] = []
    for i, (cid, doc_tokens) in enumerate(zip(index.chunk_ids, corpus_sets)):
        match = True
        for or_groups in required_sets:
            or_match = False
            for must, must_not in or_groups:
                if must.issubset(doc_tokens) and doc_tokens.isdisjoint(must_not):
                    or_match = True
                    break
            if not or_match:
                match = False
                break
        if match:
            results.append(cid)
    return results


# ── Proximity search ──────────────────────────────────────────────────────────

def proximity_search(index: BM25Index, term1: str, term2: str, distance: int = 10) -> List[str]:
    """
    Returns chunk_ids where term1 and term2 appear within *distance* tokens.
    """
    t1 = tokenize(term1, index.stop_words)
    t2 = tokenize(term2, index.stop_words)
    if not t1 or not t2:
        return []
    w1, w2 = t1[0], t2[0]

    results: List[str] = []
    for cid, tokens in zip(index.chunk_ids, index.tokenized_corpus):
        positions1 = [i for i, t in enumerate(tokens) if t == w1]
        positions2 = [i for i, t in enumerate(tokens) if t == w2]
        found = any(abs(p1 - p2) <= distance for p1 in positions1 for p2 in positions2)
        if found:
            results.append(cid)
    return results
