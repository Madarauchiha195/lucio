"""
qa/answer_generator.py
Builds prompts for the legal QA task and generates structured answers.

Output schema per question:
    {question_id, answer, document, page, evidence}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from config import LLM_CONTEXT_PASSAGES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a legal document analyst. You will be given a set of excerpts from legal documents, \
each labelled with its source file and page number. Your job is to answer the user's question \
using ONLY the information contained in those excerpts.

Rules:
1. Answer ONLY from the provided excerpts. Do not use external knowledge.
2. For every specific fact or claim, include an inline citation: (DocumentName, p.PageNumber).
3. If the excerpts do not contain enough information, say so explicitly.
4. Be concise and precise. Restrict your answer to the key facts.

After your narrative answer, output a JSON block (fenced with ```json ... ```) with:
{
  "sources": [
    {
      "document": "FileName.pdf",
      "page": 3,
      "evidence": "Exact excerpt text you relied on"
    }
  ]
}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnswerResult:
    question_id: int
    question:    str
    answer:      str
    sources:     List[Dict[str, Any]] = field(default_factory=list)
    raw:         str = field(default="", repr=False)

    def to_dict(self) -> dict:
        doc      = self.sources[0].get("document", "") if self.sources else ""
        page     = self.sources[0].get("page", 0)     if self.sources else 0
        evidence = self.sources[0].get("evidence", "") if self.sources else ""
        return {
            "question_id": self.question_id,
            "answer":      self.answer,
            "document":    doc,
            "page":        page,
            "evidence":    evidence,
        }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_context(chunks) -> str:
    """Format the top-N passages into the LLM context block."""
    lines = []
    for i, chunk in enumerate(chunks[:LLM_CONTEXT_PASSAGES], start=1):
        lines.append(f"[EXCERPT {i}]")
        lines.append(f"Source: {chunk.doc_name} | Page: {chunk.page_number}")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


def _parse_sources(raw: str) -> List[Dict[str, Any]]:
    """Extract the JSON sources block from the LLM response."""
    pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    match   = pattern.search(raw)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("sources", [])
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: inline citation pattern (Document, p.N)
    sources = []
    for doc, page in re.findall(r"\(([^,]+),\s*p\.?(\d+)\)", raw):
        sources.append({"document": doc.strip(), "page": int(page), "evidence": ""})
    return sources


def _strip_json_block(text: str) -> str:
    return re.sub(r"```json.*?```", "", text, flags=re.DOTALL).strip()


# ── Sync helper ───────────────────────────────────────────────────────────────

def answer_question(question_id: int, question: str, chunks) -> AnswerResult:
    """Synchronous wrapper — used by the old pipeline.py path."""
    from qa.llm_client import get_adapter
    adapter     = get_adapter()
    context     = _build_context(chunks)
    user_prompt = f"DOCUMENT EXCERPTS:\n\n{context}\n\nQUESTION: {question}"
    raw         = adapter.generate(SYSTEM_PROMPT, user_prompt)
    sources     = _parse_sources(raw)
    answer      = _strip_json_block(raw)
    return AnswerResult(question_id=question_id, question=question,
                        answer=answer, sources=sources, raw=raw)


# ── Async batch answerer ──────────────────────────────────────────────────────

async def answer_all_async(questions: List[str], retriever) -> List[AnswerResult]:
    """
    Answer all questions concurrently using the async LLM client.
    retriever must expose get_candidate_chunks(query, k).
    """
    from qa.llm_client import batch_generate

    contexts = [_build_context(retriever.get_candidate_chunks(q)) for q in questions]
    prompts  = [
        (SYSTEM_PROMPT, f"DOCUMENT EXCERPTS:\n\n{ctx}\n\nQUESTION: {q}")
        for q, ctx in zip(questions, contexts)
    ]

    logger.info("Batch answering %d questions asynchronously …", len(questions))
    raw_answers = await batch_generate(prompts)

    results = []
    for idx, (q, raw) in enumerate(zip(questions, raw_answers), start=1):
        sources = _parse_sources(raw)
        answer  = _strip_json_block(raw)
        results.append(AnswerResult(
            question_id=idx, question=q,
            answer=answer, sources=sources, raw=raw,
        ))
    return results
