"""
llm_answer.py
Build prompts and call OpenAI GPT-4 to answer questions with citations.

The LLM is instructed to:
  1. Answer ONLY from the provided excerpts.
  2. Include inline citations in the form: (DocName, p.N)
  3. Return JSON with keys: "answer" and "sources".

Public API:
    answer_question(question, chunks) -> AnswerResult
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from config import GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from chunker import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a legal document analyst. You will be given a set of excerpts from legal documents, \
each labelled with its source file and page number. Your job is to answer the user's question \
using ONLY the information contained in those excerpts.

Rules:
1. Answer ONLY from the provided excerpts. Do not use external knowledge.
2. For every specific fact or claim in your answer, include an inline citation in this exact format: \
(DocumentName, p.PageNumber). Example: The contract is governed by New York law (ServiceAgreement.pdf, p.3).
3. If the excerpts do not contain enough information to answer the question, say so explicitly.
4. Be concise and precise.

After your answer, output a JSON block (fenced with ```json ... ```) with an array of sources used:
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


def _format_context(chunks: List[Chunk]) -> str:
    """Format chunks into a numbered context block for the prompt."""
    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[EXCERPT {i}]")
        lines.append(f"Source: {chunk.doc_name} | Page: {chunk.page_number}")
        lines.append(chunk.text)
        lines.append("")          # blank line between excerpts
    return "\n".join(lines)


def _extract_sources_json(text: str) -> List[Dict[str, Any]]:
    """
    Try to extract the JSON sources block from the LLM response.
    Returns an empty list if the block is missing or malformed.
    """
    pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
        return data.get("sources", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _strip_json_block(text: str) -> str:
    """Remove the trailing JSON sources block from the answer text."""
    return re.sub(r"```json.*?```", "", text, flags=re.DOTALL).strip()


@dataclass
class AnswerResult:
    question_id: int
    question: str
    answer: str
    sources: List[Dict[str, Any]]               # [{"document": ..., "page": ..., "evidence": ...}]
    raw_response: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        # Challenge format: if multiple sources, just map the first one for simplicity, or return list
        # For submission, if they expect a flat dict:
        doc = self.sources[0].get("document", "") if self.sources else ""
        page = self.sources[0].get("page", 0) if self.sources else 0
        evidence = self.sources[0].get("evidence", "") if self.sources else ""

        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "document": doc,
            "page": page,
            "evidence": evidence,
        }

    def pretty(self) -> str:
        lines = [
            f"Q: {self.question}",
            f"A: {self.answer}",
            "Sources:",
        ]
        for s in self.sources:
            lines.append(f"  • {s.get('document', '?')} — page {s.get('page', '?')}")
        return "\n".join(lines)


def answer_question(question_id: int, question: str, chunks: List[Chunk]) -> AnswerResult:
    """
    Send the question and retrieved chunks to GPT-4 and parse the response.
    Returns an AnswerResult with the answer text and structured citations.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai package not installed. Run: pip install google-genai")

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    context_text = _format_context(chunks)
    user_message = f"DOCUMENT EXCERPTS:\n\n{context_text}\n\nQUESTION: {question}"

    logger.debug("Calling %s for question: %r …", LLM_MODEL, question[:60])

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=[
            types.Content(role="user", parts=[
                types.Part.from_text(text=user_message)
            ])
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    )

    raw = response.text or ""
    sources = _extract_sources_json(raw)
    answer_text = _strip_json_block(raw)

    # Fallback: if the model didn't produce a JSON block, extract citations from text
    if not sources:
        inline_pattern = re.compile(r"\(([^,]+),\s*p\.?(\d+)\)")
        for doc, page in inline_pattern.findall(answer_text):
            sources.append({"document": doc.strip(), "page": int(page), "evidence": ""})

    logger.info("LLM answered. %d source(s) cited.", len(sources))
    return AnswerResult(
        question_id=question_id,
        question=question,
        answer=answer_text,
        sources=sources,
        raw_response=raw,
    )
