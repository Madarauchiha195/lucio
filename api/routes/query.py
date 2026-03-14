"""
api/routes/query.py
POST /api/query – answer a question, streaming tokens via SSE.
POST /api/query/batch – answer all 15 fixed questions.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import BM25_INDEX_PATH, FAISS_INDEX_PATH

router = APIRouter()

# Lazily loaded singletons (loaded once on first query)
_retriever = None
_retriever_lock = threading.Lock()


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    with _retriever_lock:
        if _retriever is not None:
            return _retriever
        if not BM25_INDEX_PATH.exists() or not FAISS_INDEX_PATH.exists():
            raise RuntimeError("Index not built. Run POST /api/build first.")
        from index_bm25 import load_bm25
        from index_faiss import load_faiss, _load_model
        from retrieval import HybridRetriever

        bm25  = load_bm25(BM25_INDEX_PATH)
        faiss = load_faiss(FAISS_INDEX_PATH)
        model = _load_model(faiss.model_name)
        _retriever = HybridRetriever(bm25, faiss, model)
    return _retriever


class QueryRequest(BaseModel):
    question: str


def _answer_in_thread(question: str, result_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Run retrieval + LLM in a thread, push result to queue."""
    try:
        retriever = _get_retriever()
        chunks = retriever.search(question)
        from llm_answer import answer_question
        result = answer_question(0, question, chunks)
        loop.call_soon_threadsafe(result_queue.put_nowait, result)
    except Exception as exc:
        loop.call_soon_threadsafe(result_queue.put_nowait, exc)


@router.post("/query")
async def query_stream(req: QueryRequest):
    """
    Stream the answer as SSE events:
      - data: {"type": "chunk", "doc": ..., "page": ..., "text": ...}  per retrieved chunk
      - data: {"type": "answer", "text": ...}                           full answer
      - data: {"type": "sources", "sources": [...]}                     citation list
      - data: {"type": "done"}
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    loop = asyncio.get_event_loop()
    result_queue: asyncio.Queue = asyncio.Queue()

    thread = threading.Thread(
        target=_answer_in_thread,
        args=(question, result_queue, loop),
        daemon=True,
    )
    thread.start()

    async def generator() -> AsyncGenerator[dict, None]:
        # First: emit the retrieved chunks so UI can show them while LLM processes
        try:
            retriever = _get_retriever()
            chunks = retriever.search(question)
            for c in chunks:
                payload = json.dumps({
                    "type": "chunk",
                    "doc": c.doc_name,
                    "page": c.page_number,
                    "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
                })
                yield {"data": payload}
        except Exception as exc:
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}
            return

        # Then: wait for LLM answer
        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=60)
        except asyncio.TimeoutError:
            yield {"data": json.dumps({"type": "error", "message": "LLM timed out"})}
            return

        if isinstance(result, Exception):
            yield {"data": json.dumps({"type": "error", "message": str(result)})}
            return

        yield {"data": json.dumps({"type": "answer", "text": result.answer})}
        yield {"data": json.dumps({"type": "sources", "sources": result.sources})}
        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(generator())


@router.post("/query/batch")
async def query_batch():
    """Answer all 15 fixed questions (non-streaming, returns JSON)."""
    from questions import QUESTIONS
    from pipeline import run_questions

    loop = asyncio.get_event_loop()
    result_queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            results = run_questions(QUESTIONS)
            loop.call_soon_threadsafe(result_queue.put_nowait, results)
        except Exception as exc:
            loop.call_soon_threadsafe(result_queue.put_nowait, exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    results = await asyncio.wait_for(result_queue.get(), timeout=120)
    if isinstance(results, Exception):
        raise HTTPException(status_code=500, detail=str(results))

    return {"results": [r.to_dict() for r in results if r]}
