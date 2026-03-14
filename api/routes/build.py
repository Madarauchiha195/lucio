"""
api/routes/build.py
POST /api/build – trigger index rebuild in a background thread.
GET  /api/build/progress – SSE stream of build log lines.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

# Shared build state
_build_state = {
    "running": False,
    "done": False,
    "error": None,
    "elapsed_seconds": 0.0,
    "log": [],           # list of log message strings
}
_log_queue: asyncio.Queue = None


class _QueueLogHandler(logging.Handler):
    """Forwards log records to the SSE queue."""
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._queue = queue
        self._loop = loop

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        _build_state["log"].append(msg)
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        except Exception:
            pass


def _run_build(force: bool, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Runs build_index in a thread and streams progress to queue."""
    handler = _QueueLogHandler(queue, loop)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        import time
        t0 = time.time()
        from pipeline import build_index
        build_index(force=force)
        elapsed = time.time() - t0
        _build_state["done"] = True
        _build_state["elapsed_seconds"] = elapsed
        loop.call_soon_threadsafe(queue.put_nowait, "__DONE__")
    except Exception as exc:
        _build_state["error"] = str(exc)
        loop.call_soon_threadsafe(queue.put_nowait, f"__ERROR__{exc}")
    finally:
        _build_state["running"] = False
        root.removeHandler(handler)


@router.post("/build")
async def trigger_build(force: bool = False):
    global _log_queue
    if _build_state["running"]:
        return JSONResponse({"status": "already_running"}, status_code=409)

    _build_state.update({"running": True, "done": False, "error": None, "elapsed_seconds": 0.0, "log": []})
    loop = asyncio.get_event_loop()
    _log_queue = asyncio.Queue()

    thread = threading.Thread(
        target=_run_build,
        args=(force, _log_queue, loop),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@router.get("/build/progress")
async def build_progress():
    """SSE stream: emits log lines until build finishes."""
    async def generator() -> AsyncGenerator[dict, None]:
        # Emit cached log lines first (in case client reconnects)
        for line in _build_state["log"]:
            yield {"data": line}

        if _log_queue is None:
            yield {"data": "__DONE__"}
            return

        while True:
            try:
                line = await asyncio.wait_for(_log_queue.get(), timeout=30)
                yield {"data": line}
                if line.startswith("__DONE__") or line.startswith("__ERROR__"):
                    break
            except asyncio.TimeoutError:
                yield {"data": "__PING__"}   # keep-alive

    return EventSourceResponse(generator())
