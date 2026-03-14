"""
api/routes/status.py
GET /api/status – reports whether indexes are built and ready.
"""
from pathlib import Path
from fastapi import APIRouter
from config import BM25_INDEX_PATH, FAISS_INDEX_PATH, CHUNKS_CACHE_PATH

router = APIRouter()


@router.get("/status")
def get_status():
    bm25_ready  = BM25_INDEX_PATH.exists()
    faiss_ready = FAISS_INDEX_PATH.exists()
    chunks_ready = CHUNKS_CACHE_PATH.exists()
    ready = bm25_ready and faiss_ready and chunks_ready

    doc_count = 0
    chunk_count = 0
    build_time = 0.0
    
    if chunks_ready:
        try:
            import pickle
            with open(CHUNKS_CACHE_PATH, "rb") as f:
                chunks = pickle.load(f)
            chunk_count = len(chunks)
            doc_count = len({c.doc_name for c in chunks})
            
            # Try to get build time from FAISS meta if saved
            meta_path = FAISS_INDEX_PATH.with_suffix(".meta.pkl")
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                    build_time = meta.get("build_time", 0.0)
        except Exception:
            pass

    # Fallback to in-memory build state if just built
    from api.routes.build import _build_state
    if _build_state.get("elapsed_seconds"):
        build_time = _build_state["elapsed_seconds"]

    return {
        "ready": ready,
        "bm25_ready": bm25_ready,
        "faiss_ready": faiss_ready,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "build_duration_seconds": round(build_time, 1) if build_time else None,
    }
