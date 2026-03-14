"""
api/routes/documents.py
GET /api/documents – list all files in the documents/ folder.
"""
from fastapi import APIRouter
from config import DOCS_DIR, SUPPORTED_EXTENSIONS

router = APIRouter()


@router.get("/documents")
def list_documents():
    files = []
    if DOCS_DIR.exists():
        for f in sorted(DOCS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append({
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "type": f.suffix.lower().lstrip("."),
                })
    return {"documents": files, "total": len(files)}
