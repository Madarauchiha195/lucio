"""
api/main.py
FastAPI application entry-point for the Lucio Studio backend.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import io
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import build, query, documents, status

app = FastAPI(
    title="Lucio Studio API",
    description="Hybrid BM25+FAISS RAG pipeline for legal document QA",
    version="1.0.0",
)

# Allow Next.js dev server (port 3000) and production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(build.router,     prefix="/api")
app.include_router(query.router,     prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(status.router,    prefix="/api")


@app.get("/")
def root():
    return {"message": "Lucio Studio API is running. Visit /docs for Swagger UI."}
