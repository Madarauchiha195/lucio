"""
config.py
Central configuration for the Document QA pipeline.
Adjust values here – no other file needs to change.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)               # loads .env and overrides existing vars

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DOCS_DIR    = BASE_DIR / "documents"    # place your PDFs / DOCX / HTML here
CACHE_DIR   = BASE_DIR / "cache"        # auto-created; stores indexes & embeddings
OUTPUT_DIR  = BASE_DIR / "output"       # auto-created; per-question result JSONs

DOCS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Supported document extensions ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".csv"}

# ── Ingestion ─────────────────────────────────────────────────────────────────
INGESTION_WORKERS = 6                   # ProcessPoolExecutor workers (4–8)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000                    # target characters per chunk
CHUNK_OVERLAP = 150                     # overlap between adjacent chunks (~15%)

# ── Embedding model (local, no API cost) ──────────────────────────────────────
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH     = 256                   # sentences per embedding batch
EMBED_DIM       = 384                   # dimension of all-MiniLM-L6-v2

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH  = CACHE_DIR / "faiss_index.bin"
CHUNKS_CACHE_PATH = CACHE_DIR / "chunks.pkl"
EMBED_CACHE_PATH  = CACHE_DIR / "embeddings.npy"

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = CACHE_DIR / "bm25_index.pkl"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_BM25  = 20                        # BM25 candidate pool
TOP_K_FAISS = 20                        # FAISS candidate pool
TOP_K_FINAL = 10                        # chunks sent to LLM after fusion
RRF_K       = 60                        # RRF constant

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL      = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1024

# Concurrent LLM calls for the 15-question batch
LLM_CONCURRENCY = 5
