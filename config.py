"""
config.py
Central configuration for the Advanced Legal Retrieval QA pipeline.
All toggles live here — no other file needs to change.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)               # loads .env and overrides existing vars

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DOCS_DIR    = BASE_DIR / "documents"
CACHE_DIR   = BASE_DIR / "cache"
OUTPUT_DIR  = BASE_DIR / "output"

DOCS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Supported document extensions ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".csv"}

# ── Ingestion ─────────────────────────────────────────────────────────────────
INGESTION_WORKERS = 6

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 1000          # characters (for char chunker backward compat)
CHUNK_OVERLAP     = 150
TOKEN_CHUNK_MIN   = 400           # tokens (tiktoken)
TOKEN_CHUNK_MAX   = 600           # tokens
TOKEN_CHUNK_OVERLAP_PCT = 0.12    # 12% overlap

# ── Embedding model ───────────────────────────────────────────────────────────
# Set USE_OPENAI_API=true in .env to use OpenAI embeddings instead
USE_OPENAI_API      = os.getenv("USE_OPENAI_API", "false").lower() == "true"
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH         = 256
EMBED_DIM           = 384         # for all-MiniLM-L6-v2; auto-detected for others

# ── FAISS / ANN ───────────────────────────────────────────────────────────────
FAISS_INDEX_PATH  = CACHE_DIR / "faiss_index.bin"
CHUNKS_CACHE_PATH = CACHE_DIR / "chunks.pkl"
EMBED_CACHE_PATH  = CACHE_DIR / "embeddings.npy"

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = CACHE_DIR / "bm25_index.pkl"

# ── Retrieval ─────────────────────────────────────────────────────────────────
BM25_TOPK       = int(os.getenv("BM25_TOPK",   "30"))
ANN_TOPK        = int(os.getenv("ANN_TOPK",    "30"))
RERANKER_TOPK   = int(os.getenv("RERANKER_TOPK","10"))
TOP_K_FINAL     = 10
RRF_K           = 60

# ── Reranker ───────────────────────────────────────────────────────────────────
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_DEVICE      = os.getenv("GPU_DEVICE", "cpu")   # "cuda", "mps", or "cpu"

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1024
LLM_CONCURRENCY = 5

# Maximum passages injected into LLM context (keep low for speed)
LLM_CONTEXT_PASSAGES = 3

# ── Performance ───────────────────────────────────────────────────────────────
RUNTIME_LIMIT_SECONDS = int(os.getenv("RUNTIME_LIMIT_SECONDS", "30"))
