# Lucio Studio — Advanced Legal Document QA

A production-grade, hybrid retrieval system for legal document analysis featuring BM25 + dense vector search, cross-encoder reranking, async LLM answering, and a legal document graph.

---

## 🗂 Project Structure

```
lucio/
├── ingestion/          # File detection + per-format extractors (PDF/DOCX/HTML/CSV with OCR)
├── processing/         # Dual chunker: page_chunker + token_chunker (tiktoken)
├── indexing/           # BM25 (boolean+proximity) + FAISS vector index + hybrid RRF retriever
├── reranker/           # Cross-encoder reranker (ms-marco-MiniLM)
├── qa/                 # LLM client adapters (Gemini/OpenAI/Local) + batch async answering
├── evidence_mapper/    # networkx document graph with citation extraction
├── ui_utils/           # Boolean/proximity/clause/citation search tools
├── performance/        # Benchmark harness with bottleneck analysis
├── tests/              # pytest unit + integration tests
├── api/                # FastAPI web API + Next.js Studio frontend
├── studio/             # Next.js 14 frontend
├── run_challenge.py    # ⭐ Single entry-point for Lucio Challenge
├── config.py           # All configuration flags
└── documents/          # Place your docs here (PDF, DOCX, TXT, CSV, HTML)
```

---

## 🚀 Quick Start (Local)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for Studio frontend)
- CUDA toolkit (optional, for GPU speedup)

### 2. Install Dependencies
```bash
cd d:\lucio
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install tiktoken networkx sentence-transformers chromadb python-magic-bin google-genai openpyxl
```

### 3. Configure API Keys & Settings
Edit `d:\lucio\.env`:
```env
GEMINI_API_KEY=your-gemini-key

# Optional toggles
USE_OPENAI_API=false
OPENAI_API_KEY=sk-...
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GPU_DEVICE=cpu       # or: cuda  mps
BM25_TOPK=30
ANN_TOPK=30
RERANKER_TOPK=10
RUNTIME_LIMIT_SECONDS=30
```

### 4. Add Your Documents
Place all PDFs, DOCX, TXT, CSV, or HTML files in `d:\lucio\documents\`.

### 5. Run the Challenge Pipeline
```bash
python run_challenge.py --force
```

| Flag | Effect |
|---|---|
| `--force` | Rebuild index even if cached |
| `--strategy page` | Use page chunker instead of token |
| `--questions path.xlsx` | Custom question file |
| `--benchmark` | Print per-stage timing breakdown |
| `--download` | Run corpus downloader first |
| `--verbose` | DEBUG logging |

---

## 🌐 Running the Web Studio

**Terminal 1 — Backend:**
```bash
python -m uvicorn api.main:app --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd studio
npm install
npm run dev
```
Open **http://localhost:3000**

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage:
- `test_extractor.py` — TXT, HTML, CSV parsers
- `test_bm25.py` — basic search, boolean AND/NOT, proximity search
- `test_integration.py` — 200-doc simulation, schema validation, runtime check

---

## ⚙️ LLM Toggle (Gemini vs OpenAI vs Local)

| Mode | .env setting |
|---|---|
| **Gemini** (default) | `GEMINI_API_KEY=...` |
| **OpenAI** | `USE_OPENAI_API=true` + `OPENAI_API_KEY=...` + `LLM_MODEL=gpt-4o` |
| **Local LLM** | Install `llama-cpp-python`, set `LOCAL_MODEL_PATH=/path/to/mistral.gguf` |

---

## ☁️ Running on Google Colab

See `COLAB_GUIDE.md` or upload `lucio_colab.ipynb` directly to Colab.

## 🖥️ Running on a Rented GPU

See `OPS_GUIDE.md` for step-by-step instructions to rent A10G / RTX3090 and run the pipeline at full speed.
