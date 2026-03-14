# Lucio Studio – Legal Document QA

A high-performance Document QA system built with Python (FastAPI), React (Next.js 14), BM25 keyword search, FAISS vector search, and OpenAI GPT-4.

This project allows you to drag-and-drop your own legal documents (PDF, DOCX, TXT, HTML) and instantly chat with them. The AI will provide answers with **precise citations** pointing to the exact document and page number where it found the information.

---

## 🚀 Getting Started

### 1. Requirements
* Python 3.10+
* Node.js 18+
* An OpenAI API Key

### 2. Setup the Backend (FastAPI)
Open a terminal and run the following commands:
```bash
# Navigate to the project folder
cd d:\lucio

# (Optional but recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn[standard] python-multipart sse-starlette

# Set your API Key
copy .env.example .env
```
👉 **IMPORTANT:** Open the `.env` file in a text editor and paste your OpenAI API key (`OPENAI_API_KEY=sk-...`). Without this, the AI cannot generate answers.

### 3. Setup the Frontend (Next.js Studio)
Open a **second** terminal and run:
```bash
cd d:\lucio\studio
npm install
```

---

## 🏃‍♂️ How to Run the App
Whenever you want to use the app, you need to run **both** servers.

### Terminal 1 - Start the Backend API
```bash
cd d:\lucio
# If using a virtual env, activate it first: .\venv\Scripts\activate
python -m uvicorn api.main:app --port 8000 --reload
```

### Terminal 2 - Start the Frontend UI
```bash
cd d:\lucio\studio
npm run dev
```

Finally, open your browser and go to: **[http://localhost:3000](http://localhost:3000)**

---

## 📂 Adding Your Own Documents

It is very easy to use your own files!

1. Open the `d:\lucio\documents\` folder.
2. Delete the sample files.
3. Paste in your own PDFs, Word documents (`.docx`), text files (`.txt`), or HTML files.
4. Go to the web app frontend (`http://localhost:3000`).
5. Click the **"Build Index"** button in the left sidebar.

The system will now ingest all your documents in parallel. You will see the live progress log, and once finished, it will display exactly **how many seconds** it took to process your files. 

After it says "Ready", you can start asking questions!

---

## 🏗 Architecture & Technologies

* **Ingestion:** `PyMuPDF` (PDFs), `python-docx` (Word). Runs on 6 parallel workers.
* **Chunking:** Sentence-aware sliding window (~1000 chars, 20% overlap).
* **Hybrid Search:** `rank_bm25` (Keyword) + `faiss-cpu` (Semantic Vector).
* **Reranking:** Reciprocal Rank Fusion (RRF) combines both search scores.
* **LLM Engine:** OpenAI `gpt-4o` with strict citation prompting.
* **Backend Framework:** FastAPI with Server-Sent Events (SSE) streaming.
* **Frontend UI:** Next.js 14 App Router, React, Tailwind CSS. Dark theme glassmorphism.
* **Caching:** Embeddings and indexes are cached to disk so you only pay the embedding cost once per document.
