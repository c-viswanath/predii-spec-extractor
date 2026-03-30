# Ford F-150 RAG — Specification Extractor & Conversational Interface

A **Retrieval-Augmented Generation (RAG)** system for the 2014 Ford F-150 Workshop Manual. Ask questions in natural language and get specifications, procedures, and torque values — all processed locally.

## Features

- **Hybrid Retrieval** — Semantic search (FAISS + MiniLM) + keyword search with spelling-variant matching
- **4-Tier Pipeline** — Cache → Session Memory → Pre-extracted Spec DB → Full RAG
- **Multi-Model Routing** — Llama 3.2:1B (intent), 3.2:3B (simple), 3.1:8B (complex)
- **Structured Extraction** — JSON spec extraction with tables & CSV export
- **ChatGPT-Style UI** — Typewriter animation, resizable panels, PDF deep-links, emoji chat labels, auto-titling
- **Privacy-First** — All models run locally via Ollama; no data leaves your machine

## Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Parsing | PyMuPDF + OCR fallback |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Vector Index | FAISS (CPU) |
| LLM Runtime | Ollama (Llama 3.x) |
| API Server | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) installed

```bash
# Pull required models
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

### Setup

```bash
# Install Python deps
pip install -r requirements.txt

# Build vector index (first time only)
python src/pipeline.py

# Start backend
python server.py

# In a new terminal — start frontend
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### Data

Place the workshop manual PDF at: `data/sample-service-manual 1.pdf`

## Project Structure

```
predii_spec_extractor/
├── server.py              # FastAPI server (4-tier pipeline)
├── requirements.txt
├── src/
│   ├── pdf_parser.py      # PyMuPDF + OCR extraction
│   ├── chunker.py         # Sliding-window chunker (800/150)
│   ├── embedder.py        # MiniLM + FAISS vector store
│   ├── extractor.py       # LLM spec extraction & QA
│   └── pipeline.py        # Index builder + pre-extraction
├── frontend/
│   └── src/
│       ├── App.jsx        # React chat interface
│       └── index.css      # Dark theme styles
├── report/
│   └── main.tex           # LaTeX technical report
└── data/                  # Workshop manual PDF (gitignored)
```

## Architecture

```
User Query → FastAPI Server
  ├─ Tier 1: Query Cache        (<1ms)
  ├─ Tier 2: Session Memory     (in-session follow-ups)
  ├─ Tier 3: Pre-extracted DB   (keyword lookup)
  └─ Tier 4: Full RAG
       ├─ FAISS semantic search (top-15)
       ├─ Keyword hybrid search (top-10)
       ├─ Chunk reranking (top-5 → LLM)
       └─ Ollama LLM generation
```
