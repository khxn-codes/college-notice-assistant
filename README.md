# NoticeIQ: RAG-Based AI Assistant for College Notices

> An intelligent, citation-backed question-answering assistant for college students and faculty. NoticeIQ indexes institutional notices, resolves conflicting or revised announcements with a **"Latest Wins"** policy, and provides exact source citations with zero hallucination.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com/)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
  - [Running on Windows](#running-on-windows)
  - [Running on Linux or macOS](#running-on-linux-or-macos)
- [Adding & Managing Notices](#adding--managing-notices)
- [Running Tests](#running-tests)
- [Running the Evaluation Suite](#running-the-evaluation-suite)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Project Overview

Students frequently miss crucial deadlines and updates because notices are buried across college ERP portals, official PDFs, and messaging groups. When a notice is rescheduled or revised, outdated information persists, causing confusion.

**NoticeIQ** solves this with a privacy-preserving, local-first RAG pipeline:
1. Ingests raw PDFs and text notices via an inbox directory or web UI.
2. Extracts text, normalizes metadata (dates, categories, slugs), and links revised notices to earlier ones.
3. Employs hybrid retrieval (dense semantic vectors + sparse BM25 keyword matching) with Reciprocal Rank Fusion (RRF).
4. Enforces strict factual grounding: answers are accompanied by `[Title, Date, p.N]` citations, and questions not backed by notices trigger an exact refusal phrase:
   > *"I could not find this in the available notices."*

---

## Key Features

- **Hybrid Search (Dense + BM25):** Combines `sentence-transformers` embeddings with BM25 keyword search to accurately match both conceptual queries and exact identifiers (course codes, fee figures, room numbers).
- **Latest-Wins Conflict Resolution:** Automatically identifies revision notices (e.g., *"Mid-Sem Exam Revised Schedule"* vs *"Mid-Sem Exam Schedule"*), links the superseding notice, and downranks or filters outdated notices while explicitly informing the user which notice was superseded.
- **Strict Grounding & Citations:** Every claim is cited as `[Notice Title, YYYY-MM-DD, p.X]`. Low temperature generation prevents hallucinations.
- **Swappable Connector Interface:** Pluggable connector architecture allows seamless drop-in of folder watchers or future ERP APIs.
- **Comprehensive UI:** Three-page Streamlit interface:
  - 💬 **Chat:** Ask questions, filter by category, view streaming answers with collapsible source passage viewers.
  - 📋 **Notices:** Browse indexed notices, view metadata, upload new documents, edit superseded links, or delete records.
  - 📊 **Evaluation:** Run retrieval-only or full-generation evaluations, view Hit@5, MRR, Answer Accuracy, Citation Hit Rate, and compare configurations with charts and downloadable CSVs.
- **Robust Failure Handling:** Includes API rate-limit exponential backoff, immediate abort on quota exhaustion with partial result preservation, and an LLM answer cache to avoid paying for repeated evaluations.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        A[Notices in data/inbox or UI Upload] --> B[PyMuPDF / pypdf Text Extraction]
        B --> C[Metadata & Revision Parser]
        C --> D[Sliding Window Chunker]
        D --> E[BAAI/bge-small-en-v1.5 Embedder]
        E --> F[(ChromaDB Vector Store)]
        D --> G[(In-Memory BM25 Index)]
        C --> H[(SQLite app.db Metadata)]
    end

    subgraph Query & RAG Pipeline
        Q[User Query] --> R{Category Filter?}
        R --> S[Dense Vector Search]
        R --> T[Sparse BM25 Search]
        S & T --> U[Reciprocal Rank Fusion - RRF]
        U --> V[Latest-Wins Re-ranker]
        V --> W[Top-K Relevant Passages]
        W --> X[Prompt Builder with Citation Rules]
        X --> Y[Configurable LLM Provider]
        Y --> Z[Cited Answer or Exact Refusal]
    end

    subgraph Evaluation & UI
        Z --> UI[Streamlit Web UI]
        EVAL[eval/test_set.json] --> EVAL_RUNNER[Evaluation Runner]
        EVAL_RUNNER --> METRICS[Hit@5, MRR, Accuracy, Latency]
        METRICS --> UI
    end
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | Streamlit |
| **Vector Database** | ChromaDB (persistent local storage in `data/chroma`) |
| **Keyword Search** | `rank_bm25` |
| **Embeddings** | `sentence-transformers` (`BAAI/bge-small-en-v1.5` default, run locally) |
| **PDF Extraction** | PyMuPDF (`fitz`), fallback to `pypdf` |
| **Metadata DB** | SQLite (`data/app.db`) |
| **LLM Providers** | Google Gemini (`google-generativeai`), OpenAI, Anthropic, or local endpoints |
| **Testing** | `pytest`, `pytest-asyncio`, `httpx` |
| **Launcher** | PowerShell (`run.ps1`), Bash (`run.sh`), `Makefile` |

---

## Folder Structure

```
college-notice-assistant/
├── PRD.md                  # Product Requirements Document
├── TECH_SPEC.md            # Technical Architecture & Milestones
├── AGENTS.md               # Guidelines & rules for AI coding pairs
├── README.md               # Project documentation (this file)
├── .env.example            # Environment variable template
├── requirements.txt        # Pinned production dependencies
├── run.ps1                 # Windows PowerShell one-click launcher
├── run.sh                  # Linux / macOS Bash one-click launcher
├── Makefile                # Standard developer tasks (install, run, test)
├── app/
│   ├── main.py             # FastAPI app and API route definitions
│   ├── config.py           # Configuration management via Pydantic/env
│   ├── db.py               # SQLite models, connection, and queries
│   ├── ingestion/
│   │   ├── connector.py    # Base Connector abstract interface
│   │   ├── folder_connector.py # Inbox folder connector implementation
│   │   ├── extract.py      # PDF & text file extraction with page tracking
│   │   ├── metadata.py     # Filename & header metadata parser
│   │   ├── chunker.py      # Character/token sliding window chunking
│   │   ├── indexer.py      # ChromaDB embeddings + BM25 indexing
│   │   └── revisions.py    # Superseded notice matching logic
│   ├── retrieval/
│   │   ├── retriever.py    # Hybrid RRF search, filtering, and query logic
│   │   └── latest_wins.py  # Conflict detection and superseded downranking
│   └── llm/
│       ├── client.py       # Multi-provider LLM client with retry & quota handling
│       └── prompts.py      # Strict system prompts and citation formatting
├── ui/
│   └── streamlit_app.py    # Streamlit multipage application (Chat, Notices, Eval)
├── eval/
│   ├── test_set.json       # Labeled evaluation test questions
│   ├── runner.py           # Test execution, retry handling, and metrics computation
│   └── results/            # Persistent JSON and CSV evaluation benchmark runs
├── data/
│   ├── inbox/              # Drop incoming notice PDFs/TXTs here
│   ├── processed/          # Stored documents post-ingestion
│   ├── chroma/             # ChromaDB vector store files
│   └── app.db              # SQLite metadata database
└── tests/                  # Pytest automated test suite (110+ unit/e2e tests)
```

---

## Setup & Installation

### Prerequisites
- **Python 3.11+** installed and on your `PATH`
- An API key for your chosen LLM provider (e.g., Google Gemini, OpenAI)

### Configuration
1. Clone the repository:
   ```bash
   git clone https://github.com/khxn-codes/college-notice-assistant.git
   cd college-notice-assistant
   ```
2. Copy the template environment file:
   ```bash
   # Windows
   copy .env.example .env

   # Linux / macOS
   cp .env.example .env
   ```
3. Edit `.env` and fill in your LLM credentials:
   ```ini
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-2.5-flash
   LLM_API_KEY=your_gemini_api_key_here
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   CHUNK_SIZE=500
   CHUNK_OVERLAP=80
   TOP_K=5
   DATA_DIR=./data
   ```

4. Create a virtual environment and install dependencies:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

### Running on Windows

Run the provided PowerShell script (it automatically creates required directories and starts both services):

```powershell
.\.venv\Scripts\Activate.ps1
.\run.ps1
```

Or using Makefile:
```powershell
make install
make run
```

---

### Running on Linux or macOS

```bash
source .venv/bin/activate
chmod +x run.sh
./run.sh
```

Or using Makefile:
```bash
make install
make run
```

Once running:
- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Backend:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

> **Note:** Press `Ctrl+C` in the terminal to stop both services.

---

## Adding & Managing Notices

NoticeIQ supports two simple methods to add college notices:

### Method 1: Drop Files into `data/inbox/`
Place your PDF or TXT notices into the `data/inbox/` folder. For optimal metadata parsing, follow the naming convention:
```
YYYY-MM-DD_category_slug-title.pdf
```
*Examples:*
- `2026-10-01_exam_midsem-schedule.pdf`
- `2026-10-05_exam_midsem-schedule-revised.pdf`
- `2026-09-15_fees_odd-semester-payment-deadline.pdf`

Then click **Sync Inbox** on the **Notices** page in the UI or call `POST /sync`.

### Method 2: Upload Directly via the Web UI
1. Navigate to the **Notices** page in the Streamlit UI.
2. Use the file uploader to drag & drop one or multiple PDF / TXT files.
3. The backend will parse, chunk, embed, and index them in real-time.

---

## Running Tests

NoticeIQ has a comprehensive test suite of 110+ tests covering extraction, chunking, deduplication, hybrid retrieval, latest-wins logic, prompt constraints, and evaluation metrics:

```bash
# Activate venv first, then:
pytest -v

# Or using Makefile
make test
```

---

## Running the Evaluation Suite

NoticeIQ includes a built-in evaluation framework to benchmark RAG configurations against ground-truth questions in `eval/test_set.json`.

### 1. In the Web UI
1. Open the **Evaluation** tab in Streamlit.
2. Choose evaluation mode:
   - **Retrieval-Only Mode (Fast & Free):** Evaluates dense+BM25 ranking without calling the LLM. Calculates **Hit@5** and **MRR** (Mean Reciprocal Rank).
   - **Full Evaluation:** Generates responses and evaluates keyword accuracy, citation hit rate, refusal accuracy, and latency.
3. Select your desired configuration (e.g., `hybrid_rrf_k5`, `hybrid_rrf_k3`, or `dense_only`).
4. Click **Run Evaluation** to view real-time progress, comparison bar charts, and exportable CSVs.

### 2. From the CLI
```bash
python eval/runner.py
```
Evaluation runs are saved with timestamps in `eval/results/`.

---

## Known Limitations

- **Scanned Image PDFs:** Pure image/scanned PDFs without embedded OCR text layers will yield empty text. A future OCR pipeline (e.g., Tesseract/PaddleOCR) is required for scanned flyers.
- **Complex Multipage Tables:** Complex nested financial or hostel room allocation tables may lose row-column structure during naive chunking.
- **Provider API Quotas:** When running large test suites with free-tier cloud LLMs (such as Google Gemini free tier), rate limits (15 RPM) or daily quotas can occur. NoticeIQ implements exponential backoff and run abortion to protect data and retain completed results.

---

## Future Work

- **Automatic ERP Connector:** Implement an automated connector that periodically pulls announcements from college ERP endpoints when university APIs are authorized.
- **Multilingual Support (Hindi & English):** Bilingual query understanding and responses to assist students comfortable in regional languages.
- **Voice Queries:** Web Audio / Whisper speech-to-text integration for mobile hands-free interactions.
- **OCR Preprocessing:** Integrated Tesseract / EasyOCR fallback for scanned circulars and stamped notices.
- **Fine-Tuned Cross-Encoder Reranker:** Integration of `bge-reranker-base` to further refine hybrid retrieval ranking.

---

## License

This project is developed for educational and institutional research purposes under the MIT License.
