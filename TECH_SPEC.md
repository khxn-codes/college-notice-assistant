# TECH SPEC: NoticeIQ (RAG-Based AI Assistant for Answering College Notice Queries with Citations)

## 1. Architecture

```
Exported notices -> ingestion (extract, dedupe, metadata) -> chunk -> embed -> vector DB (Chroma)
                                                                                    |
User question -> (optional category filter) -> hybrid retrieval (vector + BM25)
              -> rerank/sort by date rule -> prompt builder -> LLM -> answer + citations
                                                                                    |
                                            query logs -> evaluation module -> dashboard
```

## 2. Tech stack
- Language: Python 3.11+
- Backend: FastAPI
- UI: Streamlit (calls the FastAPI backend)
- PDF extraction: PyMuPDF (`pymupdf`), fallback `pypdf`
- Embeddings: `sentence-transformers` (`BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`), run locally
- Vector DB: ChromaDB (persistent local directory)
- Keyword search: `rank_bm25`
- LLM: provider-agnostic wrapper; model and API key via environment variables
- Storage: SQLite for notice metadata and query logs
- Testing: pytest
- Packaging: `requirements.txt`, `.env.example`, `run.ps1` (Windows), `run.sh` (Linux/macOS)

## 3. Folder structure

```
college-notice-assistant/
  PRD.md
  TECH_SPEC.md
  AGENTS.md
  README.md
  .env.example
  requirements.txt
  run.ps1               # Windows one-click launcher
  run.sh                # Linux/macOS one-click launcher
  Makefile              # Developer task shortcuts
  data/
    inbox/            # drop exported notices here
    processed/        # raw files after ingestion
    chroma/           # vector DB persistence
    app.db            # SQLite
  eval/
    test_set.json     # evaluation questions
    results/          # saved evaluation runs
  app/
    main.py           # FastAPI app and routes
    config.py         # settings from env
    db.py             # SQLite models and helpers
    ingestion/
      connector.py    # base Connector interface
      folder_connector.py  # reads data/inbox (current implementation)
      extract.py      # PDF/TXT text extraction
      metadata.py     # title/date/category parsing
      chunker.py
      indexer.py      # embeddings + Chroma + BM25 index
    retrieval/
      retriever.py    # hybrid search, filters, date logic
    llm/
      client.py       # LLM wrapper
      prompts.py      # prompt templates
    rag.py            # orchestrates retrieve -> prompt -> answer
    evaluation/
      runner.py       # runs test set across configs
      metrics.py
  ui/
    streamlit_app.py  # chat, upload, library, evaluation pages
  tests/
```

## 4. Data model

### notices (SQLite)
| field | type | notes |
|---|---|---|
| id | text (uuid) | primary key |
| title | text | from filename or first heading |
| notice_date | date | from filename convention or text; fallback upload date |
| category | text | exam, fees, hostel, events, general |
| source_file | text | original filename |
| file_hash | text | SHA-256, used for dedupe |
| supersedes_id | text, nullable | set when a newer notice revises an older one |
| indexed_at | datetime | |

### Chunk metadata (stored in Chroma)
`notice_id, title, notice_date, category, page, chunk_index`

### query_logs (SQLite)
`id, question, answer, retrieved_chunk_ids, config_name, latency_ms, created_at`

### Filename convention (preferred)
`YYYY-MM-DD_category_short-title.pdf`, for example `2026-10-05_exam_midsem-schedule.pdf`. If the convention is not followed, parse the date from text, otherwise use the upload date, and let the user edit the metadata in the UI.

## 5. Ingestion pipeline
1. Connector yields new files from `data/inbox/` (or from the upload endpoint).
2. Compute SHA-256. If the hash already exists, skip it.
3. Extract text per page (PyMuPDF). Keep page numbers.
4. Parse metadata (title, date, category).
5. Chunk text: default 500 tokens with 80 overlap, split on paragraph or sentence boundaries. Chunk size and overlap are configurable so the evaluation can compare settings.
6. Embed chunks and store in Chroma with metadata. Update the BM25 index.
7. Insert the notice row in SQLite.
8. Revision detection: if a new notice has the same category and a highly similar title to an older notice and a later date, set `supersedes_id`. Allow manual override in the UI.
9. Move the raw file to `data/processed/`.

The connector interface (`Connector.fetch_new() -> list[RawNotice]`) must stay swappable so an ERP connector can replace the folder connector later without changing the rest of the system.

## 6. Retrieval
- Hybrid search: top-k from vector similarity (k=8) and top-k from BM25 (k=8), merged with Reciprocal Rank Fusion, final top 5.
- Optional metadata filter by category.
- Latest-wins rule: if retrieved chunks come from notices linked by `supersedes_id`, drop or de-prioritize the superseded notice's chunks and tell the LLM which notice supersedes which.
- Two configurations for evaluation:
  - Config A: vector-only, chunk size 800.
  - Config B: hybrid, chunk size 500.
  (Config values live in `config.py` so more can be added.)

## 7. Prompt rules (LLM)
System prompt requirements:
- Answer only from the provided context passages.
- If the context does not contain the answer, reply exactly: "I could not find this in the available notices."
- Always cite sources as `[Title, Date, p.N]`.
- If notices conflict, use the most recent and mention the older one it supersedes.
- Do not invent dates, amounts, or names.
- Keep answers short and direct.

Context is passed as numbered passages with title, date, and page. Low temperature (0 to 0.2).

## 8. API endpoints (FastAPI)
| method | path | purpose |
|---|---|---|
| POST | `/notices/upload` | upload and ingest a file |
| POST | `/notices/sync` | ingest everything in `data/inbox/` |
| GET | `/notices` | list notices with metadata |
| PATCH | `/notices/{id}` | edit title, date, category, supersedes |
| DELETE | `/notices/{id}` | remove a notice and its chunks |
| POST | `/ask` | body: `question`, optional `category`, `config`; returns answer, citations, retrieved chunks, latency |
| POST | `/eval/run` | run the test set for given configs |
| GET | `/eval/results` | list saved evaluation runs |
| GET | `/health` | health check |

## 9. UI (Streamlit pages)
1. **Chat:** question box, category filter, answer with citations, expandable "Sources" showing retrieved chunks.
2. **Notices:** upload area, sync button, table of indexed notices with last-updated time, edit and delete.
3. **Evaluation:** run the test set, show a table and bar chart comparing configs (accuracy, citation hit rate, refusal accuracy, average latency).

## 10. Evaluation
`eval/test_set.json` entries:

```json
{
  "id": "q01",
  "question": "What is the revised date of the mid-sem exam?",
  "expected_answer_contains": ["17 October"],
  "expected_source": "2026-10-05_exam_midsem-revised.pdf",
  "answerable": true
}
```

- 20-30 questions: mostly answerable, at least 5 unanswerable, at least 3 about revised notices.
- Metrics:
  - Answer accuracy: expected keywords present (optionally LLM-as-judge as a second opinion).
  - Citation hit rate: expected source appears in the citations.
  - Refusal accuracy: unanswerable questions correctly refused.
  - Average latency.
- Results saved to `eval/results/` and shown in the dashboard.

## 11. Configuration (`.env.example`)
```
LLM_PROVIDER=
LLM_MODEL=
LLM_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=500
CHUNK_OVERLAP=80
TOP_K=5
DATA_DIR=./data
```
Never commit `.env`.

## 12. Non-functional requirements
- Runs locally with one command (`.\run.ps1` on Windows or `make run`).
- No student personal data is ingested or stored.
- Graceful errors: unreadable PDF, empty text, LLM failure (show a clear message, do not crash).
- Logging for ingestion and queries.
- Unit tests for chunking, metadata parsing, dedupe, and the latest-wins logic; one end-to-end test for `/ask`.

## 13. Milestones (give Antigravity one at a time)
1. **Scaffold:** repo structure, config, SQLite setup, FastAPI health check. *Done when:* `/health` returns OK and tests run.
2. **Ingestion:** folder connector, extraction, metadata parsing, dedupe, chunking, Chroma indexing. *Done when:* sample notices in `data/inbox/` are indexed and duplicates are skipped.
3. **Retrieval and RAG:** hybrid retrieval, prompt builder, LLM client, `/ask` with citations and the refusal rule. *Done when:* answerable questions return cited answers and unanswerable ones are refused.
4. **Latest-wins logic:** revision detection and the superseded-notice handling. *Done when:* a revised-notice question returns the newest information.
5. **UI:** Streamlit chat, notices, and upload pages. *Done when:* the full flow works from the browser.
6. **Evaluation:** test set, runner, metrics, dashboard comparing two configs. *Done when:* a results table and chart are shown.
7. **Polish:** README with setup and screenshots, error handling, final tests. *Done when:* a fresh clone runs with one command.*

## 14. Rules for the coding agent
See `AGENTS.md` in the repo root. In short: follow the PRD and this spec, work one milestone at a time, keep the connector swappable, never hardcode secrets, and run tests after every change.
