"""FastAPI application entry point.

Routes (Milestone 1 + 2 + 3):
  GET  /health            – liveness check
  POST /notices/upload    – upload and ingest a single file
  POST /notices/sync      – ingest all files in data/inbox/
  GET  /notices           – list indexed notices
  POST /ask               – RAG question answering with citations

Further routes (Milestone 4+):
  PATCH /notices/{id}, DELETE /notices/{id}, /eval/*
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.db import get_connection, init_db
from app.ingestion.folder_connector import FolderConnector
from app.ingestion.pipeline import ingest_file
from app.rag import answer_question
from app.retrieval.bm25_store import rebuild_bm25_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks before the app accepts requests."""
    logger.info("Starting College Notice Assistant API …")
    init_db()
    # Best-effort: rebuild BM25 index from any chunks already in Chroma
    try:
        rebuild_bm25_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 index could not be built at startup: %s", exc)
    yield


app = FastAPI(
    title="College Notice Assistant",
    description="RAG chatbot for college notices — FastAPI backend.",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    category: str | None = None
    config: str | None = None   # "A" = vector-only, "B" = hybrid


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness probe — returns ``{"status": "ok"}``."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------


@app.post("/notices/upload", tags=["notices"])
async def upload_notice(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a single PDF or TXT notice and ingest it immediately.

    Returns the ingestion result (status, notice_id, title, chunks).
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Only PDF and TXT are accepted.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path, original_filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Rebuild BM25 after every successful ingest
    if result["status"] == "indexed":
        try:
            rebuild_bm25_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 rebuild failed after upload: %s", exc)

    return JSONResponse(content=result, status_code=200)


@app.post("/notices/sync", tags=["notices"])
async def sync_notices() -> JSONResponse:
    """Ingest all un-indexed files currently in ``data/inbox/``.

    Returns a list of per-file ingestion results.
    """
    connector = FolderConnector()
    raw_notices = connector.fetch_new()

    if not raw_notices:
        return JSONResponse(content={"message": "No new files found in inbox.", "results": []})

    results = []
    any_indexed = False
    for raw in raw_notices:
        try:
            result = ingest_file(raw.path, original_filename=raw.original_filename)
            if result["status"] == "indexed":
                any_indexed = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to ingest %s: %s", raw.original_filename, exc)
            result = {
                "status": "error",
                "notice_id": None,
                "title": raw.original_filename,
                "chunks": 0,
                "error": str(exc),
            }
        results.append(result)

    # Rebuild BM25 once after all ingestion
    if any_indexed:
        try:
            rebuild_bm25_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 rebuild failed after sync: %s", exc)

    return JSONResponse(content={"results": results})


@app.get("/notices", tags=["notices"])
async def list_notices() -> JSONResponse:
    """Return all indexed notices ordered by notice_date descending."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, notice_date, category, source_file, indexed_at, supersedes_id
            FROM notices
            ORDER BY notice_date DESC, indexed_at DESC
            """
        ).fetchall()

    notices = [dict(row) for row in rows]
    return JSONResponse(content={"notices": notices, "count": len(notices)})


# ---------------------------------------------------------------------------
# RAG — /ask
# ---------------------------------------------------------------------------


@app.post("/ask", tags=["rag"])
async def ask(request: AskRequest) -> JSONResponse:
    """Answer a question from the indexed notices using RAG.

    Body:
        question (str):  The student's question.
        category (str?): Optional category filter.
        config (str?):   "A" = vector-only, "B" = hybrid. Defaults to env setting.

    Returns:
        answer, citations, retrieved chunks, latency_ms, config label.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="'question' must not be empty.")

    logger.info(
        "POST /ask  question='%.80s'  category=%s  config=%s",
        request.question,
        request.category,
        request.config,
    )

    result = answer_question(
        question=request.question,
        category=request.category,
        config=request.config,
    )

    # ── Log to query_logs ─────────────────────────────────────────────────────
    _log_query(
        question=request.question,
        answer=result["answer"],
        chunk_ids=[c["chunk_id"] for c in result["chunks"]],
        config_name=result["config"],
        latency_ms=result["latency_ms"],
    )

    return JSONResponse(content=result)


def _log_query(
    question: str,
    answer: str,
    chunk_ids: list[str],
    config_name: str,
    latency_ms: int,
) -> None:
    """Persist a query record to the ``query_logs`` table."""
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO query_logs
                  (id, question, answer, retrieved_chunk_ids, config_name, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    question,
                    answer,
                    json.dumps(chunk_ids),
                    config_name,
                    latency_ms,
                ),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write query log: %s", exc)
