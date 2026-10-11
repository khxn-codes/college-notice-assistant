"""FastAPI application entry point.

Routes (Milestone 1 + 2):
  GET  /health            – liveness check
  POST /notices/upload    – upload and ingest a single file
  POST /notices/sync      – ingest all files in data/inbox/
  GET  /notices           – list indexed notices

Further routes will be added in later milestones:
  PATCH /notices/{id}, DELETE /notices/{id}, /ask, /eval/*
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_connection, init_db
from app.ingestion.folder_connector import FolderConnector
from app.ingestion.pipeline import ingest_file

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
    yield


app = FastAPI(
    title="College Notice Assistant",
    description="RAG chatbot for college notices — FastAPI backend.",
    version="0.2.0",
    lifespan=lifespan,
)

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

    # Write upload to a temp file so ingest_file can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path, original_filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    status_code = 200 if result["status"] == "indexed" else 200  # always 200; caller reads "status"
    return JSONResponse(content=result, status_code=status_code)


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
    for raw in raw_notices:
        try:
            result = ingest_file(raw.path, original_filename=raw.original_filename)
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
