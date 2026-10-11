"""Ingestion pipeline orchestrator.

Ties together: connector → extract → metadata → dedupe → chunk → index → DB.

This module exposes a single public function ``ingest_file()`` that the
FastAPI routes call for both upload and sync operations.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from datetime import date
from pathlib import Path

from app.config import settings
from app.db import get_connection
from app.ingestion.chunker import chunk_pages
from app.ingestion.extract import extract_pages
from app.ingestion.indexer import index_chunks
from app.ingestion.metadata import parse_metadata
from app.ingestion.revisions import detect_superseded_notice_in_db

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(path: Path, original_filename: str | None = None) -> dict:
    """Ingest a single notice file end-to-end.

    Steps (per TECH_SPEC §5):
      1. Compute SHA-256; skip if already indexed.
      2. Extract text per page.
      3. Parse metadata from filename.
      4. Chunk text.
      5. Embed chunks and store in Chroma.
      6. Insert notice row in SQLite.
      7. Move file to data/processed/.

    Args:
        path:              Absolute path to the file.
        original_filename: Override for the stored source_file name.

    Returns:
        Dict with keys: ``status`` (``"indexed"`` or ``"duplicate"``),
        ``notice_id``, ``title``, ``chunks``.
    """
    fname = original_filename or path.name
    file_hash = _sha256(path)

    # ── Step 1: Dedupe ────────────────────────────────────────────────────────
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id, title FROM notices WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    if existing:
        logger.info("Duplicate detected: %s (matches notice id=%s). Skipping.", fname, existing["id"])
        return {"status": "duplicate", "notice_id": existing["id"], "title": existing["title"], "chunks": 0}

    # ── Step 2: Extract ───────────────────────────────────────────────────────
    pages = extract_pages(path)
    if not pages:
        logger.warning("No text extracted from %s. It may be a scanned PDF. Skipping.", fname)
        return {"status": "empty", "notice_id": None, "title": fname, "chunks": 0}

    # ── Step 3: Metadata ──────────────────────────────────────────────────────
    meta = parse_metadata(Path(fname))

    # ── Step 4: Chunk ─────────────────────────────────────────────────────────
    page_tuples = [(p.page, p.text) for p in pages]
    chunks = chunk_pages(page_tuples)
    if not chunks:
        logger.warning("Chunker produced no chunks for %s. Skipping.", fname)
        return {"status": "empty", "notice_id": None, "title": meta.title, "chunks": 0}

    # ── Step 5: Index into Chroma ─────────────────────────────────────────────
    notice_id = str(uuid.uuid4())
    notice_date_str = meta.notice_date.isoformat()
    index_chunks(
        chunks=chunks,
        notice_id=notice_id,
        title=meta.title,
        notice_date=notice_date_str,
        category=meta.category,
    )

    # ── Step 6: Insert SQLite row & Revision Detection ────────────────────────
    with get_connection() as conn:
        supersedes_id = detect_superseded_notice_in_db(
            title=meta.title,
            notice_date=meta.notice_date,
            category=meta.category,
            source_file=fname,
            conn=conn,
        )
        conn.execute(
            """
            INSERT INTO notices (id, title, notice_date, category, source_file, file_hash, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (notice_id, meta.title, notice_date_str, meta.category, fname, file_hash, supersedes_id),
        )
        conn.commit()

    logger.info(
        "Ingested '%s' → id=%s  chunks=%d  date=%s  category=%s  supersedes=%s",
        fname,
        notice_id,
        len(chunks),
        notice_date_str,
        meta.category,
        supersedes_id,
    )

    # ── Step 7: Move to processed/ ────────────────────────────────────────────
    processed_dir = Path(settings.data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / fname
    # Avoid overwriting an existing file with the same name
    if dest.exists():
        dest = processed_dir / f"{notice_id}_{fname}"
    shutil.move(str(path), str(dest))
    logger.info("Moved %s → %s", fname, dest)

    return {
        "status": "indexed",
        "notice_id": notice_id,
        "title": meta.title,
        "chunks": len(chunks),
        "supersedes_id": supersedes_id,
    }
