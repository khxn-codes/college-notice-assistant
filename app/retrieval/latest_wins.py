"""Latest-wins logic for retrieval.

TECH_SPEC §6 & Milestone 4:
  - If retrieved chunks come from both a notice and the notice that supersedes it,
    drop or down-rank the superseded notice's chunks.
  - Pass notes to the LLM indicating which notice supersedes which
    (e.g. "Notice B (2026-10-05) supersedes Notice A (2026-10-01)").
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def apply_latest_wins(
    chunks: list[RetrievedChunk],
    drop: bool = True,
    supersedes_map: dict[str, str | None] | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Filter or down-rank chunks from notices that have been superseded by newer notices.

    Args:
        chunks:          List of :class:`RetrievedChunk` returned by search.
        drop:            If True, remove superseded chunks completely.
                         If False, down-rank superseded chunks to the end.
        supersedes_map:  Optional explicit mapping of ``{notice_id: supersedes_id}``
                         for testing without SQLite.
        conn:            Optional SQLite connection. If None, opens a temporary one.

    Returns:
        Tuple of ``(processed_chunks, supersedes_notes)``.
    """
    if not chunks:
        return [], []

    notice_ids = {c.notice_id for c in chunks if c.notice_id}
    if len(notice_ids) <= 1:
        return chunks, []

    # Map notice_id -> (title, notice_date, supersedes_id)
    notice_info: dict[str, dict] = {}

    # Initialise from chunk metadata as fallback
    for c in chunks:
        if c.notice_id and c.notice_id not in notice_info:
            notice_info[c.notice_id] = {
                "title": c.title,
                "notice_date": c.notice_date,
                "supersedes_id": supersedes_map.get(c.notice_id) if supersedes_map else None,
            }

    # Query DB for full supersedes_id links and exact titles/dates
    if supersedes_map is None:
        try:
            from app.db import get_connection

            own_conn = conn is None
            c_conn = conn or get_connection()
            try:
                placeholders = ",".join("?" for _ in notice_ids)
                rows = c_conn.execute(
                    f"SELECT id, title, notice_date, supersedes_id FROM notices WHERE id IN ({placeholders})",
                    list(notice_ids),
                ).fetchall()
                for r in rows:
                    notice_info[r["id"]] = {
                        "title": r["title"],
                        "notice_date": str(r["notice_date"] or ""),
                        "supersedes_id": r["supersedes_id"],
                    }
            finally:
                if own_conn:
                    c_conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not query notices table for latest-wins check: %s", exc)

    # Check for direct or chained supersedes relationships
    superseded_ids: set[str] = set()
    notes: list[str] = []

    for nid in notice_ids:
        curr_newer = nid
        curr_target = notice_info.get(nid, {}).get("supersedes_id")

        while curr_target:
            if curr_target in notice_ids:
                superseded_ids.add(curr_target)
                newer_meta = notice_info.get(curr_newer, {})
                older_meta = notice_info.get(curr_target, {})

                newer_title = newer_meta.get("title") or curr_newer
                newer_date = newer_meta.get("notice_date") or ""
                older_title = older_meta.get("title") or curr_target
                older_date = older_meta.get("notice_date") or ""

                newer_str = f"{newer_title} ({newer_date})".replace(" ()", "")
                older_str = f"{older_title} ({older_date})".replace(" ()", "")
                note = f"{newer_str} supersedes {older_str}"

                if note not in notes:
                    notes.append(note)

            # Follow transitive chain if target itself supersedes another
            curr_target = notice_info.get(curr_target, {}).get("supersedes_id")

    if not superseded_ids:
        return chunks, []

    logger.info(
        "Latest-wins triggered: dropping/downranking superseded notices %s; notes=%s",
        superseded_ids,
        notes,
    )

    if drop:
        filtered_chunks = [c for c in chunks if c.notice_id not in superseded_ids]
    else:
        from app.retrieval.retriever import RetrievedChunk

        kept = [c for c in chunks if c.notice_id not in superseded_ids]
        downranked = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                notice_id=c.notice_id,
                title=c.title,
                notice_date=c.notice_date,
                category=c.category,
                page=c.page,
                chunk_index=c.chunk_index,
                score=c.score * 0.01,
            )
            for c in chunks
            if c.notice_id in superseded_ids
        ]
        filtered_chunks = kept + downranked

    return filtered_chunks, notes
