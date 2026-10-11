"""Unit tests for retrieval filtering and latest-wins logic.

TECH_SPEC §6 & Milestone 4:
  - If retrieved chunks come from both a notice and the notice that supersedes it,
    drop or down-rank the superseded notice's chunks.
  - Pass a note to the LLM: "Notice B (date) supersedes Notice A (date)".
"""

from __future__ import annotations

from pathlib import Path
import pytest

from app.db import get_connection, init_db
from app.retrieval.latest_wins import apply_latest_wins
from app.retrieval.retriever import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    notice_id: str,
    title: str,
    notice_date: str,
    score: float = 1.0,
    text: str = "Test text",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        notice_id=notice_id,
        title=title,
        notice_date=notice_date,
        category="exam",
        page=1,
        chunk_index=0,
        score=score,
    )


def test_no_filtering_single_notice() -> None:
    chunks = [
        _make_chunk("c1", "n1", "Notice 1", "2026-10-01"),
        _make_chunk("c2", "n1", "Notice 1", "2026-10-01"),
    ]
    filtered, notes = apply_latest_wins(chunks, drop=True, supersedes_map={"n1": None})
    assert len(filtered) == 2
    assert notes == []


def test_no_filtering_unrelated_notices() -> None:
    chunks = [
        _make_chunk("c1", "n1", "Notice 1", "2026-10-01"),
        _make_chunk("c2", "n2", "Notice 2", "2026-10-02"),
    ]
    # Neither supersedes the other
    filtered, notes = apply_latest_wins(chunks, drop=True, supersedes_map={"n1": None, "n2": None})
    assert len(filtered) == 2
    assert notes == []


def test_drop_superseded_notice_chunks() -> None:
    """When chunks come from both Notice B and Notice A (B supersedes A), drop A's chunks."""
    chunk_a1 = _make_chunk("c_a1", "n_a", "Midsem Schedule", "2026-10-01", score=0.8)
    chunk_a2 = _make_chunk("c_a2", "n_a", "Midsem Schedule", "2026-10-01", score=0.7)
    chunk_b1 = _make_chunk("c_b1", "n_b", "Midsem Schedule Revised", "2026-10-05", score=0.9)

    chunks = [chunk_b1, chunk_a1, chunk_a2]
    # n_b supersedes n_a
    supersedes_map = {"n_b": "n_a", "n_a": None}

    filtered, notes = apply_latest_wins(chunks, drop=True, supersedes_map=supersedes_map)

    # A's chunks should be dropped
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c_b1"
    assert filtered[0].notice_id == "n_b"

    # Note should state B supersedes A
    assert len(notes) == 1
    assert "Midsem Schedule Revised (2026-10-05) supersedes Midsem Schedule (2026-10-01)" in notes[0]


def test_downrank_superseded_notice_chunks() -> None:
    """When drop=False, superseded notice chunks are demoted with reduced scores."""
    chunk_a = _make_chunk("c_a", "n_a", "Midsem Schedule", "2026-10-01", score=1.0)
    chunk_b = _make_chunk("c_b", "n_b", "Midsem Schedule Revised", "2026-10-05", score=0.8)

    # Notice A was originally ranked higher than Notice B
    chunks = [chunk_a, chunk_b]
    supersedes_map = {"n_b": "n_a", "n_a": None}

    filtered, notes = apply_latest_wins(chunks, drop=False, supersedes_map=supersedes_map)

    # Both chunks kept, but B is now first and A is demoted
    assert len(filtered) == 2
    assert filtered[0].chunk_id == "c_b"
    assert filtered[1].chunk_id == "c_a"
    assert filtered[1].score < filtered[0].score


def test_transitive_supersession_chain() -> None:
    """Notice C supersedes B, and B supersedes A."""
    chunk_a = _make_chunk("c_a", "n_a", "Notice A", "2026-10-01")
    chunk_b = _make_chunk("c_b", "n_b", "Notice B", "2026-10-05")
    chunk_c = _make_chunk("c_c", "n_c", "Notice C", "2026-10-10")

    chunks = [chunk_c, chunk_b, chunk_a]
    supersedes_map = {"n_c": "n_b", "n_b": "n_a", "n_a": None}

    filtered, notes = apply_latest_wins(chunks, drop=True, supersedes_map=supersedes_map)

    # Only C should remain
    assert len(filtered) == 1
    assert filtered[0].notice_id == "n_c"
    assert any("Notice C" in n and "Notice B" in n for n in notes)
    assert any("Notice B" in n and "Notice A" in n for n in notes)


def test_latest_wins_db_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """apply_latest_wins looks up supersedes_id directly from SQLite when no map is provided."""
    monkeypatch.setattr("app.config.settings.data_dir", str(tmp_path))
    init_db()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO notices (id, title, notice_date, category, source_file, file_hash, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("id_old", "Midsem Schedule", "2026-10-01", "exam", "old.pdf", "hash1", None),
        )
        conn.execute(
            """
            INSERT INTO notices (id, title, notice_date, category, source_file, file_hash, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("id_new", "Midsem Schedule Revised", "2026-10-05", "exam", "new.pdf", "hash2", "id_old"),
        )
        conn.commit()

    chunk_old = _make_chunk("c_old", "id_old", "Midsem Schedule", "2026-10-01")
    chunk_new = _make_chunk("c_new", "id_new", "Midsem Schedule Revised", "2026-10-05")

    filtered, notes = apply_latest_wins([chunk_new, chunk_old], drop=True)
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c_new"
    assert len(notes) == 1
    assert "Midsem Schedule Revised" in notes[0]
    assert "Midsem Schedule" in notes[0]
