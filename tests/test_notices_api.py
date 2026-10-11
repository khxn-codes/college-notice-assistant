"""Tests for notice management endpoints including PATCH /notices/{id} manual override."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.db import get_connection, init_db
from app.main import app


@pytest.fixture(autouse=True)
def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.data_dir", str(tmp_path))
    init_db()


def _insert_notice(
    notice_id: str,
    title: str = "Test Notice",
    notice_date: str = "2026-10-01",
    category: str = "exam",
    supersedes_id: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO notices (id, title, notice_date, category, source_file, file_hash, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (notice_id, title, notice_date, category, f"{notice_id}.pdf", f"hash_{notice_id}", supersedes_id),
        )
        conn.commit()


def test_patch_notice_success_supersedes_id() -> None:
    """PATCH /notices/{id} manually links a notice to an older one."""
    _insert_notice("n1", title="Midsem Schedule", notice_date="2026-10-01")
    _insert_notice("n2", title="Midsem Schedule Revised", notice_date="2026-10-05")

    client = TestClient(app)
    response = client.patch("/notices/n2", json={"supersedes_id": "n1"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert data["notice"]["id"] == "n2"
    assert data["notice"]["supersedes_id"] == "n1"

    # Verify SQLite update
    with get_connection() as conn:
        row = conn.execute("SELECT supersedes_id FROM notices WHERE id = 'n2'").fetchone()
        assert row["supersedes_id"] == "n1"


def test_patch_notice_clear_supersedes_id() -> None:
    """Setting supersedes_id to None or null clears the override."""
    _insert_notice("n1", title="Midsem Schedule", notice_date="2026-10-01")
    _insert_notice("n2", title="Midsem Schedule Revised", notice_date="2026-10-05", supersedes_id="n1")

    client = TestClient(app)
    response = client.patch("/notices/n2", json={"supersedes_id": None})

    assert response.status_code == 200
    assert response.json()["notice"]["supersedes_id"] is None

    with get_connection() as conn:
        row = conn.execute("SELECT supersedes_id FROM notices WHERE id = 'n2'").fetchone()
        assert row["supersedes_id"] is None


def test_patch_notice_update_metadata_fields() -> None:
    """PATCH updates title, notice_date, and category."""
    _insert_notice("n1", title="Old Title", notice_date="2026-10-01", category="exam")

    with patch("app.main.rebuild_bm25_index"), patch("app.main.update_notice_chunks_metadata"):
        client = TestClient(app)
        response = client.patch(
            "/notices/n1",
            json={
                "title": "New Title",
                "notice_date": "2026-10-15",
                "category": "fees",
            },
        )

    assert response.status_code == 200
    data = response.json()["notice"]
    assert data["title"] == "New Title"
    assert data["notice_date"] == "2026-10-15"
    assert data["category"] == "fees"


def test_patch_notice_not_found() -> None:
    client = TestClient(app)
    response = client.patch("/notices/non-existent-id", json={"title": "Hello"})
    assert response.status_code == 404


def test_patch_notice_invalid_supersedes_target() -> None:
    """Cannot set supersedes_id to a notice ID that does not exist in the database."""
    _insert_notice("n1")
    client = TestClient(app)
    response = client.patch("/notices/n1", json={"supersedes_id": "ghost-id"})
    assert response.status_code == 400


def test_patch_notice_self_supersede_rejected() -> None:
    """A notice cannot supersede itself."""
    _insert_notice("n1")
    client = TestClient(app)
    response = client.patch("/notices/n1", json={"supersedes_id": "n1"})
    assert response.status_code == 400


def test_patch_notice_invalid_date_format() -> None:
    _insert_notice("n1")
    client = TestClient(app)
    response = client.patch("/notices/n1", json={"notice_date": "15-10-2026"})
    assert response.status_code == 422
