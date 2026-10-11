"""End-to-end test for Latest-wins logic using a mocked LLM.

TECH_SPEC §13 Milestone 4:
  - Revision detection links the newer notice to the older one.
  - When asking a question, retrieval drops the superseded notice chunks.
  - Note "Notice B (date) supersedes Notice A (date)" is sent to the LLM.
  - Answer uses the newest notice and mentions that it supersedes the older one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.db import get_connection, init_db
from app.llm.client import BaseLLMClient
from app.main import app
from app.retrieval.retriever import RetrievedChunk


class MockLatestWinsLLMClient(BaseLLMClient):
    """Mock LLM client that records prompt content and returns a realistic supersedes answer."""

    def __init__(self) -> None:
        self.received_system: str | None = None
        self.received_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.received_system = system
        self.received_user = user
        return (
            "Mid-term examinations will commence on October 17 [Midsem Schedule Revised, 2026-10-05, p.1]. "
            "Please note that Midsem Schedule Revised (2026-10-05) supersedes Midsem Schedule (2026-10-01)."
        )


@pytest.fixture(autouse=True)
def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.data_dir", str(tmp_path))
    init_db()


def test_latest_wins_e2e_flow() -> None:
    """Full end-to-end test of revision detection, retrieval filtering, and RAG answer."""
    # 1. Setup notices in database: Notice B supersedes Notice A
    notice_a_id = "notice-a-uuid"
    notice_b_id = "notice-b-uuid"

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO notices (id, title, notice_date, category, source_file, file_hash, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notice_a_id,
                "Midsem Schedule",
                "2026-10-01",
                "exam",
                "2026-10-01_exam_midsem-schedule.pdf",
                "hash_a",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO notices (id, title, notice_date, category, source_file, file_hash, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notice_b_id,
                "Midsem Schedule Revised",
                "2026-10-05",
                "exam",
                "2026-10-05_exam_midsem-schedule-revised.pdf",
                "hash_b",
                notice_a_id,
            ),
        )
        conn.commit()

    # 2. Both notices would have matched the search query
    mock_retrieved_chunks = [
        RetrievedChunk(
            chunk_id="chunk_b",
            text="Midterm exams are rescheduled to October 17.",
            notice_id=notice_b_id,
            title="Midsem Schedule Revised",
            notice_date="2026-10-05",
            category="exam",
            page=1,
            chunk_index=0,
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="chunk_a",
            text="Midterm exams begin on October 10.",
            notice_id=notice_a_id,
            title="Midsem Schedule",
            notice_date="2026-10-01",
            category="exam",
            page=1,
            chunk_index=0,
            score=0.88,
        ),
    ]

    mock_llm = MockLatestWinsLLMClient()

    # 3. Call POST /ask
    with patch("app.rag.retrieve", return_value=mock_retrieved_chunks), \
         patch("app.rag.get_llm_client", return_value=mock_llm):
        client = TestClient(app)
        response = client.post(
            "/ask",
            json={"question": "What is the revised date of the mid-sem exam?", "category": "exam"},
        )

    # 4. Assertions on the response
    assert response.status_code == 200
    data = response.json()

    # Check answer
    assert "October 17" in data["answer"]
    assert "supersedes" in data["answer"]

    # Check that superseded chunk (chunk_a from Notice A) was dropped
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "chunk_b"
    assert data["chunks"][0]["title"] == "Midsem Schedule Revised"

    # Check citations only list the newest notice
    assert len(data["citations"]) == 1
    assert data["citations"][0]["title"] == "Midsem Schedule Revised"
    assert data["citations"][0]["notice_date"] == "2026-10-05"

    # Check notes in response
    assert len(data["notes"]) == 1
    assert "Midsem Schedule Revised (2026-10-05) supersedes Midsem Schedule (2026-10-01)" in data["notes"][0]

    # 5. Check what was sent to the mocked LLM
    assert mock_llm.received_user is not None
    # User message contains revision notes
    assert "Revision Notes:" in mock_llm.received_user
    assert "Midsem Schedule Revised (2026-10-05) supersedes Midsem Schedule (2026-10-01)" in mock_llm.received_user
    # User message contains the newest passage
    assert "Midterm exams are rescheduled to October 17." in mock_llm.received_user
    # User message does NOT contain the outdated passage from Notice A
    assert "Midterm exams begin on October 10." not in mock_llm.received_user
