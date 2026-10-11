"""End-to-end tests for POST /ask endpoint (mocking LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.db import get_connection, init_db
from app.llm.client import BaseLLMClient
from app.main import app
from app.retrieval.retriever import RetrievedChunk


class DummyLLMClient(BaseLLMClient):
    """Predictable mock LLM for testing without external API calls."""

    def __init__(self, answer: str = "Midterm exams begin on October 20 [Midsem Schedule, 2026-10-05, p.1].") -> None:
        self.answer = answer
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.answer


@pytest.fixture(autouse=True)
def _setup_db(tmp_path, monkeypatch):
    """Use a temporary data directory for each test to keep state clean."""
    monkeypatch.setattr("app.config.settings.data_dir", str(tmp_path))
    init_db()


def test_ask_endpoint_success() -> None:
    """POST /ask returns 200 with answer, citations, chunks, and latency."""
    mock_chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="Midterm exams begin on October 20.",
            notice_id="notice_1",
            title="Midsem Schedule",
            notice_date="2026-10-05",
            category="exam",
            page=1,
            chunk_index=0,
            score=0.95,
        )
    ]
    dummy_client = DummyLLMClient()

    with patch("app.rag.retrieve", return_value=mock_chunks), \
         patch("app.rag.get_llm_client", return_value=dummy_client):
        client = TestClient(app)
        response = client.post(
            "/ask",
            json={"question": "When do midterms begin?", "category": "exam"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "chunks" in data
    assert "latency_ms" in data
    assert "config" in data

    assert data["answer"] == dummy_client.answer
    assert len(data["citations"]) == 1
    assert data["citations"][0]["title"] == "Midsem Schedule"
    assert data["citations"][0]["notice_date"] == "2026-10-05"
    assert data["citations"][0]["page"] == 1
    assert len(data["chunks"]) == 1


def test_ask_endpoint_refusal_response() -> None:
    """When LLM returns refusal phrase, /ask serves it without errors."""
    refusal_msg = "I could not find this in the available notices."
    dummy_client = DummyLLMClient(answer=refusal_msg)

    with patch("app.rag.retrieve", return_value=[]), \
         patch("app.rag.get_llm_client", return_value=dummy_client):
        client = TestClient(app)
        response = client.post("/ask", json={"question": "What is the principal's car number?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == refusal_msg


def test_ask_endpoint_empty_question_fails() -> None:
    """POST /ask with empty question returns 422 Unprocessable Entity."""
    client = TestClient(app)
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_endpoint_logs_to_db() -> None:
    """POST /ask persists question and answer into query_logs table."""
    dummy_client = DummyLLMClient(answer="Test answer [Test, 2026-01-01, p.1]")

    with patch("app.rag.retrieve", return_value=[]), \
         patch("app.rag.get_llm_client", return_value=dummy_client):
        client = TestClient(app)
        res = client.post("/ask", json={"question": "Is attendance mandatory?"})
        assert res.status_code == 200

    with get_connection() as conn:
        rows = conn.execute("SELECT question, answer FROM query_logs").fetchall()
        assert len(rows) == 1
        assert rows[0]["question"] == "Is attendance mandatory?"
        assert rows[0]["answer"] == dummy_client.answer


def test_ask_config_override() -> None:
    """POST /ask passes config 'A' (vector-only) or 'B' (hybrid) to answer_question."""
    dummy_client = DummyLLMClient()

    with patch("app.rag.retrieve", return_value=[]) as mock_retrieve, \
         patch("app.rag.get_llm_client", return_value=dummy_client):
        client = TestClient(app)
        res = client.post("/ask", json={"question": "Test config", "config": "A"})
        assert res.status_code == 200
        assert res.json()["config"] == "A"
        # mode should have been mapped to "vector" for config "A"
        mock_retrieve.assert_called_once_with("Test config", category=None, mode="vector")
