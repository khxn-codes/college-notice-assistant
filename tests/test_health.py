"""Tests for the FastAPI health endpoint (Milestone 1)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """GET /health should return HTTP 200 with body {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_content_type_is_json() -> None:
    """GET /health should respond with JSON content-type."""
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
