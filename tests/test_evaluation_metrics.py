"""Unit tests for evaluation metrics and runner using fake results."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.evaluation.metrics import (
    STANDARD_REFUSAL,
    check_answer_accuracy,
    check_citation_hit,
    check_refusal_accuracy,
    compute_metrics,
)
from app.main import app


def test_check_answer_accuracy_matches() -> None:
    """Case-insensitive matching of all substrings."""
    answer = "The revised date for the exam is Monday, 17 October 2026."
    assert check_answer_accuracy(answer, ["17 October", "monday"]) is True
    assert check_answer_accuracy(answer, ["17 OCTOBER", "EXAM"]) is True


def test_check_answer_accuracy_missing_keyword() -> None:
    """Fails when at least one expected substring is missing."""
    answer = "The exam will take place on 17 October."
    assert check_answer_accuracy(answer, ["17 October", "November"]) is False


def test_check_answer_accuracy_empty_expected() -> None:
    """Empty expected list evaluates to True."""
    assert check_answer_accuracy("Any answer text", []) is True


def test_check_citation_hit_direct_and_mapped() -> None:
    """Matches via direct source_file or mapped notice_id."""
    citations = [
        {
            "title": "Class Timings Revised",
            "notice_id": "nid_123",
            "notice_date": "2026-09-18",
            "page": 1,
            "source_file": "2026-09-18_admin_class-timings-revised.pdf",
        }
    ]
    # Direct match
    assert check_citation_hit("2026-09-18_admin_class-timings-revised.pdf", citations) is True

    # Mapped match via notice_id
    cits_without_src = [
        {
            "title": "Timings",
            "notice_id": "nid_456",
            "page": 1,
        }
    ]
    source_map = {"nid_456": "2026-09-18_admin_class-timings-revised.pdf"}
    assert check_citation_hit("2026-09-18_admin_class-timings-revised.pdf", cits_without_src, source_map) is True


def test_check_citation_hit_title_stem_match() -> None:
    """Matches when citation title correlates with source filename stem."""
    citations = [{"title": "Class Timings Revised", "notice_id": "x", "page": 1}]
    assert check_citation_hit("2026-09-18_admin_class-timings-revised.pdf", citations) is True


def test_check_citation_hit_miss() -> None:
    """Returns False when expected source is not in citations."""
    citations = [{"title": "NCC Recruitment", "source_file": "2026-09-21_events_ncc-recruitment.pdf"}]
    assert check_citation_hit("2026-09-18_admin_class-timings-revised.pdf", citations) is False
    assert check_citation_hit(None, citations) is False


def test_check_refusal_accuracy() -> None:
    """Exact string match required for refusal."""
    assert check_refusal_accuracy(STANDARD_REFUSAL) is True
    assert check_refusal_accuracy(f"  {STANDARD_REFUSAL}  \n") is True
    assert check_refusal_accuracy("I don't know the answer.") is False
    assert check_refusal_accuracy("I could not find this in the available notices. Here is extra guess:") is False


def test_compute_metrics_with_synthetic_results() -> None:
    """Aggregates accuracy, citation hit rate, refusal accuracy, and latency."""
    test_cases = [
        {
            "id": "q01",
            "question": "What is the revised date?",
            "expected_answer_contains": ["17 October"],
            "expected_source": "exam-revised.pdf",
            "answerable": True,
        },
        {
            "id": "q02",
            "question": "Where is the event?",
            "expected_answer_contains": ["Auditorium"],
            "expected_source": "event.pdf",
            "answerable": True,
        },
        {
            "id": "q03",
            "question": "What is the Wi-Fi password?",
            "expected_answer_contains": [],
            "expected_source": None,
            "answerable": False,
        },
        {
            "id": "q04",
            "question": "What is the principal's car number?",
            "expected_answer_contains": [],
            "expected_source": None,
            "answerable": False,
        },
    ]

    fake_results = [
        # q01: answerable PASS (both keyword and citation hit)
        {
            "answer": "The revised date is 17 October.",
            "citations": [{"source_file": "exam-revised.pdf", "title": "Exam Revised"}],
            "latency_ms": 100,
        },
        # q02: answerable FAIL (missing keyword 'Auditorium')
        {
            "answer": "The event is held in Main Hall.",
            "citations": [{"source_file": "event.pdf", "title": "Event"}],
            "latency_ms": 200,
        },
        # q03: unanswerable PASS (exact refusal)
        {
            "answer": STANDARD_REFUSAL,
            "citations": [],
            "latency_ms": 150,
        },
        # q04: unanswerable FAIL (hallucinated instead of refusal)
        {
            "answer": "Car number is UP78 AB 1234.",
            "citations": [],
            "latency_ms": 250,
        },
    ]

    metrics = compute_metrics(test_cases, fake_results)

    # 1 of 2 answerable passed keyword check -> 0.5
    assert metrics["answer_accuracy"] == 0.5
    # 2 of 2 answerable had citations matching their source -> 1.0
    assert metrics["citation_hit_rate"] == 1.0
    # 1 of 2 unanswerable correctly refused -> 0.5
    assert metrics["refusal_accuracy"] == 0.5
    # Average latency: (100 + 200 + 150 + 250) / 4 = 175.0 ms
    assert metrics["average_latency_ms"] == 175.0

    assert metrics["total_questions"] == 4
    assert metrics["passed_count"] == 2
    assert metrics["failed_count"] == 2


def test_compute_metrics_empty() -> None:
    """Safe division and zero defaults when no questions provided."""
    metrics = compute_metrics([], [])
    assert metrics["answer_accuracy"] == 0.0
    assert metrics["citation_hit_rate"] == 0.0
    assert metrics["refusal_accuracy"] == 0.0
    assert metrics["average_latency_ms"] == 0.0
    assert metrics["total_questions"] == 0


def test_compute_metrics_handles_error_gracefully() -> None:
    """Error in a result records API error without crashing or counting as wrong answer."""
    test_cases = [
        {
            "id": "q01",
            "question": "What is the date?",
            "expected_answer_contains": ["October"],
            "expected_source": "exam.pdf",
            "answerable": True,
        }
    ]
    error_results = [
        {
            "answer": "API Error: Timeout",
            "citations": [],
            "latency_ms": 0,
            "error": "Timeout",
        }
    ]

    metrics = compute_metrics(test_cases, error_results)
    assert metrics["answer_accuracy"] == 0.0
    assert metrics["passed_count"] == 0
    assert metrics["failed_count"] == 0
    assert metrics["api_errors"] == 1
    assert "API Error: Timeout" in metrics["question_results"][0]["failure_reasons"]


def test_eval_endpoints_with_mock_runner(tmp_path: Path) -> None:
    """Test POST /eval/run and GET /eval/results FastAPI endpoints."""
    fake_run = {
        "run_id": "eval_test_001",
        "timestamp": "2026-10-11T23:59:59",
        "total_test_cases": 1,
        "configs": {
            "A": {
                "config_name": "A",
                "metrics": {
                    "answer_accuracy": 1.0,
                    "citation_hit_rate": 1.0,
                    "refusal_accuracy": 1.0,
                    "average_latency_ms": 120.0,
                    "total_questions": 1,
                    "passed_count": 1,
                    "failed_count": 0,
                },
                "per_question": [],
            }
        },
    }

    results_dir = tmp_path / "eval" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "eval_test_001.json").write_text(json.dumps(fake_run), encoding="utf-8")

    with patch("app.evaluation.runner.run_evaluation", return_value=fake_run):
        client = TestClient(app)

        # POST /eval/run
        post_res = client.post("/eval/run", json={"configs": ["A"]})
        assert post_res.status_code == 200
        assert post_res.json()["run_id"] == "eval_test_001"

    # GET /eval/results using mocked EVAL_RESULTS_DIR
    with patch("app.main.EVAL_RESULTS_DIR", results_dir):
        client = TestClient(app)
        get_res = client.get("/eval/results")
        assert get_res.status_code == 200
        data = get_res.json()
        assert "runs" in data
        assert data["count"] == 1
        assert data["runs"][0]["run_id"] == "eval_test_001"
