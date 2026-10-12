"""Tests for evaluation runner, config switching, and error handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.db import init_db
from app.evaluation.runner import load_test_set, run_evaluation
from app.llm.client import BaseLLMClient


class MockLLM(BaseLLMClient):
    """Predictable mock LLM for evaluation tests."""

    def __init__(self, answer: str = "Test answer on 17 October.") -> None:
        self.answer = answer

    def complete(self, system: str, user: str) -> str:
        return self.answer


class FailingLLM(BaseLLMClient):
    """Mock LLM that raises an exception on specific calls."""

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("LLM API rate limit exceeded")


@pytest.fixture(autouse=True)
def _setup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.data_dir", str(tmp_path))
    init_db()


def test_load_test_set(tmp_path: Path) -> None:
    """Test loading test cases from JSON file."""
    test_file = tmp_path / "test.json"
    sample = [{"id": "q01", "question": "Sample?", "answerable": True}]
    test_file.write_text(json.dumps(sample), encoding="utf-8")

    cases = load_test_set(test_file)
    assert len(cases) == 1
    assert cases[0]["id"] == "q01"


def test_run_evaluation_handles_api_errors_without_crashing(tmp_path: Path) -> None:
    """Runner catches per-question exceptions, marks them as error, excludes from accuracy, and completes full run."""
    sample_test_set = [
        {
            "id": "q01",
            "question": "What is the date?",
            "expected_answer_contains": ["October"],
            "expected_source": "notice.pdf",
            "answerable": True,
        },
        {
            "id": "q02",
            "question": "Unanswerable question?",
            "expected_answer_contains": [],
            "expected_source": None,
            "answerable": False,
        },
    ]
    test_set_file = tmp_path / "test_set.json"
    test_set_file.write_text(json.dumps(sample_test_set), encoding="utf-8")
    results_dir = tmp_path / "results"

    failing_client = FailingLLM()

    with patch("app.ingestion.indexer.reindex_collection_for_config", return_value=0), \
         patch("app.rag.retrieve", return_value=[]):
        report = run_evaluation(
            config_names=["A"],
            delay_seconds=0.0,
            test_set_path=test_set_file,
            results_dir=results_dir,
            llm_client=failing_client,
        )

    assert "run_id" in report
    assert "configs" in report
    assert "A" in report["configs"]
    cfg_a = report["configs"]["A"]

    assert cfg_a["metrics"]["total_questions"] == 2
    assert cfg_a["metrics"]["api_errors"] == 2
    assert cfg_a["metrics"]["passed_count"] == 0
    assert cfg_a["metrics"]["failed_count"] == 0
    # Questions excluded from accuracy metrics
    assert cfg_a["metrics"]["answerable_count"] == 0
    assert cfg_a["metrics"]["unanswerable_count"] == 0

    # Verify per-question status is ERROR
    for q in cfg_a["per_question"]:
        assert q["status"] == "ERROR"
        assert q["is_error"] is True

    # Verify JSON and CSV files were created
    saved_json = list(results_dir.glob("eval_*.json"))
    assert len(saved_json) == 1
    saved_csv = list(results_dir.glob("eval_*.csv"))
    assert len(saved_csv) == 1
    csv_text = saved_csv[0].read_text(encoding="utf-8")
    assert "API Errors" in csv_text


class MockReturns429Client(BaseLLMClient):
    """Mock LLM client that returns 429 response message."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, system: str, user: str) -> str:
        self.call_count += 1
        return "429 Too Many Requests: rate limit exceeded"


class Mock429ThenSuccessClient(BaseLLMClient):
    """Mock LLM client that returns 429 twice, then succeeds on 3rd attempt."""

    def __init__(self, success_ans: str = "Date is 17 October.") -> None:
        self.call_count = 0
        self.success_ans = success_ans

    def complete(self, system: str, user: str) -> str:
        self.call_count += 1
        if self.call_count <= 2:
            return "429 Too Many Requests: rate limit exceeded"
        return self.success_ans


def test_runner_retries_429_exponential_backoff_and_succeeds(tmp_path: Path) -> None:
    """Runner retries on 429 up to 3 times; if it recovers, question passes."""
    sample_test_set = [
        {
            "id": "q01",
            "question": "When is exam?",
            "expected_answer_contains": ["17 October"],
            "expected_source": "exam.pdf",
            "answerable": True,
        }
    ]
    test_set_file = tmp_path / "test_set.json"
    test_set_file.write_text(json.dumps(sample_test_set), encoding="utf-8")
    results_dir = tmp_path / "results"

    recovering_client = Mock429ThenSuccessClient(success_ans="Exam is on 17 October.")

    from app.retrieval.retriever import RetrievedChunk
    mock_chunk = RetrievedChunk(
        chunk_id="c1",
        text="Date is 17 October",
        notice_id="n1",
        title="Exam Notice",
        notice_date="2026-10-05",
        category="exam",
        page=1,
        chunk_index=0,
        score=0.9,
    )

    with patch("app.ingestion.indexer.reindex_collection_for_config", return_value=0), \
         patch("app.rag.retrieve", return_value=[mock_chunk]):
        report = run_evaluation(
            config_names=["A"],
            delay_seconds=0.0,
            test_set_path=test_set_file,
            results_dir=results_dir,
            llm_client=recovering_client,
        )

    # 2 retries + 1 initial = 3 calls
    assert recovering_client.call_count == 3
    cfg_a = report["configs"]["A"]
    assert cfg_a["metrics"]["passed_count"] == 1
    assert cfg_a["metrics"]["api_errors"] == 0
    assert cfg_a["per_question"][0]["status"] == "PASS"


def test_runner_persistent_429_marks_error_and_excludes_from_metrics(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Mock client returning persistent 429 retries 3 times, marks as ERROR, logs chunk count, and reports API errors."""
    import logging
    sample_test_set = [
        {
            "id": "q_429",
            "question": "What is the fee amount?",
            "expected_answer_contains": ["5000"],
            "expected_source": "fee.pdf",
            "answerable": True,
        }
    ]
    test_set_file = tmp_path / "test_set.json"
    test_set_file.write_text(json.dumps(sample_test_set), encoding="utf-8")
    results_dir = tmp_path / "results"

    client_429 = MockReturns429Client()

    with patch("app.ingestion.indexer.reindex_collection_for_config", return_value=0), \
         patch("app.rag.retrieve", return_value=[]), \
         caplog.at_level(logging.INFO):
        report = run_evaluation(
            config_names=["A"],
            delay_seconds=0.0,
            test_set_path=test_set_file,
            results_dir=results_dir,
            llm_client=client_429,
        )

    # 1 initial + 3 retries = 4 attempts total
    assert client_429.call_count == 4
    cfg_a = report["configs"]["A"]
    assert cfg_a["metrics"]["api_errors"] == 1
    assert cfg_a["metrics"]["passed_count"] == 0
    assert cfg_a["metrics"]["failed_count"] == 0
    # Excluded from accuracy
    assert cfg_a["metrics"]["answerable_count"] == 0
    assert cfg_a["per_question"][0]["status"] == "ERROR"

    # Verify requirement (6): logs the number of retrieved chunks per question
    assert any("retrieved 0 chunks" in record.message for record in caplog.records)


def test_runner_rerun_failed_only_option(tmp_path: Path) -> None:
    """Re-run failed/error questions only option filters to questions from prior run."""
    sample_test_set = [
        {
            "id": "q01",
            "question": "Question 1?",
            "expected_answer_contains": ["Answer 1"],
            "answerable": True,
        },
        {
            "id": "q02",
            "question": "Question 2?",
            "expected_answer_contains": ["Answer 2"],
            "answerable": True,
        },
    ]
    test_set_file = tmp_path / "test_set.json"
    test_set_file.write_text(json.dumps(sample_test_set), encoding="utf-8")
    results_dir = tmp_path / "results"

    # Run 1: q01 fails, q02 passes
    class MixedLLM(BaseLLMClient):
        def complete(self, system: str, user: str) -> str:
            if "Question 1" in user:
                return "Wrong answer"
            return "Answer 2 is correct"

    with patch("app.ingestion.indexer.reindex_collection_for_config", return_value=0), \
         patch("app.rag.retrieve", return_value=[]):
        first_report = run_evaluation(
            config_names=["A"],
            delay_seconds=0.0,
            test_set_path=test_set_file,
            results_dir=results_dir,
            llm_client=MixedLLM(),
        )

    assert first_report["configs"]["A"]["metrics"]["total_questions"] == 2
    assert first_report["configs"]["A"]["metrics"]["failed_count"] == 1
    assert first_report["configs"]["A"]["metrics"]["passed_count"] == 1

    # Run 2: Re-run failed/error only
    class PassAllLLM(BaseLLMClient):
        def complete(self, system: str, user: str) -> str:
            return "Answer 1 is now correct"

    with patch("app.ingestion.indexer.reindex_collection_for_config", return_value=0), \
         patch("app.rag.retrieve", return_value=[]):
        second_report = run_evaluation(
            config_names=["A"],
            delay_seconds=0.0,
            test_set_path=test_set_file,
            results_dir=results_dir,
            llm_client=PassAllLLM(),
            rerun_failed_only=True,
            previous_run_id=first_report["run_id"],
        )

    # Only q01 should have been re-evaluated
    assert second_report["total_test_cases"] == 1
    assert second_report["configs"]["A"]["metrics"]["total_questions"] == 1
    assert second_report["configs"]["A"]["per_question"][0]["id"] == "q01"
    assert second_report["configs"]["A"]["metrics"]["passed_count"] == 1


def test_classify_error_causes() -> None:
    """Verify classify_error_cause identifies rate limit, quota, model not found, auth."""
    from app.llm.client import classify_error_cause

    assert classify_error_cause("429 Too Many Requests") == "rate limit"
    assert classify_error_cause("Rate limit exceeded") == "rate limit"
    assert classify_error_cause("Quota exceeded for quota metric") == "quota"
    assert classify_error_cause("ResourceExhausted: check quota") == "quota"
    assert classify_error_cause("404 Model not found: models/gemini-pro") == "model not found"
    assert classify_error_cause("401 Unauthorized API key not valid") == "auth"
    assert classify_error_cause("PermissionDenied") == "auth"


def test_run_evaluation_success_with_mock_llm(tmp_path: Path) -> None:
    """Runner computes positive metrics with predictable mock responses."""
    sample_test_set = [
        {
            "id": "q01",
            "question": "What is the date?",
            "expected_answer_contains": ["17 October"],
            "expected_source": "2026-10-05_exam_midsem.pdf",
            "answerable": True,
        }
    ]
    test_set_file = tmp_path / "test_set.json"
    test_set_file.write_text(json.dumps(sample_test_set), encoding="utf-8")
    results_dir = tmp_path / "results"

    mock_client = MockLLM(answer="The revised date is 17 October.")

    from app.retrieval.retriever import RetrievedChunk
    mock_chunk = RetrievedChunk(
        chunk_id="c1",
        text="Date is 17 October",
        notice_id="n1",
        title="Exam Midsem",
        notice_date="2026-10-05",
        category="exam",
        page=1,
        chunk_index=0,
        score=0.9,
    )

    with patch("app.ingestion.indexer.reindex_collection_for_config", return_value=1), \
         patch("app.rag.retrieve", return_value=[mock_chunk]):
        report = run_evaluation(
            config_names=["A", "B"],
            delay_seconds=0.0,
            test_set_path=test_set_file,
            results_dir=results_dir,
            llm_client=mock_client,
        )

    assert "A" in report["configs"]
    assert "B" in report["configs"]
    assert report["configs"]["A"]["metrics"]["answer_accuracy"] == 1.0
    assert report["configs"]["B"]["metrics"]["answer_accuracy"] == 1.0
