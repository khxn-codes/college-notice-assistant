"""Unit tests for RAG prompt templates and message building."""

from __future__ import annotations

import pytest

from app.llm.prompts import SYSTEM_PROMPT, build_user_message


def test_system_prompt_contains_exact_refusal_phrase() -> None:
    """The system prompt must specify the exact refusal wording from TECH_SPEC §7."""
    assert "I could not find this in the available notices." in SYSTEM_PROMPT


def test_system_prompt_contains_citation_format() -> None:
    """Citation format [Title, Date, p.N] must be required in the prompt."""
    assert "[Title, Date, p.N]" in SYSTEM_PROMPT


def test_system_prompt_restricts_outside_knowledge() -> None:
    """Prompt must explicitly forbid answering from outside knowledge."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "only" in prompt_lower
    assert "passages" in prompt_lower


def test_build_user_message_empty_passages() -> None:
    """When no passages are retrieved, user message mentions no relevant passages."""
    msg = build_user_message("When are exams?", [])
    assert "(No relevant passages were found.)" in msg
    assert "Question: When are exams?" in msg


def test_build_user_message_formats_passages_correctly() -> None:
    """Passages must be numbered and formatted with [N] Title | Date | p.Page."""
    passages = [
        {
            "index": 1,
            "title": "Exam Schedule",
            "notice_date": "2026-10-05",
            "page": 2,
            "text": "Mid-term exams begin on October 20.",
        },
        {
            "index": 2,
            "title": "Fee Notice",
            "notice_date": "2026-10-01",
            "page": 1,
            "text": "Tuition fees are due by October 15.",
        },
    ]
    msg = build_user_message("When are midterms?", passages)

    assert "[1] Exam Schedule | 2026-10-05 | p.2" in msg
    assert "Mid-term exams begin on October 20." in msg
    assert "[2] Fee Notice | 2026-10-01 | p.1" in msg
    assert "Tuition fees are due by October 15." in msg
    assert "Question: When are midterms?" in msg
