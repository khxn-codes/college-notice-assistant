"""Unit tests for notice revision detection.

Covers:
  - Title normalization & stripping of keywords ('revised', 'extended', 'rescheduled').
  - Detection when title contains revision keywords.
  - Detection when filename contains revision keywords.
  - Category matching and date precedence (earlier date required).
  - Selection of the most recent older notice when multiple candidates exist.
  - Database-backed revision detection during notice ingestion.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest

from app.db import get_connection, init_db
from app.ingestion.revisions import (
    are_titles_or_filenames_revisions,
    clean_slug_or_filename,
    find_superseded_notice,
    has_revision_keyword,
    strip_revision_words,
)


def test_has_revision_keyword() -> None:
    assert has_revision_keyword("Midsem Schedule Revised")
    assert has_revision_keyword("hostel-fee-payment-extended.pdf")
    assert has_revision_keyword("exam_rescheduled_notice")
    assert not has_revision_keyword("Midsem Schedule")
    assert not has_revision_keyword("Hostel Fee Payment")


def test_strip_revision_words() -> None:
    assert strip_revision_words("Midsem Schedule Revised") == "midsem schedule"
    assert strip_revision_words("Midsem Exam Rescheduled") == "midsem exam"
    assert strip_revision_words("Hostel Fee Payment Extended") == "hostel fee payment"
    assert strip_revision_words("General Rules") == "general rules"


def test_clean_slug_or_filename() -> None:
    assert clean_slug_or_filename("2026-10-05_exam_midsem-schedule-revised.pdf") == "midsem schedule"
    assert clean_slug_or_filename("2026-10-01_exam_midsem-schedule.pdf") == "midsem schedule"
    assert clean_slug_or_filename("midsem-extended.txt") == "midsem"


def test_revision_match_by_title_keyword() -> None:
    """Titles match after removing 'revised'."""
    assert are_titles_or_filenames_revisions(
        new_title="Midsem Schedule Revised",
        new_filename="2026-10-05_exam_midsem-schedule-revised.pdf",
        old_title="Midsem Schedule",
        old_filename="2026-10-01_exam_midsem-schedule.pdf",
    )


def test_revision_match_by_filename_keyword() -> None:
    """Filename contains 'revised' and slug matches older notice slug."""
    assert are_titles_or_filenames_revisions(
        new_title="Midsem Revised",
        new_filename="2026-10-05_exam_midsem-revised.pdf",
        old_title="Midsem Schedule",
        old_filename="2026-10-01_exam_midsem-schedule.pdf",
    )


def test_revision_match_extended_and_rescheduled() -> None:
    assert are_titles_or_filenames_revisions(
        new_title="Hostel Fee Payment Extended",
        new_filename="2026-10-15_fees_hostel-fee-payment-extended.pdf",
        old_title="Hostel Fee Payment",
        old_filename="2026-10-01_fees_hostel-fee-payment.pdf",
    )
    assert are_titles_or_filenames_revisions(
        new_title="End Semester Exam Rescheduled",
        new_filename="2026-11-05_exam_end-sem-rescheduled.pdf",
        old_title="End Semester Exam",
        old_filename="2026-10-20_exam_end-sem.pdf",
    )


def test_revision_non_match_unrelated_titles() -> None:
    """Unrelated titles should not match as revisions."""
    assert not are_titles_or_filenames_revisions(
        new_title="Library Timings Revised",
        new_filename="2026-10-05_general_library-timings-revised.pdf",
        old_title="Hostel Rules",
        old_filename="2026-10-01_general_hostel-rules.pdf",
    )


def test_find_superseded_notice_matches_older_same_category() -> None:
    existing = [
        {
            "id": "old-1",
            "title": "Midsem Schedule",
            "notice_date": "2026-10-01",
            "category": "exam",
            "source_file": "2026-10-01_exam_midsem-schedule.pdf",
        }
    ]
    res = find_superseded_notice(
        title="Midsem Schedule Revised",
        notice_date="2026-10-05",
        category="exam",
        source_file="2026-10-05_exam_midsem-schedule-revised.pdf",
        existing_notices=existing,
    )
    assert res == "old-1"


def test_find_superseded_notice_rejects_different_category() -> None:
    existing = [
        {
            "id": "old-1",
            "title": "Fee Payment",
            "notice_date": "2026-10-01",
            "category": "fees",
            "source_file": "2026-10-01_fees_fee-payment.pdf",
        }
    ]
    # New notice is 'exam', old notice is 'fees'
    res = find_superseded_notice(
        title="Fee Payment Extended",
        notice_date="2026-10-05",
        category="exam",
        source_file="2026-10-05_exam_fee-payment-extended.pdf",
        existing_notices=existing,
    )
    assert res is None


def test_find_superseded_notice_rejects_newer_or_equal_date() -> None:
    existing = [
        {
            "id": "old-1",
            "title": "Midsem Schedule",
            "notice_date": "2026-10-10",
            "category": "exam",
            "source_file": "2026-10-10_exam_midsem-schedule.pdf",
        }
    ]
    # New notice is 2026-10-05, candidate date is 2026-10-10 (later!)
    res = find_superseded_notice(
        title="Midsem Schedule Revised",
        notice_date="2026-10-05",
        category="exam",
        source_file="2026-10-05_exam_midsem-schedule-revised.pdf",
        existing_notices=existing,
    )
    assert res is None


def test_find_superseded_notice_chooses_most_recent_candidate() -> None:
    """When multiple older matching notices exist, picks the most recent one."""
    existing = [
        {
            "id": "notice-v1",
            "title": "Midsem Schedule",
            "notice_date": "2026-10-01",
            "category": "exam",
            "source_file": "2026-10-01_exam_midsem-schedule.pdf",
        },
        {
            "id": "notice-v2",
            "title": "Midsem Schedule Revised",
            "notice_date": "2026-10-08",
            "category": "exam",
            "source_file": "2026-10-08_exam_midsem-schedule-revised.pdf",
        },
    ]
    res = find_superseded_notice(
        title="Midsem Schedule Extended",
        notice_date="2026-10-15",
        category="exam",
        source_file="2026-10-15_exam_midsem-schedule-extended.pdf",
        existing_notices=existing,
    )
    assert res == "notice-v2"


def test_ingestion_sets_supersedes_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingesting an older notice then a revised notice automatically links supersedes_id in DB."""
    from unittest.mock import patch
    from app.ingestion.pipeline import ingest_file

    monkeypatch.setattr("app.config.settings.data_dir", str(tmp_path))
    init_db()

    file_v1 = tmp_path / "2026-10-01_exam_midsem-schedule.txt"
    file_v1.write_text("Midterm exams begin on October 10.", encoding="utf-8")

    file_v2 = tmp_path / "2026-10-05_exam_midsem-schedule-revised.txt"
    file_v2.write_text("Midterm exams rescheduled to October 17.", encoding="utf-8")

    with patch("app.ingestion.pipeline.index_chunks"):
        res1 = ingest_file(file_v1)
        assert res1["status"] == "indexed"
        id1 = res1["notice_id"]
        assert res1["supersedes_id"] is None

        res2 = ingest_file(file_v2)
        assert res2["status"] == "indexed"
        id2 = res2["notice_id"]
        assert res2["supersedes_id"] == id1

    with get_connection() as conn:
        row2 = conn.execute("SELECT supersedes_id FROM notices WHERE id = ?", (id2,)).fetchone()
        assert row2["supersedes_id"] == id1
