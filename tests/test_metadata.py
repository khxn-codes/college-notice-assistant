"""Unit tests for app/ingestion/metadata.py — filename parsing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.ingestion.metadata import parse_metadata


class TestConventionFilenames:
    """Files that follow the YYYY-MM-DD_category_slug convention."""

    def test_full_convention(self) -> None:
        meta = parse_metadata(Path("2026-10-05_exam_midsem-schedule.pdf"))
        assert meta.notice_date == date(2026, 10, 5)
        assert meta.category == "exam"
        assert meta.title == "Midsem Schedule"
        assert meta.source_file == "2026-10-05_exam_midsem-schedule.pdf"

    def test_category_preserved_lowercase(self) -> None:
        meta = parse_metadata(Path("2026-09-01_fees_tuition-payment.pdf"))
        assert meta.category == "fees"

    def test_category_any_word(self) -> None:
        """Category is not restricted to a fixed list."""
        meta = parse_metadata(Path("2026-08-15_hostel_mess-menu.txt"))
        assert meta.category == "hostel"

    def test_slug_with_multiple_hyphens(self) -> None:
        meta = parse_metadata(Path("2026-10-05_general_end-sem-revised-timetable.pdf"))
        assert meta.title == "End Sem Revised Timetable"

    def test_notice_date_correct(self) -> None:
        meta = parse_metadata(Path("2025-01-31_events_annual-day.pdf"))
        assert meta.notice_date == date(2025, 1, 31)


class TestFallbackFilenames:
    """Files that do NOT follow the convention."""

    def test_plain_name_falls_back_to_today(self) -> None:
        meta = parse_metadata(Path("SomeRandomNotice.pdf"))
        assert meta.notice_date == date.today()
        assert meta.category == "general"

    def test_partial_date_falls_back(self) -> None:
        """A name that starts with a date but lacks category slug is not matched."""
        meta = parse_metadata(Path("2026-10-05.pdf"))
        assert meta.category == "general"

    def test_title_cleaned_from_stem(self) -> None:
        meta = parse_metadata(Path("hostel_notice_2026.pdf"))
        assert meta.title == "Hostel Notice 2026"

    def test_source_file_always_set(self) -> None:
        meta = parse_metadata(Path("notice.pdf"))
        assert meta.source_file == "notice.pdf"
