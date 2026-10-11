"""Unit tests for SHA-256 dedupe logic in FolderConnector."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from app.ingestion.folder_connector import FolderConnector, _sha256


class TestSha256:
    def test_known_content(self, tmp_path: Path) -> None:
        f = tmp_path / "notice.txt"
        f.write_bytes(b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _sha256(f) == expected

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_bytes(b"aaa")
        b.write_bytes(b"bbb")
        assert _sha256(a) != _sha256(b)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_bytes(b"identical content")
        b.write_bytes(b"identical content")
        assert _sha256(a) == _sha256(b)


class TestFolderConnectorDedupe:
    """Test that FolderConnector skips files already in the DB."""

    def test_empty_inbox_returns_empty(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        connector = FolderConnector(inbox_dir=inbox)
        # Patch _known_hashes to return empty set (no DB needed)
        connector._known_hashes = lambda: set()  # type: ignore[method-assign]
        assert connector.fetch_new() == []

    def test_new_file_returned(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        notice = inbox / "2026-10-01_exam_test.pdf"
        notice.write_bytes(b"%PDF-1.4 sample content")

        connector = FolderConnector(inbox_dir=inbox)
        connector._known_hashes = lambda: set()  # type: ignore[method-assign]
        results = connector.fetch_new()
        assert len(results) == 1
        assert results[0].original_filename == "2026-10-01_exam_test.pdf"

    def test_duplicate_file_skipped(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        notice = inbox / "2026-10-01_exam_test.pdf"
        notice.write_bytes(b"some content")
        file_hash = _sha256(notice)

        connector = FolderConnector(inbox_dir=inbox)
        connector._known_hashes = lambda: {file_hash}  # type: ignore[method-assign]
        results = connector.fetch_new()
        assert results == []

    def test_unsupported_extension_ignored(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "image.png").write_bytes(b"\x89PNG")
        (inbox / "doc.docx").write_bytes(b"PK")

        connector = FolderConnector(inbox_dir=inbox)
        connector._known_hashes = lambda: set()  # type: ignore[method-assign]
        assert connector.fetch_new() == []

    def test_mix_new_and_duplicate(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        old = inbox / "old.txt"
        new = inbox / "new.txt"
        old.write_bytes(b"old content")
        new.write_bytes(b"new content")
        old_hash = _sha256(old)

        connector = FolderConnector(inbox_dir=inbox)
        connector._known_hashes = lambda: {old_hash}  # type: ignore[method-assign]
        results = connector.fetch_new()
        assert len(results) == 1
        assert results[0].original_filename == "new.txt"
