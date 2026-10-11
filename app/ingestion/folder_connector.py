"""FolderConnector — reads new files from ``data/inbox/``.

Only PDF and TXT files are accepted. Files whose SHA-256 hash already
appears in the ``notices`` table are considered duplicates and skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.db import get_connection
from app.ingestion.connector import Connector, RawNotice

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class FolderConnector(Connector):
    """Reads all un-ingested PDF/TXT files from the inbox directory."""

    def __init__(self, inbox_dir: Path | None = None) -> None:
        self._inbox = inbox_dir or (Path(settings.data_dir) / "inbox")

    def fetch_new(self) -> list[RawNotice]:
        """Return files in inbox that are not already hashed in the DB.

        Returns:
            List of :class:`RawNotice` for files ready to ingest.
        """
        self._inbox.mkdir(parents=True, exist_ok=True)
        candidates = [
            p for p in sorted(self._inbox.iterdir())
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
        ]
        if not candidates:
            logger.info("Inbox %s is empty — nothing to ingest.", self._inbox)
            return []

        existing_hashes = self._known_hashes()
        result: list[RawNotice] = []
        for path in candidates:
            file_hash = _sha256(path)
            if file_hash in existing_hashes:
                logger.info("Skipping duplicate: %s (hash already indexed)", path.name)
                continue
            result.append(RawNotice(path=path, original_filename=path.name))

        logger.info(
            "FolderConnector: %d new file(s) out of %d found in inbox.",
            len(result),
            len(candidates),
        )
        return result

    @staticmethod
    def _known_hashes() -> set[str]:
        """Return all file_hash values already present in the notices table."""
        with get_connection() as conn:
            rows = conn.execute("SELECT file_hash FROM notices WHERE file_hash IS NOT NULL").fetchall()
        return {row["file_hash"] for row in rows}


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
