"""SQLite database initialisation and helpers.

Creates the database file at ``{DATA_DIR}/app.db`` and sets up the
``notices`` and ``query_logs`` tables defined in TECH_SPEC section 4.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_NOTICES = """
CREATE TABLE IF NOT EXISTS notices (
    id            TEXT PRIMARY KEY,          -- UUID
    title         TEXT NOT NULL,
    notice_date   DATE,                      -- ISO-8601; fallback = upload date
    category      TEXT,                      -- exam | fees | hostel | events | general
    source_file   TEXT,                      -- original filename
    file_hash     TEXT UNIQUE,               -- SHA-256 for dedupe
    supersedes_id TEXT REFERENCES notices(id),  -- nullable
    indexed_at    DATETIME NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_QUERY_LOGS = """
CREATE TABLE IF NOT EXISTS query_logs (
    id                  TEXT PRIMARY KEY,    -- UUID
    question            TEXT NOT NULL,
    answer              TEXT,
    retrieved_chunk_ids TEXT,               -- JSON array of chunk IDs
    config_name         TEXT,
    latency_ms          INTEGER,
    created_at          DATETIME NOT NULL DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_db_path() -> Path:
    """Return the absolute path to the SQLite database file."""
    db_path = Path(settings.data_dir) / "app.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """Open and return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Create all tables if they do not already exist.

    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    logger.info("Initialising database at %s", get_db_path())
    with get_connection() as conn:
        conn.execute(_CREATE_NOTICES)
        conn.execute(_CREATE_QUERY_LOGS)
        conn.commit()
    logger.info("Database ready.")
