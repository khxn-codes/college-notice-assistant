"""Revision detection logic for notices.

Identifies whether a newly ingested notice revises or supersedes an older notice
in the same category, based on title similarity, revision keywords, and notice dates.

TECH_SPEC §5, §6 and Milestone 4 requirements:
  - Same category.
  - Earlier date for the older notice.
  - Highly similar title or filename containing 'revised', 'extended', 'rescheduled',
    or matching after removing such words.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

REVISION_KEYWORDS = {"revised", "extended", "rescheduled", "revision", "extension"}
_REVISION_PATTERN = re.compile(
    r"\b(revised|extended|rescheduled|revision|extension)\b",
    re.IGNORECASE,
)


def has_revision_keyword(text: str) -> bool:
    """Return True if ``text`` contains revision keywords."""
    cleaned = text.replace("-", " ").replace("_", " ")
    return bool(_REVISION_PATTERN.search(cleaned))


def strip_revision_words(text: str) -> str:
    """Remove revision keywords, punctuation, and extra whitespace."""
    cleaned = _REVISION_PATTERN.sub(" ", text)
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return " ".join(cleaned.lower().split())


def clean_slug_or_filename(filename: str) -> str:
    """Extract and normalise the core title slug from a filename."""
    stem = Path(filename).stem
    # If standard convention YYYY-MM-DD_category_slug
    m = re.match(r"^\d{4}-\d{2}-\d{2}_[a-z0-9]+_(.+)$", stem, re.IGNORECASE)
    if m:
        stem = m.group(1)
    else:
        # Remove leading date YYYY-MM-DD if present without category
        stem = re.sub(r"^\d{4}-\d{2}-\d{2}[-_]?", "", stem)
        # Remove known category prefix if present
        stem = re.sub(r"^(exam|fees|hostel|events|general)[-_]", "", stem, flags=re.IGNORECASE)
    return strip_revision_words(stem)


def are_titles_or_filenames_revisions(
    new_title: str,
    new_filename: str,
    old_title: str,
    old_filename: str,
    similarity_threshold: float = 0.8,
) -> bool:
    """Determine if a new notice title/filename revises an older notice.

    Checks:
      1. Titles match after removing revision keywords.
      2. Filename slugs match after removing date, category, and revision keywords.
      3. Title similarity ratio >= similarity_threshold after removing revision words.
      4. If either title or filename contains a revision keyword, token containment
         or sequence similarity >= 0.7.
    """
    norm_new_title = strip_revision_words(new_title)
    norm_old_title = strip_revision_words(old_title)

    # 1. Exact match after stripping revision words
    if norm_new_title and norm_old_title and norm_new_title == norm_old_title:
        return True

    # 2. Filename slug exact match after stripping
    slug_new = clean_slug_or_filename(new_filename)
    slug_old = clean_slug_or_filename(old_filename)
    if slug_new and slug_old and slug_new == slug_old:
        return True

    # 3. High title similarity
    if norm_new_title and norm_old_title:
        sim = SequenceMatcher(None, norm_new_title, norm_old_title).ratio()
        if sim >= similarity_threshold:
            return True

    # 4. Keyword presence check
    has_keyword = (
        has_revision_keyword(new_title)
        or has_revision_keyword(new_filename)
        or has_revision_keyword(old_title)
        or has_revision_keyword(old_filename)
    )
    if has_keyword:
        # Check token containment in titles
        tokens_new = set(norm_new_title.split())
        tokens_old = set(norm_old_title.split())
        if tokens_new and tokens_old:
            if tokens_new.issubset(tokens_old) or tokens_old.issubset(tokens_new):
                return True

        # Check token containment in filename slugs
        fn_tokens_new = set(slug_new.split())
        fn_tokens_old = set(slug_old.split())
        if fn_tokens_new and fn_tokens_old:
            if fn_tokens_new.issubset(fn_tokens_old) or fn_tokens_old.issubset(fn_tokens_new):
                return True

        # Partial similarity when revision keyword is present
        if norm_new_title and norm_old_title:
            if SequenceMatcher(None, norm_new_title, norm_old_title).ratio() >= 0.7:
                return True
        if slug_new and slug_old:
            if SequenceMatcher(None, slug_new, slug_old).ratio() >= 0.7:
                return True

    return False


def _parse_date_str(val: str | date | datetime) -> date:
    """Coerce string or date into date object."""
    if isinstance(val, (datetime, date)):
        return val if isinstance(val, date) and not isinstance(val, datetime) else val.date()
    # Assume ISO string YYYY-MM-DD
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except Exception:
        return date.today()


def find_superseded_notice(
    title: str,
    notice_date: str | date,
    category: str,
    source_file: str,
    existing_notices: list[dict],
) -> str | None:
    """Find the ID of the older notice that this new notice supersedes, or None.

    Args:
        title: Title of the newly ingested notice.
        notice_date: Date of the newly ingested notice.
        category: Category of the newly ingested notice.
        source_file: Original filename of the newly ingested notice.
        existing_notices: List of candidate notice dicts, each having
                          ``id``, ``title``, ``notice_date``, ``category``,
                          and ``source_file``.

    Returns:
        UUID string of the superseded notice, or None.
    """
    new_dt = _parse_date_str(notice_date)
    new_cat = category.strip().lower()

    matches: list[dict] = []
    for cand in existing_notices:
        cand_cat = (cand.get("category") or "").strip().lower()
        if cand_cat != new_cat:
            continue

        cand_dt = _parse_date_str(cand.get("notice_date", ""))
        if cand_dt >= new_dt:
            # Candidate notice is not older
            continue

        cand_title = cand.get("title", "")
        cand_file = cand.get("source_file", "")

        if are_titles_or_filenames_revisions(
            new_title=title,
            new_filename=source_file,
            old_title=cand_title,
            old_filename=cand_file,
        ):
            matches.append(cand)

    if not matches:
        return None

    # Pick the most recent older notice among matches
    matches.sort(
        key=lambda c: _parse_date_str(c.get("notice_date", "")),
        reverse=True,
    )
    superseded_id = matches[0]["id"]
    logger.info(
        "Detected revision: '%s' (%s) supersedes '%s' (%s, date=%s)",
        title,
        source_file,
        matches[0].get("title"),
        superseded_id,
        matches[0].get("notice_date"),
    )
    return str(superseded_id)


def detect_superseded_notice_in_db(
    title: str,
    notice_date: str | date,
    category: str,
    source_file: str,
    conn: sqlite3.Connection,
) -> str | None:
    """Query SQLite for older notices in the same category and find superseded notice."""
    new_dt = _parse_date_str(notice_date).isoformat()
    rows = conn.execute(
        """
        SELECT id, title, notice_date, category, source_file
        FROM notices
        WHERE LOWER(category) = LOWER(?) AND notice_date < ?
        ORDER BY notice_date DESC
        """,
        (category.strip(), new_dt),
    ).fetchall()

    candidate_notices = [dict(r) for r in rows]
    return find_superseded_notice(
        title=title,
        notice_date=notice_date,
        category=category,
        source_file=source_file,
        existing_notices=candidate_notices,
    )
