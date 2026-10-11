"""Notice metadata parsing.

Parses title, date, and category from the filename convention:
    ``YYYY-MM-DD_category_short-title.pdf``

Falls back gracefully when the convention is not followed:
- Date: today's date (upload date)
- Category: "general"
- Title: stem of the filename, cleaned up
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches: 2026-10-05_exam_midsem-schedule.pdf
_CONVENTION_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"_(?P<category>[a-z0-9]+)"
    r"_(?P<slug>.+)$",
    re.IGNORECASE,
)


@dataclass
class NoticeMetadata:
    """Parsed metadata for a single notice file."""

    title: str
    notice_date: date
    category: str
    source_file: str


def parse_metadata(path: Path) -> NoticeMetadata:
    """Parse title, date, and category from a notice filename.

    Tries the ``YYYY-MM-DD_category_slug`` convention first.
    Falls back to today's date and 'general' category if parsing fails.

    Args:
        path: Path to the notice file (only the filename is used).

    Returns:
        :class:`NoticeMetadata` with the parsed (or default) values.
    """
    stem = path.stem  # filename without extension
    source_file = path.name

    m = _CONVENTION_RE.match(stem)
    if m:
        try:
            notice_date = date(
                int(m.group("year")),
                int(m.group("month")),
                int(m.group("day")),
            )
        except ValueError:
            logger.warning(
                "Invalid date in filename '%s'; falling back to today.", path.name
            )
            notice_date = date.today()

        category = m.group("category").lower()
        slug = m.group("slug")
        title = _slug_to_title(slug)
        logger.debug("Parsed metadata from filename: %s -> %s / %s / %s", path.name, title, notice_date, category)
    else:
        logger.info(
            "Filename '%s' does not follow the YYYY-MM-DD_category_slug convention; "
            "using defaults.",
            path.name,
        )
        notice_date = date.today()
        category = "general"
        title = _slug_to_title(stem)

    return NoticeMetadata(
        title=title,
        notice_date=notice_date,
        category=category,
        source_file=source_file,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug_to_title(slug: str) -> str:
    """Convert a filename slug to a readable title.

    Replaces hyphens and underscores with spaces, then title-cases the result.
    Example: ``midsem-schedule`` → ``Midsem Schedule``
    """
    return re.sub(r"[-_]+", " ", slug).strip().title()
