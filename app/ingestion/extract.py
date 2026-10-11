"""Text extraction from PDF and TXT files.

PDF extraction uses PyMuPDF as the primary engine with pypdf as fallback.
Each page is returned as a separate :class:`PageText` object so that
chunk metadata can record the correct page number.

Unreadable or empty PDFs (e.g., scanned images without OCR) are logged
and skipped — the caller receives an empty list, not an exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Text content of a single page."""

    page: int    # 1-indexed
    text: str


def extract_pages(path: Path) -> list[PageText]:
    """Extract text per page from a PDF or TXT file.

    Args:
        path: Absolute path to the source file.

    Returns:
        List of :class:`PageText` objects, one per page.
        Returns an empty list (with a warning) if extraction fails or
        produces no text.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _extract_txt(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    logger.warning("Unsupported file type '%s' for %s — skipping.", suffix, path.name)
    return []


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_txt(path: Path) -> list[PageText]:
    """Read a plain-text file as a single page."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.error("Cannot read TXT file %s: %s", path.name, exc)
        return []

    if not text:
        logger.warning("TXT file %s is empty — skipping.", path.name)
        return []

    return [PageText(page=1, text=text)]


def _extract_pdf(path: Path) -> list[PageText]:
    """Extract text from a PDF, trying PyMuPDF then pypdf as fallback."""
    pages = _extract_pdf_pymupdf(path)
    if pages:
        return pages

    logger.warning(
        "PyMuPDF returned no text for %s — trying pypdf fallback.", path.name
    )
    pages = _extract_pdf_pypdf(path)
    if pages:
        return pages

    logger.warning(
        "Both extractors returned no text for %s. "
        "It may be a scanned PDF without OCR. Skipping.",
        path.name,
    )
    return []


def _extract_pdf_pymupdf(path: Path) -> list[PageText]:
    """Primary PDF extractor using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.error("PyMuPDF could not open %s: %s", path.name, exc)
        return []

    pages: list[PageText] = []
    try:
        for page_num in range(len(doc)):
            try:
                text = doc[page_num].get_text("text").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PyMuPDF failed on page %d of %s: %s", page_num + 1, path.name, exc
                )
                continue
            if text:
                pages.append(PageText(page=page_num + 1, text=text))
    finally:
        doc.close()

    return pages


def _extract_pdf_pypdf(path: Path) -> list[PageText]:
    """Fallback PDF extractor using pypdf."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.error("pypdf could not open %s: %s", path.name, exc)
        return []

    pages: list[PageText] = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pypdf failed on page %d of %s: %s", page_num + 1, path.name, exc
            )
            continue
        if text:
            pages.append(PageText(page=page_num + 1, text=text))

    return pages
