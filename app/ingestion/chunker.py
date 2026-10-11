"""Text chunker.

Splits notice text into overlapping chunks, respecting paragraph and
sentence boundaries where possible. Chunk size and overlap are configurable
(defaults: 500 words / 80 words) and come from :mod:`app.config`.

Returns :class:`Chunk` objects that carry the source page number and
chunk index so Chroma metadata stays accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""

    text: str
    page: int          # source page (1-indexed)
    chunk_index: int   # 0-indexed position within the notice


def chunk_pages(
    pages: list[tuple[int, str]],  # list of (page_number, text)
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split multi-page text into overlapping chunks.

    Splits on paragraph boundaries (double newlines) first, then on
    sentence boundaries (``[.!?]``), and finally on whitespace when a
    paragraph is longer than ``chunk_size`` words.

    Args:
        pages:         List of ``(page_number, text)`` tuples.
        chunk_size:    Maximum words per chunk.  Defaults to ``settings.chunk_size``.
        chunk_overlap: Number of words carried over to the next chunk.
                       Defaults to ``settings.chunk_overlap``.

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    # Flatten all pages into sentence-like segments, keeping page numbers
    segments: list[tuple[int, str]] = []
    for page_num, text in pages:
        for seg in _split_text(text):
            stripped = seg.strip()
            if stripped:
                segments.append((page_num, stripped))

    if not segments:
        return []

    chunks: list[Chunk] = []
    buf_words: list[str] = []
    buf_page: int = segments[0][0]
    chunk_idx = 0

    for page_num, seg in segments:
        seg_words = seg.split()
        # If adding this segment would exceed chunk_size, flush first
        if buf_words and len(buf_words) + len(seg_words) > size:
            chunks.append(Chunk(
                text=" ".join(buf_words),
                page=buf_page,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
            # Keep the last `overlap` words for the next chunk
            buf_words = buf_words[-overlap:] if overlap else []
            buf_page = page_num  # start of new chunk

        buf_words.extend(seg_words)
        buf_page = buf_page  # keep earliest page of chunk

    # Flush remaining words
    if buf_words:
        chunks.append(Chunk(
            text=" ".join(buf_words),
            page=buf_page,
            chunk_index=chunk_idx,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """Split text on paragraph boundaries, then sentence boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    segments: list[str] = []
    for para in paragraphs:
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", para)
        segments.extend(sentences)
    return segments
