"""Unit tests for app/ingestion/chunker.py."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import Chunk, chunk_pages


def _pages(text: str, page: int = 1) -> list[tuple[int, str]]:
    """Helper: wrap a single string as a page list."""
    return [(page, text)]


class TestChunkPages:
    def test_empty_input_returns_empty(self) -> None:
        assert chunk_pages([]) == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_pages([(1, "   \n\n  ")]) == []

    def test_short_text_is_single_chunk(self) -> None:
        text = "The exam is on 17 October 2026."
        chunks = chunk_pages(_pages(text), chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].page == 1
        assert chunks[0].chunk_index == 0

    def test_chunk_index_increments(self) -> None:
        # 200-word paragraph × 3 with size=250 should produce multiple chunks
        word = "word"
        para = (" ".join([word] * 200))
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = chunk_pages(_pages(text), chunk_size=250, chunk_overlap=0)
        assert len(chunks) >= 2
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_overlap_carries_words(self) -> None:
        # Build sentences so the chunker has boundaries to split on.
        # Each sentence is ~25 words; 8 sentences = 200 words total.
        sentences = [
            " ".join(str(i * 25 + j) for j in range(25)) + "."
            for i in range(8)
        ]
        text = " ".join(sentences)
        chunks = chunk_pages(_pages(text), chunk_size=60, chunk_overlap=20)
        assert len(chunks) >= 2
        # The last 20 words of chunk 0 should appear at the start of chunk 1
        first_end = chunks[0].text.split()[-20:]
        second_start = chunks[1].text.split()[:20]
        assert first_end == second_start

    def test_page_number_recorded(self) -> None:
        pages = [(1, "Page one text."), (2, "Page two text.")]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=0)
        # With plenty of space, all text fits in one chunk starting on page 1
        assert chunks[0].page == 1

    def test_multi_page_chunks_respect_size(self) -> None:
        word = "hello"
        pages = [(1, " ".join([word] * 300)), (2, " ".join([word] * 300))]
        chunks = chunk_pages(pages, chunk_size=250, chunk_overlap=0)
        for c in chunks:
            assert len(c.text.split()) <= 300  # some slack for segment boundaries

    def test_returns_chunk_objects(self) -> None:
        chunks = chunk_pages(_pages("Hello world."), chunk_size=500, chunk_overlap=0)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_overlap_no_repeated_words(self) -> None:
        text = " ".join(str(i) for i in range(400))
        chunks = chunk_pages(_pages(text), chunk_size=100, chunk_overlap=0)
        all_words: list[str] = []
        for c in chunks:
            all_words.extend(c.text.split())
        # With zero overlap every word should appear exactly once
        assert len(all_words) == len(set(all_words))
