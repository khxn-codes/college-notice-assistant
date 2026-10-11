"""Unit tests for Reciprocal Rank Fusion and category filter in retriever.py."""

from __future__ import annotations

import pytest

from app.retrieval.retriever import RetrievedChunk, _reciprocal_rank_fusion, _bm25_search


def _make_chunk(chunk_id: str, score: float = 1.0, category: str = "general") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        notice_id="notice-1",
        title="Test Notice",
        notice_date="2026-01-01",
        category=category,
        page=1,
        chunk_index=0,
        score=score,
    )


class TestRRF:
    """Tests for _reciprocal_rank_fusion."""

    def test_single_list_scores_preserved_order(self) -> None:
        """With only one list, items come back in their original rank order."""
        a = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
        result = _reciprocal_rank_fusion(a, [], top_k=3)
        ids = [r.chunk_id for r in result]
        assert ids == ["c1", "c2", "c3"]

    def test_chunk_in_both_lists_ranks_higher(self) -> None:
        """A chunk that appears in both lists should score above chunks in only one."""
        # c2 appears in both; c1 and c3 each appear in only one
        list_a = [_make_chunk("c1"), _make_chunk("c2")]
        list_b = [_make_chunk("c2"), _make_chunk("c3")]
        result = _reciprocal_rank_fusion(list_a, list_b, top_k=3)
        assert result[0].chunk_id == "c2", "c2 should rank first as it appears in both lists"

    def test_top_k_limits_results(self) -> None:
        list_a = [_make_chunk(f"c{i}") for i in range(5)]
        result = _reciprocal_rank_fusion(list_a, [], top_k=3)
        assert len(result) == 3

    def test_empty_lists_return_empty(self) -> None:
        assert _reciprocal_rank_fusion([], [], top_k=5) == []

    def test_scores_are_positive(self) -> None:
        list_a = [_make_chunk("c1"), _make_chunk("c2")]
        result = _reciprocal_rank_fusion(list_a, [], top_k=2)
        for r in result:
            assert r.score > 0.0

    def test_rrf_score_is_sum_of_reciprocal_ranks(self) -> None:
        """c1 at rank 1 in both lists → score = 1/(60+1) + 1/(60+1) = 2/61."""
        list_a = [_make_chunk("c1")]
        list_b = [_make_chunk("c1")]
        result = _reciprocal_rank_fusion(list_a, list_b, top_k=1)
        expected = 2.0 / (60 + 1)
        assert abs(result[0].score - expected) < 1e-9

    def test_no_duplicates_in_output(self) -> None:
        list_a = [_make_chunk("c1"), _make_chunk("c2")]
        list_b = [_make_chunk("c1"), _make_chunk("c3")]
        result = _reciprocal_rank_fusion(list_a, list_b, top_k=5)
        ids = [r.chunk_id for r in result]
        assert len(ids) == len(set(ids)), "Output must not contain duplicate chunk IDs"

    def test_result_sorted_descending_by_score(self) -> None:
        list_a = [_make_chunk(f"c{i}") for i in range(4)]
        list_b = [_make_chunk(f"c{i}") for i in range(4)]
        result = _reciprocal_rank_fusion(list_a, list_b, top_k=4)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)


class TestCategoryFilter:
    """Tests for category filtering in BM25 search (offline, no Chroma needed)."""

    def test_bm25_category_filter_excludes_wrong_category(self) -> None:
        """BM25 search with a category filter must not return chunks from other categories."""
        from unittest.mock import patch
        from app.retrieval.bm25_store import BM25Doc

        exam_doc = BM25Doc(
            chunk_id="exam_c0",
            text="mid semester exam schedule october",
            metadata={"notice_id": "n1", "title": "Exam Notice", "notice_date": "2026-10-01",
                      "category": "exam", "page": 1, "chunk_index": 0},
        )
        fees_doc = BM25Doc(
            chunk_id="fees_c0",
            text="tuition fee payment october deadline",
            metadata={"notice_id": "n2", "title": "Fee Notice", "notice_date": "2026-10-01",
                      "category": "fees", "page": 1, "chunk_index": 0},
        )

        from rank_bm25 import BM25Okapi
        corpus = [d.text.lower().split() for d in [exam_doc, fees_doc]]
        bm25 = BM25Okapi(corpus)

        with patch("app.retrieval.retriever.get_bm25_index", return_value=(bm25, [exam_doc, fees_doc])):
            results = _bm25_search("october exam schedule", category="exam", k=5)

        assert all(r.category == "exam" for r in results), \
            "BM25 with category='exam' must not return non-exam chunks"

    def test_bm25_no_filter_returns_all_categories(self) -> None:
        """Without a category filter every matching chunk is eligible."""
        from unittest.mock import patch
        from app.retrieval.bm25_store import BM25Doc

        docs = [
            BM25Doc(chunk_id="c1", text="exam october schedule",
                    metadata={"notice_id": "n1", "title": "T", "notice_date": "2026-01-01",
                              "category": "exam", "page": 1, "chunk_index": 0}),
            BM25Doc(chunk_id="c2", text="fees october deadline",
                    metadata={"notice_id": "n2", "title": "T", "notice_date": "2026-01-01",
                              "category": "fees", "page": 1, "chunk_index": 0}),
        ]
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([d.text.lower().split() for d in docs])

        with patch("app.retrieval.retriever.get_bm25_index", return_value=(bm25, docs)):
            results = _bm25_search("october", category=None, k=5)

        categories = {r.category for r in results}
        assert len(categories) > 1, "Without a filter, multiple categories should be returned"
