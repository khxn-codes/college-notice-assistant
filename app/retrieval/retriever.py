"""Hybrid retriever — vector similarity + BM25 + Reciprocal Rank Fusion.

TECH_SPEC §6:
  - Vector top-k=8 from Chroma, BM25 top-k=8 from rank_bm25.
  - Merge with RRF (k=60 constant), keep final top-5.
  - Optional category filter applied to both legs.
  - Retrieval mode ("hybrid" | "vector") controlled via config / caller arg.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings
from app.retrieval.bm25_store import get_bm25_index
from app.retrieval.latest_wins import apply_latest_wins

logger = logging.getLogger(__name__)

# RRF smoothing constant (standard value from the paper)
_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A single chunk returned by the retriever."""

    chunk_id: str
    text: str
    notice_id: str
    title: str
    notice_date: str
    category: str
    page: int
    chunk_index: int
    score: float   # RRF score (higher = more relevant)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    category: str | None = None,
    top_k: int | None = None,
    mode: str | None = None,
    apply_latest_wins_rule: bool = False,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid (or vector-only) retrieval for *query*.

    Args:
        query:                  The user's question.
        category:               Optional category filter (e.g. "exam", "fees").
                                When provided, only chunks from that category are returned.
        top_k:                  Final number of chunks to return. Defaults to ``settings.top_k``.
        mode:                   ``"hybrid"`` or ``"vector"``. Defaults to ``settings.retrieval_mode``.
        apply_latest_wins_rule: If True, filter out superseded notice chunks.
        collection_name:        Target Chroma collection name (optional, defaults to main collection).

    Returns:
        List of :class:`RetrievedChunk` sorted by descending score.
    """
    k = top_k if top_k is not None else settings.top_k
    retrieval_mode = mode if mode is not None else settings.retrieval_mode

    # ── Vector leg ────────────────────────────────────────────────────────────
    vector_results = _vector_search(query, category=category, k=settings.vector_k, collection_name=collection_name)

    if retrieval_mode == "vector":
        # Vector-only — convert scores and return
        chunks = _dedupe_by_id(vector_results)
        final_chunks = chunks[:k]
    else:
        # ── BM25 leg ──────────────────────────────────────────────────────────
        bm25_results = _bm25_search(query, category=category, k=settings.bm25_k)

        # ── RRF merge ─────────────────────────────────────────────────────────
        final_chunks = _reciprocal_rank_fusion(vector_results, bm25_results, top_k=k)
        logger.info(
            "Retrieval [%s]: query='%.60s…'  vector=%d  bm25=%d  merged=%d",
            retrieval_mode,
            query,
            len(vector_results),
            len(bm25_results),
            len(final_chunks),
        )

    if apply_latest_wins_rule:
        final_chunks, _ = apply_latest_wins(final_chunks, drop=True)

    return final_chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _vector_search(
    query: str,
    category: str | None,
    k: int,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    """Query Chroma for the top-k most similar chunks."""
    try:
        from app.ingestion.indexer import get_collection
        collection = get_collection(collection_name or "notices")
    except Exception as exc:  # noqa: BLE001
        logger.error("Chroma unavailable for vector search: %s", exc)
        return []

    if collection.count() == 0:
        return []

    where: dict | None = {"category": category} if category else None

    try:
        result = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Chroma query failed: %s", exc)
        return []

    chunks: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        # Chroma cosine distance → similarity score (1 - distance)
        chunks.append(_make_chunk(cid, doc, meta, score=1.0 - dist))
    return chunks


def _bm25_search(
    query: str, category: str | None, k: int
) -> list[RetrievedChunk]:
    """Search the in-memory BM25 index for the top-k keyword matches."""
    bm25, docs = get_bm25_index()
    if bm25 is None or not docs:
        logger.debug("BM25 index is empty — skipping BM25 leg.")
        return []

    tokenised_query = query.lower().split()
    if not tokenised_query:
        return []

    scores = bm25.get_scores(tokenised_query)
    query_terms = set(tokenised_query)

    filtered: list[tuple[int, float]] = []
    for i, (score, doc) in enumerate(zip(scores, docs)):
        if category and doc.metadata.get("category") != category:
            continue
        doc_tokens = set(doc.text.lower().split())
        if not (query_terms & doc_tokens):
            continue
        filtered.append((i, float(score)))

    # Sort descending by score, take top-k
    filtered.sort(key=lambda x: x[1], reverse=True)
    top = filtered[:k]

    chunks: list[RetrievedChunk] = []
    for idx, score in top:
        d = docs[idx]
        chunks.append(_make_chunk(d.chunk_id, d.text, d.metadata, score=max(score, 0.0001)))
    return chunks


def _reciprocal_rank_fusion(
    list_a: list[RetrievedChunk],
    list_b: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)  where k=60 (standard constant).
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(list_a, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(list_b, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    result: list[RetrievedChunk] = []
    for cid in sorted_ids[:top_k]:
        chunk = chunk_map[cid]
        chunk.score = rrf_scores[cid]
        result.append(chunk)
    return result


def _dedupe_by_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove any duplicate chunk IDs, keeping the first occurrence."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def _make_chunk(chunk_id: str, text: str, meta: dict, score: float) -> RetrievedChunk:
    """Construct a :class:`RetrievedChunk` from Chroma/BM25 raw data."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        notice_id=meta.get("notice_id", ""),
        title=meta.get("title", ""),
        notice_date=meta.get("notice_date", ""),
        category=meta.get("category", ""),
        page=int(meta.get("page", 1)),
        chunk_index=int(meta.get("chunk_index", 0)),
        score=score,
    )
