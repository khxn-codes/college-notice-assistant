"""BM25 index store — builds and persists an in-memory BM25 corpus.

The BM25 index is rebuilt from the Chroma collection after every ingest
so it stays in sync. It is held as a process-level singleton because
rank_bm25 is CPU-only and has no persistence format.

Usage:
    from app.retrieval.bm25_store import get_bm25_index, rebuild_bm25_index
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BM25Doc:
    """One document entry in the BM25 corpus."""

    chunk_id: str       # Chroma document ID, e.g. "{notice_id}_c{chunk_index}"
    text: str
    metadata: dict


# Process-level state
_bm25_index = None          # rank_bm25.BM25Okapi instance
_bm25_docs: list[BM25Doc] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rebuild_bm25_index() -> None:
    """Rebuild the BM25 index from all chunks currently in Chroma.

    Called after every upload or sync so the index is always current.
    Safe to call from FastAPI startup (lazy — skips if Chroma is empty).
    """
    global _bm25_index, _bm25_docs  # noqa: PLW0603

    try:
        from app.ingestion.indexer import _get_collection  # avoid circular at module level
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            logger.info("BM25: Chroma collection is empty — index not built.")
            _bm25_index = None
            _bm25_docs = []
            return

        # Fetch all documents (Chroma allows large get())
        result = collection.get(include=["documents", "metadatas"])
        ids: list[str] = result["ids"]
        docs: list[str] = result["documents"]
        metas: list[dict] = result["metadatas"]

        _bm25_docs = [
            BM25Doc(chunk_id=cid, text=txt, metadata=meta)
            for cid, txt, meta in zip(ids, docs, metas)
        ]

        from rank_bm25 import BM25Okapi
        tokenised = [doc.text.lower().split() for doc in _bm25_docs]
        _bm25_index = BM25Okapi(tokenised)
        logger.info("BM25 index built: %d documents.", len(_bm25_docs))

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build BM25 index: %s", exc)
        _bm25_index = None
        _bm25_docs = []


def get_bm25_index():
    """Return the current BM25 index and corpus, or (None, []) if not built."""
    return _bm25_index, _bm25_docs
