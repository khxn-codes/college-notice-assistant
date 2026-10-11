"""Indexer — embeds chunks and stores them in ChromaDB.

BM25 indexing is deferred to Milestone 3 (retrieval).
This module only handles the embedding + Chroma persistence step.

Chunk metadata stored in Chroma (per TECH_SPEC §4):
  notice_id, title, notice_date, category, page, chunk_index
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Lazy singletons — initialised on first use to avoid slow imports at startup
_embedding_fn = None
_chroma_client = None
_collection = None

_COLLECTION_NAME = "notices"


def _get_collection():
    """Return (and lazily create) the Chroma collection."""
    global _embedding_fn, _chroma_client, _collection  # noqa: PLW0603

    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_dir = Path(settings.data_dir) / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model '%s' …", settings.embedding_model)
    _embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    _chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    _collection = _chroma_client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Chroma collection '%s' ready (%d existing chunks).",
        _COLLECTION_NAME,
        _collection.count(),
    )
    return _collection


def index_chunks(
    chunks: list[Chunk],
    notice_id: str,
    title: str,
    notice_date: str,   # ISO-8601 string
    category: str,
) -> None:
    """Embed ``chunks`` and upsert them into the Chroma collection.

    Args:
        chunks:       Chunks produced by :mod:`app.ingestion.chunker`.
        notice_id:    UUID of the notice row in SQLite.
        title:        Notice title (stored as metadata).
        notice_date:  ISO-8601 date string (stored as metadata).
        category:     Notice category (stored as metadata).
    """
    if not chunks:
        logger.warning("index_chunks called with empty chunk list for notice %s — nothing to index.", notice_id)
        return

    collection = _get_collection()

    ids = [f"{notice_id}_c{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "notice_id": notice_id,
            "title": title,
            "notice_date": notice_date,
            "category": category,
            "page": c.page,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    logger.info(
        "Indexing %d chunk(s) for notice '%s' (id=%s) …",
        len(chunks),
        title,
        notice_id,
    )
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d chunk(s) into Chroma.", len(chunks))


def delete_notice_chunks(notice_id: str) -> None:
    """Remove all Chroma chunks belonging to ``notice_id``.

    Used by the DELETE /notices/{id} endpoint (Milestone 3+).
    """
    collection = _get_collection()
    collection.delete(where={"notice_id": notice_id})
    logger.info("Deleted all Chroma chunks for notice %s.", notice_id)


def update_notice_chunks_metadata(
    notice_id: str,
    title: str | None = None,
    notice_date: str | None = None,
    category: str | None = None,
) -> None:
    """Update metadata for existing Chroma chunks belonging to ``notice_id``."""
    try:
        collection = _get_collection()
        results = collection.get(where={"notice_id": notice_id}, include=["metadatas"])
        if not results or not results["ids"]:
            return

        updated_metadatas = []
        for meta in results["metadatas"]:
            new_meta = dict(meta)
            if title is not None:
                new_meta["title"] = title
            if notice_date is not None:
                new_meta["notice_date"] = notice_date
            if category is not None:
                new_meta["category"] = category
            updated_metadatas.append(new_meta)

        collection.update(ids=results["ids"], metadatas=updated_metadatas)
        logger.info("Updated metadata for %d chunk(s) in notice %s.", len(results["ids"]), notice_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update Chroma metadata for notice %s: %s", notice_id, exc)

