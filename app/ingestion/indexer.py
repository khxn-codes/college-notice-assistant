"""Indexer — embeds chunks and stores them in ChromaDB.

BM25 indexing is deferred to Milestone 3 (retrieval).
This module only handles the embedding + Chroma persistence step.

Chunk metadata stored in Chroma (per TECH_SPEC §4):
  notice_id, title, notice_date, category, page, chunk_index
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Lazy singletons — initialised on first use to avoid slow imports at startup
_embedding_fn = None
_chroma_client = None
_collections: dict[str, Any] = {}

_COLLECTION_NAME = "notices"


def get_chroma_client():
    """Return (and lazily create) the persistent ChromaDB client."""
    global _chroma_client  # noqa: PLW0603
    if _chroma_client is not None:
        return _chroma_client

    import chromadb

    chroma_dir = Path(settings.data_dir) / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    return _chroma_client


def get_embedding_function():
    """Return (and lazily create) the SentenceTransformer embedding function."""
    global _embedding_fn  # noqa: PLW0603
    if _embedding_fn is not None:
        return _embedding_fn

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    logger.info("Loading embedding model '%s' …", settings.embedding_model)
    _embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return _embedding_fn


def get_collection(name: str = _COLLECTION_NAME):
    """Return (and lazily create) the specified Chroma collection."""
    global _collections  # noqa: PLW0603
    if name in _collections:
        return _collections[name]

    client = get_chroma_client()
    fn = get_embedding_function()

    coll = client.get_or_create_collection(
        name=name,
        embedding_function=fn,
        metadata={"hnsw:space": "cosine"},
    )
    _collections[name] = coll
    logger.info(
        "Chroma collection '%s' ready (%d existing chunks).",
        name,
        coll.count(),
    )
    return coll


def _get_collection(name: str = _COLLECTION_NAME):
    """Backwards-compatible alias for get_collection."""
    return get_collection(name)


def index_chunks(
    chunks: list[Chunk],
    notice_id: str,
    title: str,
    notice_date: str,   # ISO-8601 string
    category: str,
    collection_name: str = _COLLECTION_NAME,
) -> None:
    """Embed ``chunks`` and upsert them into the Chroma collection.

    Args:
        chunks:          Chunks produced by :mod:`app.ingestion.chunker`.
        notice_id:       UUID of the notice row in SQLite.
        title:           Notice title (stored as metadata).
        notice_date:     ISO-8601 date string (stored as metadata).
        category:        Notice category (stored as metadata).
        collection_name: Target Chroma collection name (default: "notices").
    """
    if not chunks:
        logger.warning("index_chunks called with empty chunk list for notice %s — nothing to index.", notice_id)
        return

    collection = get_collection(collection_name)

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
        "Indexing %d chunk(s) for notice '%s' (id=%s) into '%s' …",
        len(chunks),
        title,
        notice_id,
        collection_name,
    )
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d chunk(s) into Chroma collection '%s'.", len(chunks), collection_name)


def reindex_collection_for_config(config_name: str, force: bool = False) -> int:
    """Re-index all stored notices into a separate Chroma collection for a config.

    When a config specifies a different chunk size, this slices notices using
    that config's chunk size and stores the resulting chunks into the config's
    designated collection without touching the main 'notices' collection.
    """
    from app.config import get_rag_config
    from app.db import get_connection
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.extract import extract_pages

    cfg = get_rag_config(config_name)
    coll = get_collection(cfg.collection_name)

    # If already indexed and force is False, skip reindexing
    if coll.count() > 0 and not force:
        logger.info(
            "Collection '%s' for Config %s already has %d chunks — skipping re-indexing.",
            cfg.collection_name,
            cfg.name,
            coll.count(),
        )
        return coll.count()

    # If force, clear the separate collection first (never clear main notices)
    if force and cfg.collection_name != _COLLECTION_NAME and coll.count() > 0:
        client = get_chroma_client()
        client.delete_collection(cfg.collection_name)
        _collections.pop(cfg.collection_name, None)
        coll = get_collection(cfg.collection_name)

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, notice_date, category, source_file FROM notices ORDER BY notice_date ASC"
        ).fetchall()

    if not rows:
        logger.warning("No notices found in database to re-index for config %s.", cfg.name)
        return 0

    total_chunks = 0
    data_dir = Path(settings.data_dir)

    for row in rows:
        # Search for source file in processed then inbox
        file_path = data_dir / "processed" / row["source_file"]
        if not file_path.exists():
            file_path = data_dir / "inbox" / row["source_file"]

        if not file_path.exists():
            logger.warning("Source file not found for notice '%s': %s", row["title"], row["source_file"])
            continue

        pages = extract_pages(file_path)
        if not pages:
            logger.warning("No pages extracted from %s for notice %s.", file_path, row["id"])
            continue

        chunks = chunk_pages(pages, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        index_chunks(
            chunks=chunks,
            notice_id=row["id"],
            title=row["title"],
            notice_date=row["notice_date"],
            category=row["category"],
            collection_name=cfg.collection_name,
        )
        total_chunks += len(chunks)

    logger.info(
        "Re-indexing complete for Config %s: %d total chunk(s) in collection '%s'.",
        cfg.name,
        total_chunks,
        cfg.collection_name,
    )
    return coll.count()


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

