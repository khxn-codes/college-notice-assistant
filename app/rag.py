"""RAG orchestrator — retrieve → prompt → LLM → answer + citations.

Called by the POST /ask endpoint. Returned dict shape:
  {
    "answer": str,
    "citations": [{"title", "notice_date", "page", "notice_id", "chunk_id"}, ...],
    "chunks":   [{"chunk_id", "text", "title", "notice_date", "page", "score"}, ...],
    "latency_ms": int,
    "config": str,
  }
"""

from __future__ import annotations

import logging
import time

from app.llm.client import BaseLLMClient, get_llm_client
from app.llm.prompts import SYSTEM_PROMPT, build_user_message
from app.retrieval.latest_wins import apply_latest_wins
from app.retrieval.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

_REFUSAL = "I could not find this in the available notices."


def answer_question(
    question: str,
    category: str | None = None,
    config: str | None = None,
    llm_client: BaseLLMClient | None = None,
    collection_name: str | None = None,
) -> dict:
    """Run the full RAG pipeline for *question*.

    Args:
        question:        The student's question.
        category:        Optional category filter passed to the retriever.
        config:          Named retrieval configuration (``"A"`` = vector-only,
                         ``"B"`` = hybrid). Defaults to the env-configured mode.
        llm_client:      Injectable LLM client (used in tests to avoid real API calls).
        collection_name: Optional Chroma collection name to query.

    Returns:
        Dict with answer, citations, chunks, latency_ms, and config name.
    """
    t0 = time.monotonic()

    # ── Map config name to retrieval mode ─────────────────────────────────────
    mode: str | None = None
    if config == "A":
        mode = "vector"
    elif config == "B":
        mode = "hybrid"
    # else: use settings.retrieval_mode default

    config_label = config or "default"

    # ── Retrieve & Apply Latest-Wins ──────────────────────────────────────────
    retrieve_kwargs = {"category": category, "mode": mode}
    if collection_name is not None:
        retrieve_kwargs["collection_name"] = collection_name
    raw_chunks: list[RetrievedChunk] = retrieve(question, **retrieve_kwargs)
    chunks, supersedes_notes = apply_latest_wins(raw_chunks, drop=True)
    logger.info(
        "RAG: question='%.80s…'  raw_chunks=%d  filtered_chunks=%d  notes=%s  config=%s",
        question,
        len(raw_chunks),
        len(chunks),
        supersedes_notes,
        config_label,
    )

    # ── Build prompt ──────────────────────────────────────────────────────────
    passages = [
        {
            "index": i + 1,
            "title": c.title,
            "notice_date": c.notice_date,
            "page": c.page,
            "text": c.text,
        }
        for i, c in enumerate(chunks)
    ]
    user_message = build_user_message(question, passages, supersedes_notes=supersedes_notes)

    # ── LLM call ──────────────────────────────────────────────────────────────
    client = llm_client or get_llm_client()
    raw_answer = client.complete(system=SYSTEM_PROMPT, user=user_message)

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info("RAG: answer generated in %d ms.", latency_ms)

    # ── Build citations from chunks that appear to be cited ───────────────────
    # We include all retrieved chunks as potential citations; the LLM's text
    # determines which are actually referenced, but we surface all of them
    # for the source viewer (as the spec requires showing retrieved chunks).
    citations = [
        {
            "title": c.title,
            "notice_date": c.notice_date,
            "page": c.page,
            "notice_id": c.notice_id,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]

    chunk_payloads = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "title": c.title,
            "notice_date": c.notice_date,
            "page": c.page,
            "score": round(c.score, 6),
        }
        for c in chunks
    ]

    return {
        "answer": raw_answer,
        "citations": citations,
        "chunks": chunk_payloads,
        "notes": supersedes_notes,
        "latency_ms": latency_ms,
        "config": config_label,
    }
