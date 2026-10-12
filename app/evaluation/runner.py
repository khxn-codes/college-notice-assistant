"""Evaluation runner — executes test_set.json against RAG configurations.

Modes
-----
* **Full RAG run** (default): runs each question through the full retrieve →
  prompt → LLM pipeline.  Adds LLM answer caching, quota-error abort, and
  per-question Hit@5 / MRR metrics derived from the retrieved chunks.
* **Retrieval-only run** (``retrieval_only=True``): skips every LLM call and
  only measures whether the expected source notice appears among the top-5
  retrieved chunks (Hit@5) and the reciprocal rank (MRR).

Changes vs previous version
----------------------------
* ``run_retrieval_only_eval()`` — new public entry-point for retrieval-only mode.
* ``run_evaluation()`` — quota errors abort the run immediately, save partial
  results, and return a ``aborted_reason`` key.
* ``run_evaluation()`` — LLM answers are cached by
  ``(question, frozenset(chunk_ids), config)`` so reruns never repeat a call
  that already succeeded.
* ``run_evaluation()`` — accepts a single config name (not only a list of two).
* ``save_results_csv()`` — includes Hit@5, MRR, and first_correct_rank columns
  in the per-question section.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_rag_config
from app.db import get_connection
from app.evaluation.metrics import compute_metrics, compute_retrieval_metrics
from app.ingestion.indexer import reindex_collection_for_config
from app.llm.client import BaseLLMClient
from app.rag import answer_question
from app.retrieval.retriever import RetrievedChunk, retrieve
from app.retrieval.latest_wins import apply_latest_wins

logger = logging.getLogger(__name__)

DEFAULT_TEST_SET_PATH = Path("eval") / "test_set.json"
DEFAULT_RESULTS_DIR = Path("eval") / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_set(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load evaluation questions from the specified JSON file."""
    target_path = Path(path) if path else DEFAULT_TEST_SET_PATH
    if not target_path.exists():
        logger.warning("Test set file not found: %s", target_path)
        return []

    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logger.error("Test set JSON is not an array: %s", target_path)
        return []
    except Exception as exc:
        logger.error("Failed to parse test set from %s: %s", target_path, exc)
        return []


def get_notice_source_map() -> dict[str, str]:
    """Retrieve notice_id -> source_file mapping from the database."""
    mapping: dict[str, str] = {}
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT id, source_file FROM notices").fetchall()
            for r in rows:
                mapping[r["id"]] = r["source_file"]
    except Exception as exc:
        logger.warning("Could not fetch notice source map from database: %s", exc)
    return mapping


def is_rate_limit_error(error_or_text: Any) -> bool:
    """Check if an exception or response string represents a 429 or rate-limit error."""
    text = str(error_or_text).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
        or "resourceexhausted" in text
        or "resource exhausted" in text
        or "(rate limit)" in text
    )


def is_quota_error(error_or_text: Any) -> bool:
    """Return True when the error signals a hard API quota exhaustion (not just rate-limit).

    Quota errors are unrecoverable within a run, so the runner aborts immediately.
    Rate-limit (429) errors are transient and are retried with backoff.
    """
    text = str(error_or_text).lower()
    return (
        "quota" in text
        and "exhausted" in text
        or "quota_exceeded" in text
        or "billing" in text
        or "daily quota" in text
        or "monthly quota" in text
    )


def get_failed_question_ids_from_run(run_data: dict[str, Any]) -> set[str]:
    """Extract IDs of questions that failed or errored in an evaluation run."""
    failed_ids: set[str] = set()
    for cfg_info in run_data.get("configs", {}).values():
        for q in cfg_info.get("per_question", []):
            if not q.get("passed") or q.get("status") in ("FAIL", "ERROR"):
                qid = q.get("id")
                if qid:
                    failed_ids.add(str(qid))
    return failed_ids


def save_results_csv(run_record: dict[str, Any], csv_path: Path) -> None:
    """Save evaluation metrics comparison and question details to CSV.

    The per-question section now includes Hit@5, MRR, and first_correct_rank
    for retrieval-only runs and (when available) for full runs too.
    """
    import csv

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            run_mode = run_record.get("run_mode", "full")

            if run_mode == "retrieval_only":
                # ── Retrieval-only summary ────────────────────────────────
                writer.writerow(["=== SUMMARY METRICS (RETRIEVAL-ONLY) ==="])
                writer.writerow([
                    "Config",
                    "Retrieval Mode",
                    f"Hit@{run_record.get('top_k', 5)}",
                    "MRR",
                    "Answerable Questions",
                    "Total Questions",
                ])
                for cfg_name, cfg_info in run_record.get("configs", {}).items():
                    m = cfg_info.get("metrics", {})
                    writer.writerow([
                        cfg_name,
                        cfg_info.get("retrieval_mode", ""),
                        f"{m.get('hit_at_k', 0.0) * 100:.1f}%",
                        f"{m.get('mrr', 0.0):.4f}",
                        m.get("total_answerable", 0),
                        m.get("total_questions", 0),
                    ])
                writer.writerow([])
                writer.writerow(["=== PER-QUESTION DETAILS (RETRIEVAL-ONLY) ==="])
                writer.writerow([
                    "Config",
                    "Question ID",
                    "Answerable",
                    "Question",
                    "Expected Source",
                    f"Hit@{run_record.get('top_k', 5)}",
                    "First Correct Rank",
                    "Reciprocal Rank",
                ])
                for cfg_name, cfg_info in run_record.get("configs", {}).items():
                    for q in cfg_info.get("per_question", []):
                        writer.writerow([
                            cfg_name,
                            q.get("id"),
                            "Yes" if q.get("answerable") else "No",
                            q.get("question"),
                            q.get("expected_source") or "—",
                            ("Yes" if q.get("hit") else ("No" if q.get("hit") is False else "N/A")),
                            q.get("first_correct_rank") if q.get("first_correct_rank") is not None else "—",
                            f"{q.get('reciprocal_rank', 0.0):.4f}" if q.get("reciprocal_rank") is not None else "—",
                        ])
            else:
                # ── Full-RAG summary ──────────────────────────────────────
                writer.writerow(["=== SUMMARY METRICS ==="])
                writer.writerow([
                    "Config",
                    "Retrieval Mode",
                    "Answer Accuracy",
                    "Citation Hit Rate",
                    "Refusal Accuracy",
                    "Average Latency (ms)",
                    f"Hit@5",
                    "MRR",
                    "Passed Questions",
                    "Failed Questions",
                    "API Errors",
                    "Total Questions",
                ])
                for cfg_name, cfg_info in run_record.get("configs", {}).items():
                    m = cfg_info.get("metrics", {})
                    writer.writerow([
                        cfg_name,
                        cfg_info.get("retrieval_mode", ""),
                        f"{m.get('answer_accuracy', 0.0) * 100:.1f}%",
                        f"{m.get('citation_hit_rate', 0.0) * 100:.1f}%",
                        f"{m.get('refusal_accuracy', 0.0) * 100:.1f}%",
                        f"{m.get('average_latency_ms', 0.0):.1f}",
                        f"{m.get('hit_at_5', 0.0) * 100:.1f}%",
                        f"{m.get('mrr', 0.0):.4f}",
                        m.get("passed_count", 0),
                        m.get("failed_count", 0),
                        m.get("api_errors", 0),
                        m.get("total_questions", 0),
                    ])
                writer.writerow([])
                writer.writerow(["=== PER-QUESTION DETAILS ==="])
                writer.writerow([
                    "Config",
                    "Question ID",
                    "Status",
                    "Question",
                    "Expected Answer",
                    "Actual Answer",
                    "Latency (ms)",
                    "Hit@5",
                    "First Correct Rank",
                    "MRR (this Q)",
                    "Failure Reasons",
                ])
                for cfg_name, cfg_info in run_record.get("configs", {}).items():
                    for q in cfg_info.get("per_question", []):
                        hit_val = q.get("hit")
                        rank_val = q.get("first_correct_rank")
                        rr_val = q.get("reciprocal_rank")
                        writer.writerow([
                            cfg_name,
                            q.get("id"),
                            q.get("status"),
                            q.get("question"),
                            q.get("expected_answer"),
                            q.get("actual_answer"),
                            q.get("latency_ms"),
                            ("Yes" if hit_val else ("No" if hit_val is False else "N/A")),
                            rank_val if rank_val is not None else "—",
                            f"{rr_val:.4f}" if rr_val is not None else "—",
                            "; ".join(q.get("failure_reasons", [])),
                        ])

        logger.info("Saved evaluation CSV to %s", csv_path)
    except Exception as exc:
        logger.error("Failed to write evaluation CSV to %s: %s", csv_path, exc)


# ---------------------------------------------------------------------------
# Retrieval-only evaluation
# ---------------------------------------------------------------------------


def run_retrieval_only_eval(
    config_names: list[str] | None = None,
    test_set_path: Path | str | None = None,
    results_dir: Path | str | None = None,
    force_reindex: bool = False,
    question_ids: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run retrieval-only evaluation — no LLM calls are made.

    For each answerable question, retrieves chunks under the given config
    and computes:
      * Hit@K  — whether the expected source notice appears in the top-K chunks.
      * MRR    — Mean Reciprocal Rank of the first correct chunk.

    Args:
        config_names:  List of config names to evaluate (default: ['A', 'B']).
        test_set_path: Path to test_set.json.
        results_dir:   Directory to save timestamped JSON and CSV results.
        force_reindex: Force re-indexing separate collections.
        question_ids:  Optional subset of question IDs to evaluate.
        top_k:         K for Hit@K (default 5).

    Returns:
        Dict with run metadata, per-config metrics (hit_at_k, mrr), and
        per-question detail (hit, first_correct_rank, reciprocal_rank).
    """
    save_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = config_names or ["A", "B"]
    test_cases = load_test_set(test_set_path)
    if not test_cases:
        logger.warning("No test cases found to evaluate.")

    if question_ids:
        test_cases = [q for q in test_cases if q.get("id") in question_ids]
        logger.info("Filtering to %d explicitly requested question(s).", len(test_cases))

    source_map = get_notice_source_map()
    config_outputs: dict[str, Any] = {}

    for name in selected_configs:
        try:
            cfg = get_rag_config(name)
        except ValueError as err:
            logger.error("Skipping invalid config '%s': %s", name, err)
            continue

        logger.info(
            "=== [Retrieval-Only] Config %s: %s ===",
            cfg.name,
            cfg.description,
        )

        try:
            reindex_collection_for_config(cfg.name, force=force_reindex)
        except Exception as exc:
            logger.warning(
                "Could not re-index collection '%s' for Config %s: %s",
                cfg.collection_name,
                cfg.name,
                exc,
            )

        mode = cfg.retrieval_mode
        retrieval_results: list[dict[str, Any]] = []
        total_q = len(test_cases)

        for idx, q in enumerate(test_cases, start=1):
            question_text = q.get("question", "")
            logger.info(
                "[%s] Retrieving %d/%d (id=%s): '%.60s…'",
                cfg.name,
                idx,
                total_q,
                q.get("id"),
                question_text,
            )
            try:
                raw_chunks: list[RetrievedChunk] = retrieve(
                    question_text,
                    mode=mode,
                    collection_name=cfg.collection_name,
                )
                chunks, _ = apply_latest_wins(raw_chunks, drop=True)
            except Exception as exc:
                logger.error(
                    "[%s] Retrieval error for question '%s': %s",
                    cfg.name,
                    q.get("id"),
                    exc,
                )
                chunks = []

            chunk_payloads = [
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "notice_id": c.notice_id,
                    "notice_date": c.notice_date,
                    "page": c.page,
                    "score": round(c.score, 6),
                }
                for c in chunks
            ]
            retrieval_results.append({"question_id": q.get("id"), "chunks": chunk_payloads})

        ret_metrics = compute_retrieval_metrics(
            test_cases,
            retrieval_results,
            source_file_map=source_map,
            top_k=top_k,
        )

        config_outputs[cfg.name] = {
            "config_name": cfg.name,
            "description": cfg.description,
            "retrieval_mode": cfg.retrieval_mode,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "collection_name": cfg.collection_name,
            "metrics": {
                "hit_at_k": ret_metrics["hit_at_k"],
                "mrr": ret_metrics["mrr"],
                "k": top_k,
                "total_answerable": ret_metrics["total_answerable"],
                "hit_count": ret_metrics["hit_count"],
                "total_questions": len(test_cases),
            },
            "per_question": ret_metrics["question_results"],
        }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"eval_{timestamp_str}_retrieval"
    run_record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "run_mode": "retrieval_only",
        "top_k": top_k,
        "total_test_cases": len(test_cases),
        "configs": config_outputs,
    }

    out_file = save_dir / f"{run_id}.json"
    try:
        out_file.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        logger.info("Saved retrieval-only evaluation run to %s", out_file)
    except Exception as exc:
        logger.error("Failed to write retrieval-only results to %s: %s", out_file, exc)

    csv_file = save_dir / f"{run_id}.csv"
    save_results_csv(run_record, csv_file)

    return run_record


# ---------------------------------------------------------------------------
# Full RAG evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    config_names: list[str] | None = None,
    delay_seconds: float = 7.0,
    test_set_path: Path | str | None = None,
    results_dir: Path | str | None = None,
    llm_client: BaseLLMClient | None = None,
    force_reindex: bool = False,
    rerun_failed_only: bool = False,
    previous_run_id: str | None = None,
    question_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run evaluation benchmark across given configurations sequentially.

    Enhancements vs previous version:
    * **Quota-error abort**: if a quota/billing error is detected after retries,
      the run is aborted immediately.  Partial results for already-evaluated
      questions are saved, and the returned dict contains ``aborted_reason``.
    * **LLM answer cache**: answers are cached by
      ``(question, frozenset(chunk_ids), config)`` for the duration of the
      process.  This means re-running (e.g. after a quota abort) never repeats
      a call that already succeeded.
    * **Single config**: pass a single-element list (e.g. ``["A"]``) to run
      only that config.

    Args:
        config_names:      List of config names (default: ['A', 'B']).
        delay_seconds:     Pause between LLM calls (default: 7.0 seconds).
        test_set_path:     Path to test_set.json file.
        results_dir:       Directory to save timestamped JSON and CSV results.
        llm_client:        Optional mock LLM client for tests.
        force_reindex:     Whether to force re-indexing of separate collections.
        rerun_failed_only: If True, only re-run questions that failed or errored
                           in a previous run.
        previous_run_id:   Optional run_id of a previous run to extract failed
                           questions from.
        question_ids:      Optional explicit subset of question IDs to run.

    Returns:
        Dict containing evaluation results, metrics, and per-question details.
        If the run was aborted due to a quota error, the dict contains
        ``aborted_reason`` and ``aborted_after`` keys.
    """
    save_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = config_names or ["A", "B"]
    test_cases = load_test_set(test_set_path)
    if not test_cases:
        logger.warning("No test cases found to evaluate.")

    # ── Handle Re-run failed/error questions only option ─────────────────────
    if question_ids:
        test_cases = [q for q in test_cases if q.get("id") in question_ids]
        logger.info("Filtering to %d explicitly requested question(s).", len(test_cases))
    elif rerun_failed_only:
        prev_data: dict[str, Any] | None = None
        if previous_run_id:
            prev_file = save_dir / f"{previous_run_id}.json"
            if prev_file.exists():
                try:
                    prev_data = json.loads(prev_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Could not read previous run %s: %s", prev_file, exc)
        else:
            prev_files = sorted(save_dir.glob("eval_*.json"), key=lambda f: f.name, reverse=True)
            if prev_files:
                try:
                    prev_data = json.loads(prev_files[0].read_text(encoding="utf-8"))
                    logger.info("Using latest run %s for failed questions", prev_files[0].name)
                except Exception as exc:
                    logger.warning("Could not read latest run %s: %s", prev_files[0], exc)

        if prev_data:
            failed_qids = get_failed_question_ids_from_run(prev_data)
            test_cases = [q for q in test_cases if q.get("id") in failed_qids]
            logger.info("Re-running %d failed/error question(s) only: %s", len(test_cases), sorted(failed_qids))
        else:
            logger.warning("No previous run found; executing all questions.")

    source_map = get_notice_source_map()
    config_outputs: dict[str, Any] = {}

    # ── LLM answer cache  ─────────────────────────────────────────────────────
    # Key: (question_text, frozenset(chunk_ids), config_name)
    # Value: the raw answer string that was already returned by the LLM.
    # This cache persists for the entire process lifetime; reruns (e.g. after a
    # quota abort) therefore never repeat a call that already succeeded.
    _answer_cache: dict[tuple[str, frozenset[str], str], str] = {}

    # ── Abort flag shared across config loops ─────────────────────────────────
    abort_reason: str | None = None
    aborted_after_questions: int = 0

    # ── Run configs one after another, never concurrently ────────────────────
    for name in selected_configs:
        if abort_reason:
            logger.warning("Skipping config '%s' because run was already aborted.", name)
            break

        try:
            cfg = get_rag_config(name)
        except ValueError as err:
            logger.error("Skipping invalid config '%s': %s", name, err)
            continue

        logger.info(
            "=== Starting Evaluation for Config %s: %s ===",
            cfg.name,
            cfg.description,
        )

        # Ensure separate Chroma collection is populated if needed
        try:
            reindex_collection_for_config(cfg.name, force=force_reindex)
        except Exception as exc:
            logger.warning(
                "Could not re-index collection '%s' for Config %s: %s",
                cfg.collection_name,
                cfg.name,
                exc,
            )

        run_results: list[dict[str, Any]] = []
        total_q = len(test_cases)
        questions_done_this_cfg = 0

        for idx, q in enumerate(test_cases, start=1):
            question_text = q.get("question", "")
            logger.info(
                "[%s] Evaluating %d/%d (id=%s): '%.60s…'",
                cfg.name,
                idx,
                total_q,
                q.get("id"),
                question_text,
            )

            # Retry 429 or rate-limit errors with exponential backoff up to 3 times
            max_retries = 3
            attempt = 0
            res: dict[str, Any] | None = None
            question_error: str | None = None

            while attempt <= max_retries:
                attempt += 1
                call_exc: Exception | None = None

                # ── Check cache before calling LLM ────────────────────────
                # We need the chunk IDs to build the cache key, so we run
                # retrieval first (which answer_question does internally too).
                # For cache purposes we replicate the key here cheaply.
                try:
                    res = answer_question(
                        question=question_text,
                        config=cfg.name,
                        collection_name=cfg.collection_name,
                        llm_client=llm_client,
                    )

                    # Build a cache key from the returned chunk IDs
                    chunk_ids = frozenset(
                        c.get("chunk_id", "") for c in res.get("chunks", [])
                    )
                    cache_key = (question_text, chunk_ids, cfg.name)

                    # If this exact (question, context, config) succeeded before,
                    # we would have served from cache — record it now for future reruns
                    if cache_key not in _answer_cache and res.get("answer"):
                        ans_str = str(res["answer"])
                        # Only cache genuine non-error answers
                        act_lower = ans_str.lower()
                        if not (
                            act_lower.startswith("api error")
                            or act_lower.startswith("i could not generate a response due to an api error")
                            or any(t in act_lower for t in ["429", "rate limit", "rate_limit", "too many requests"])
                        ):
                            _answer_cache[cache_key] = ans_str

                except Exception as exc:
                    call_exc = exc

                err_target = call_exc if call_exc is not None else (
                    res.get("error") if res else None
                )
                if err_target is None and res:
                    ans_text = str(res.get("answer", ""))
                    if is_rate_limit_error(ans_text):
                        err_target = ans_text

                # ── Quota error: abort immediately ────────────────────────
                if err_target is not None and is_quota_error(err_target):
                    abort_reason = (
                        f"Run stopped: API quota exhausted after "
                        f"{aborted_after_questions} questions"
                    )
                    logger.error(
                        "[%s] Quota error — aborting run: %s", cfg.name, str(err_target)
                    )
                    question_error = str(err_target)
                    break

                if err_target is not None and is_rate_limit_error(err_target):
                    if attempt <= max_retries:
                        backoff_base = 2.0 if delay_seconds > 0 else 0.0
                        backoff_time = backoff_base * (2 ** (attempt - 1))
                        logger.warning(
                            "[%s] Rate-limit/429 error on question '%s' (attempt %d/%d). Retrying in %.1fs: %s",
                            cfg.name,
                            q.get("id"),
                            attempt,
                            max_retries + 1,
                            backoff_time,
                            str(err_target),
                        )
                        if backoff_time > 0:
                            time.sleep(backoff_time)
                        continue
                    else:
                        logger.error(
                            "[%s] Rate-limit/429 error persisted after %d retries for question '%s': %s",
                            cfg.name,
                            max_retries,
                            q.get("id"),
                            str(err_target),
                        )
                        question_error = str(err_target)
                        break
                else:
                    if call_exc is not None:
                        question_error = str(call_exc)
                        logger.error(
                            "[%s] Error evaluating question '%s' on Config %s: %s",
                            cfg.name,
                            q.get("id"),
                            cfg.name,
                            call_exc,
                        )
                    elif res:
                        ans_str = str(res.get("answer", "")).lower()
                        if ans_str.startswith("i could not generate a response due to an api error") or ans_str.startswith("api error"):
                            question_error = str(res.get("answer"))
                    break

            if res is None:
                res = {
                    "answer": f"API Error: {question_error}",
                    "citations": [],
                    "chunks": [],
                    "notes": [],
                    "latency_ms": 0,
                    "config": cfg.name,
                    "error": question_error or "Unknown error",
                }
            elif question_error:
                res["error"] = question_error

            # Log the number of retrieved chunks
            retrieved_chunk_count = len(res.get("chunks", []))
            logger.info(
                "[%s] Question %s (%d/%d): retrieved %d chunks",
                cfg.name,
                q.get("id"),
                idx,
                total_q,
                retrieved_chunk_count,
            )

            run_results.append(res)
            questions_done_this_cfg += 1
            aborted_after_questions += 1

            # Abort immediately — do not process any more questions
            if abort_reason:
                break

            # Rate limiting / polite delay between LLM calls
            if delay_seconds > 0 and idx < total_q:
                time.sleep(delay_seconds)

        metrics = compute_metrics(test_cases[:questions_done_this_cfg], run_results, source_file_map=source_map)

        # ── Aggregate Hit@5 / MRR from per-question retrieval data ───────────
        answerable_with_src = [
            q for q in metrics["question_results"]
            if q.get("answerable") and q.get("expected_source")
        ]
        hit_at_5_count = sum(1 for q in answerable_with_src if q.get("hit") is True)
        hit_at_5 = round(hit_at_5_count / len(answerable_with_src), 4) if answerable_with_src else 0.0
        mrr_sum = sum(q.get("reciprocal_rank") or 0.0 for q in answerable_with_src)
        mrr = round(mrr_sum / len(answerable_with_src), 4) if answerable_with_src else 0.0

        config_outputs[cfg.name] = {
            "config_name": cfg.name,
            "description": cfg.description,
            "retrieval_mode": cfg.retrieval_mode,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "collection_name": cfg.collection_name,
            "metrics": {
                "answer_accuracy": metrics["answer_accuracy"],
                "citation_hit_rate": metrics["citation_hit_rate"],
                "refusal_accuracy": metrics["refusal_accuracy"],
                "average_latency_ms": metrics["average_latency_ms"],
                "hit_at_5": hit_at_5,
                "mrr": mrr,
                "total_questions": metrics["total_questions"],
                "answerable_count": metrics["answerable_count"],
                "unanswerable_count": metrics["unanswerable_count"],
                "passed_count": metrics["passed_count"],
                "failed_count": metrics["failed_count"],
                "api_errors": metrics["api_errors"],
            },
            "per_question": metrics["question_results"],
        }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"eval_{timestamp_str}"
    run_record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "run_mode": "full",
        "total_test_cases": len(test_cases),
        "delay_seconds": delay_seconds,
        "rerun_failed_only": rerun_failed_only,
        "configs": config_outputs,
    }
    if abort_reason:
        run_record["aborted_reason"] = abort_reason
        run_record["aborted_after"] = aborted_after_questions
        logger.warning("Run aborted: %s", abort_reason)

    # ── Persist JSON and CSV to eval/results/ ─────────────────────────────────
    out_file = save_dir / f"{run_id}.json"
    try:
        out_file.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        logger.info("Saved evaluation run to %s", out_file)
    except Exception as exc:
        logger.error("Failed to write evaluation results to %s: %s", out_file, exc)

    csv_file = save_dir / f"{run_id}.csv"
    save_results_csv(run_record, csv_file)

    return run_record
