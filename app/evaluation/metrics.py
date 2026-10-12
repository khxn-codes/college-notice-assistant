"""Evaluation metrics for RAG benchmark suite.

Computes:
1. Answer accuracy: every string in expected_answer_contains appears in the answer
   (case-insensitive) for answerable questions.
2. Citation hit rate: expected source appears among the citations.
3. Refusal accuracy: unanswerable questions get exactly:
   "I could not find this in the available notices."
4. Average latency: average latency in milliseconds across questions.
5. Hit@5: whether the expected source appears among the top 5 retrieved chunks
   (retrieval-only mode, answerable questions only).
6. MRR: Mean Reciprocal Rank of the first chunk from the expected source
   (retrieval-only mode, answerable questions only).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

STANDARD_REFUSAL = "I could not find this in the available notices."


def check_answer_accuracy(answer: str, expected_contains: list[str]) -> bool:
    """Check if every string in expected_contains appears in answer (case-insensitive).

    Args:
        answer: The text answer produced by the RAG pipeline.
        expected_contains: List of substring keywords expected in the answer.

    Returns:
        True if all substrings are found, False otherwise.
    """
    if not expected_contains:
        return True
    ans_lower = answer.lower()
    return all(substr.lower() in ans_lower for substr in expected_contains)


def check_citation_hit(
    expected_source: str | None,
    citations: list[dict[str, Any]],
    source_file_map: dict[str, str] | None = None,
) -> bool:
    """Check if expected_source appears among the returned citations.

    Matches by:
    - Direct source_file field if present in citation dict
    - Mapped source_file via notice_id (from SQLite notices)
    - Normalized title / filename stem similarity

    Args:
        expected_source: Expected notice filename (e.g., '2026-09-18_admin_class-timings-revised.pdf').
        citations: List of citation dicts returned by the RAG pipeline.
        source_file_map: Optional mapping of notice_id -> source_file.

    Returns:
        True if the expected source notice is cited, False otherwise.
    """
    if not expected_source:
        return False

    exp_clean = expected_source.lower().strip()
    exp_stem = exp_clean.rsplit(".", 1)[0]
    # Tokenize stem (e.g. "2026-09-18", "admin", "class", "timings", "revised")
    stem_words = [w for w in exp_stem.replace("-", " ").replace("_", " ").split() if len(w) > 2]

    for c in citations:
        # 1. Direct source_file key in citation
        src = str(c.get("source_file", "")).lower().strip()
        if src and (exp_clean in src or src in exp_clean):
            return True

        # 2. Look up notice_id in source_file_map
        nid = str(c.get("notice_id", "")).strip()
        if source_file_map and nid in source_file_map:
            mapped_src = source_file_map[nid].lower().strip()
            if exp_clean in mapped_src or mapped_src in exp_clean:
                return True

        # 3. Match citation title
        title = str(c.get("title", "")).lower().strip()
        if title:
            # Check if title matches words in expected stem
            title_clean = title.replace("-", " ").replace("_", " ")
            if title_clean in exp_stem or exp_stem in title_clean:
                return True
            # Check if title has significant word overlap with expected stem
            title_words = [w for w in title_clean.split() if len(w) > 2]
            if title_words and sum(1 for w in title_words if w in stem_words) >= min(2, len(title_words)):
                return True

    return False


def _source_matches_chunk(
    expected_source: str,
    chunk: dict[str, Any],
    source_file_map: dict[str, str] | None = None,
) -> bool:
    """Check if a single retrieved chunk comes from the expected source notice.

    Uses the same matching logic as check_citation_hit but for a single chunk.

    Args:
        expected_source: Expected notice filename.
        chunk: A chunk dict with keys like source_file, notice_id, title.
        source_file_map: Optional mapping of notice_id -> source_file.

    Returns:
        True if the chunk originates from the expected source notice.
    """
    exp_clean = expected_source.lower().strip()
    exp_stem = exp_clean.rsplit(".", 1)[0]
    stem_words = [w for w in exp_stem.replace("-", " ").replace("_", " ").split() if len(w) > 2]

    # 1. Direct source_file key
    src = str(chunk.get("source_file", "")).lower().strip()
    if src and (exp_clean in src or src in exp_clean):
        return True

    # 2. Look up notice_id in source_file_map
    nid = str(chunk.get("notice_id", "")).strip()
    if source_file_map and nid in source_file_map:
        mapped_src = source_file_map[nid].lower().strip()
        if exp_clean in mapped_src or mapped_src in exp_clean:
            return True

    # 3. Match title
    title = str(chunk.get("title", "")).lower().strip()
    if title:
        title_clean = title.replace("-", " ").replace("_", " ")
        if title_clean in exp_stem or exp_stem in title_clean:
            return True
        title_words = [w for w in title_clean.split() if len(w) > 2]
        if title_words and sum(1 for w in title_words if w in stem_words) >= min(2, len(title_words)):
            return True

    return False


def compute_retrieval_metrics(
    test_cases: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    source_file_map: dict[str, str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute Hit@K and MRR for retrieval-only evaluation (no LLM calls).

    Only considers answerable questions that have an expected_source.
    Unanswerable questions are skipped and do not affect the denominators.

    Args:
        test_cases: Ground-truth test cases from test_set.json.
        retrieval_results: Dicts with keys ``chunks`` (list of chunk dicts,
            ranked by retrieval score, highest first) and ``question_id``.
        source_file_map: Optional mapping of notice_id -> source_file.
        top_k: The k used for Hit@K (default 5).

    Returns:
        Dict with ``hit_at_k``, ``mrr``, ``k``, ``total_answerable``,
        ``hit_count``, and ``question_results`` (list of per-question detail).
    """
    total_q = min(len(test_cases), len(retrieval_results))
    if total_q == 0:
        return {
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "k": top_k,
            "total_answerable": 0,
            "hit_count": 0,
            "question_results": [],
        }

    answerable_count = 0
    hit_count = 0
    rr_sum = 0.0
    question_results: list[dict[str, Any]] = []

    for i in range(total_q):
        tc = test_cases[i]
        res = retrieval_results[i]

        qid = tc.get("id", f"q{i+1:02d}")
        is_answerable = bool(tc.get("answerable", True))
        expected_src = tc.get("expected_source")

        # Only score answerable questions with an expected source
        if not is_answerable or not expected_src:
            question_results.append({
                "id": qid,
                "question": tc.get("question", ""),
                "answerable": is_answerable,
                "expected_source": expected_src,
                "hit": None,
                "first_correct_rank": None,
                "reciprocal_rank": None,
                "chunks": [],
            })
            continue

        answerable_count += 1
        chunks = res.get("chunks", [])
        top_chunks = chunks[:top_k]

        # Find the rank (1-indexed) of the first chunk from the expected source
        first_correct_rank: int | None = None
        for rank, chunk in enumerate(top_chunks, start=1):
            if _source_matches_chunk(expected_src, chunk, source_file_map):
                first_correct_rank = rank
                break

        hit = first_correct_rank is not None
        rr = 1.0 / first_correct_rank if first_correct_rank is not None else 0.0

        if hit:
            hit_count += 1
        rr_sum += rr

        question_results.append({
            "id": qid,
            "question": tc.get("question", ""),
            "answerable": is_answerable,
            "expected_source": expected_src,
            "hit": hit,
            "first_correct_rank": first_correct_rank,
            "reciprocal_rank": round(rr, 6),
            "chunks": [
                {
                    "rank": r + 1,
                    "chunk_id": c.get("chunk_id", ""),
                    "title": c.get("title", ""),
                    "notice_id": c.get("notice_id", ""),
                }
                for r, c in enumerate(top_chunks)
            ],
        })

    hit_at_k = round(hit_count / answerable_count, 4) if answerable_count > 0 else 0.0
    mrr = round(rr_sum / answerable_count, 4) if answerable_count > 0 else 0.0

    return {
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "k": top_k,
        "total_answerable": answerable_count,
        "hit_count": hit_count,
        "question_results": question_results,
    }


def check_refusal_accuracy(answer: str) -> bool:
    """Check if the answer matches exactly the required refusal string.

    Args:
        answer: The text answer produced by the RAG pipeline.

    Returns:
        True if answer.strip() == "I could not find this in the available notices."
    """
    return answer.strip() == STANDARD_REFUSAL


def compute_metrics(
    test_cases: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
    source_file_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute benchmark metrics across test cases and their run results.

    Args:
        test_cases: List of ground-truth test cases from test_set.json.
        run_results: List of execution results from the RAG pipeline.
        source_file_map: Optional mapping of notice_id -> source_file.

    Returns:
        Dictionary containing aggregate metrics and per-question evaluations.
        Each entry in ``question_results`` also carries ``hit``,
        ``first_correct_rank``, and ``reciprocal_rank`` fields populated from
        the retrieval leg (chunks) so the CSV and UI can display them even for
        full-RAG runs.
    """
    total_q = min(len(test_cases), len(run_results))
    if total_q == 0:
        return {
            "answer_accuracy": 0.0,
            "citation_hit_rate": 0.0,
            "refusal_accuracy": 0.0,
            "average_latency_ms": 0.0,
            "total_questions": 0,
            "answerable_count": 0,
            "unanswerable_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "api_errors": 0,
            "question_results": [],
        }

    question_results: list[dict[str, Any]] = []
    answerable_total = 0
    answerable_acc_hits = 0

    citation_total = 0
    citation_hits = 0

    unanswerable_total = 0
    refusal_hits = 0

    api_errors_count = 0
    total_latency_ms = 0.0

    for i in range(total_q):
        tc = test_cases[i]
        res = run_results[i]

        qid = tc.get("id", f"q{i+1:02d}")
        question = tc.get("question", "")
        is_answerable = bool(tc.get("answerable", True))
        expected_contains = tc.get("expected_answer_contains", [])
        expected_src = tc.get("expected_source")

        actual_ans = res.get("answer", "")
        citations = res.get("citations", [])

        # Retrieval metrics (computed from retrieved chunks, non-null only for answerable
        # questions that have an expected_source)
        raw_chunks = res.get("chunks", [])
        ret_hit: bool | None = None
        ret_first_rank: int | None = None
        ret_rr: float | None = None
        if is_answerable and expected_src:
            for rank, chunk in enumerate(raw_chunks[:5], start=1):
                if _source_matches_chunk(expected_src, chunk, source_file_map):
                    ret_first_rank = rank
                    break
            ret_hit = ret_first_rank is not None
            ret_rr = round(1.0 / ret_first_rank, 6) if ret_first_rank is not None else 0.0
        latency_ms = float(res.get("latency_ms", 0.0))
        error_msg = res.get("error")

        total_latency_ms += latency_ms

        ans_acc_pass: bool | None = None
        cite_hit_pass: bool | None = None
        refusal_pass: bool | None = None
        failure_reasons: list[str] = []

        # Check whether this question experienced an API / pipeline error
        act_lower = actual_ans.lower().strip()
        is_api_err = False
        api_err_reason = error_msg

        if error_msg:
            is_api_err = True
        elif act_lower.startswith("i could not generate a response due to an api error"):
            is_api_err = True
            api_err_reason = actual_ans
        elif act_lower.startswith("api error"):
            is_api_err = True
            api_err_reason = actual_ans
        elif any(token in act_lower for token in ["429", "rate limit", "rate_limit", "too many requests"]):
            is_api_err = True
            api_err_reason = actual_ans

        if is_api_err:
            api_errors_count += 1
            failure_reasons.append(f"API Error: {api_err_reason}")
            overall_pass = False
            status_str = "ERROR"
        elif is_answerable:
            answerable_total += 1
            ans_acc_pass = check_answer_accuracy(actual_ans, expected_contains)
            if ans_acc_pass:
                answerable_acc_hits += 1
            else:
                failure_reasons.append("Missing expected keyword(s)")

            if expected_src:
                citation_total += 1
                cite_hit_pass = check_citation_hit(expected_src, citations, source_file_map)
                if cite_hit_pass:
                    citation_hits += 1
                else:
                    failure_reasons.append("Expected source notice not cited")

            # Overall pass for answerable requires both answer accuracy and citation hit (if expected)
            overall_pass = bool(ans_acc_pass and (cite_hit_pass is not False))
            status_str = "PASS" if overall_pass else "FAIL"
        else:
            unanswerable_total += 1
            refusal_pass = check_refusal_accuracy(actual_ans)
            if refusal_pass:
                refusal_hits += 1
            else:
                failure_reasons.append("Did not return exact standard refusal message")

            overall_pass = bool(refusal_pass)
            status_str = "PASS" if overall_pass else "FAIL"

        expected_display = (
            ", ".join(expected_contains) if (is_answerable and expected_contains)
            else (STANDARD_REFUSAL if not is_answerable else "—")
        )

        question_results.append(
            {
                "id": qid,
                "question": question,
                "answerable": is_answerable,
                "expected_answer": expected_display,
                "expected_answer_contains": expected_contains,
                "expected_source": expected_src,
                "actual_answer": actual_ans,
                "citations": citations,
                "latency_ms": int(latency_ms),
                "answer_accuracy_pass": ans_acc_pass,
                "citation_hit_pass": cite_hit_pass,
                "refusal_pass": refusal_pass,
                "passed": overall_pass,
                "is_error": is_api_err,
                "status": status_str,
                "failure_reasons": failure_reasons,
                # Retrieval-quality fields (populated for answerable questions with expected_source)
                "hit": ret_hit,
                "first_correct_rank": ret_first_rank,
                "reciprocal_rank": ret_rr,
            }
        )

    ans_accuracy = round(answerable_acc_hits / answerable_total, 4) if answerable_total > 0 else 0.0
    cite_rate = round(citation_hits / citation_total, 4) if citation_total > 0 else 0.0
    ref_accuracy = round(refusal_hits / unanswerable_total, 4) if unanswerable_total > 0 else 0.0
    avg_latency = round(total_latency_ms / total_q, 2) if total_q > 0 else 0.0

    passed_count = sum(1 for q in question_results if q["passed"])
    failed_count = sum(1 for q in question_results if q["status"] == "FAIL")

    return {
        "answer_accuracy": ans_accuracy,
        "citation_hit_rate": cite_rate,
        "refusal_accuracy": ref_accuracy,
        "average_latency_ms": avg_latency,
        "total_questions": total_q,
        "answerable_count": answerable_total,
        "unanswerable_count": unanswerable_total,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "api_errors": api_errors_count,
        "question_results": question_results,
    }
