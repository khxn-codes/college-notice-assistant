"""Streamlit UI for NoticeIQ: RAG-Based AI Assistant for Answering College Notice Queries with Citations.

Connects to the FastAPI backend over HTTP. Does not import backend code directly.
Features & Optimizations:
  1. Simple, lightweight CSS with no animations, transitions, blurs, or fixed positioning.
  2. Short-TTL caching via @st.cache_data for notices & health, cleared on upload/sync/edit.
  3. No unnecessary work at script top-level on reruns; no loop triggers for st.rerun().
  4. Renders only the 10 most recent chat messages by default with an expand button.
  5. Expandable "Sources" sections are collapsed by default with chunk text rendered via st.text.
  6. Fixed-height virtualised st.dataframe for the notices library.
"""

from __future__ import annotations

import os
from typing import Any
import httpx
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")

st.set_page_config(
    page_title="NoticeIQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lightweight CSS: simple fonts & badges, no blur/shadows/animations/transitions
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #f1f3f5;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 2px 6px;
        margin: 2px 4px 2px 0;
        font-size: 0.8rem;
        color: #212529;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Backend API Calls
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_health() -> bool:
    """Return True if backend /health responds with 200 (cached 10s)."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=15, show_spinner=False)
def get_notices() -> list[dict[str, Any]]:
    """Fetch list of indexed notices from GET /notices (cached 15s)."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/notices", timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("notices", [])
        return []
    except Exception:
        return []


def clear_notices_cache() -> None:
    """Invalidate notices and health cache after mutations."""
    get_notices.clear()
    check_backend_health.clear()


def upload_notice_file(file_name: str, file_bytes: bytes, mime_type: str = "application/pdf") -> dict[str, Any]:
    """Upload a single notice file via POST /notices/upload."""
    files = {"file": (file_name, file_bytes, mime_type)}
    resp = httpx.post(f"{BACKEND_URL}/notices/upload", files=files, timeout=60.0)
    resp.raise_for_status()
    clear_notices_cache()
    return resp.json()


def sync_inbox_notices() -> dict[str, Any]:
    """Trigger inbox ingestion via POST /notices/sync."""
    resp = httpx.post(f"{BACKEND_URL}/notices/sync", timeout=60.0)
    resp.raise_for_status()
    clear_notices_cache()
    return resp.json()


def ask_question(question: str, category: str | None = None) -> dict[str, Any]:
    """Ask a question via POST /ask."""
    payload: dict[str, Any] = {"question": question}
    if category and category != "All":
        payload["category"] = category
    resp = httpx.post(f"{BACKEND_URL}/ask", json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def update_notice_metadata(notice_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Update notice metadata via PATCH /notices/{id}."""
    resp = httpx.patch(f"{BACKEND_URL}/notices/{notice_id}", json=updates, timeout=15.0)
    resp.raise_for_status()
    clear_notices_cache()
    return resp.json()


def delete_notice_api(notice_id: str) -> dict[str, Any]:
    """Delete a notice and all its chunks via DELETE /notices/{id}."""
    resp = httpx.delete(f"{BACKEND_URL}/notices/{notice_id}", timeout=15.0)
    resp.raise_for_status()
    clear_notices_cache()
    return resp.json()


def format_api_error_for_ui(err_msg: str) -> str:
    """Classify and format API error with real cause (rate limit, quota, model not found, auth), key never shown."""
    text = str(err_msg).lower()
    if any(k in text for k in ["rate limit", "rate_limit", "too many requests", "429"]):
        cause = "rate limit"
    elif any(k in text for k in ["quota", "resourceexhausted", "resource exhausted"]):
        cause = "quota"
    elif any(k in text for k in ["not found", "notfound", "404", "model not found"]):
        cause = "model not found"
    elif any(k in text for k in ["auth", "unauthorized", "unauthenticated", "401", "403", "forbidden", "permission", "api_key", "api key"]):
        cause = "auth"
    else:
        cause = "API error"
    return f"API error ({cause})"


def run_evaluation_api(
    configs: list[str],
    delay_seconds: float = 7.0,
    rerun_failed_only: bool = False,
    previous_run_id: str | None = None,
    retrieval_only: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Trigger evaluation via POST /eval/run."""
    payload: dict[str, Any] = {
        "configs": configs,
        "delay_seconds": delay_seconds,
        "rerun_failed_only": rerun_failed_only,
        "retrieval_only": retrieval_only,
        "top_k": top_k,
    }
    if previous_run_id:
        payload["previous_run_id"] = previous_run_id
    resp = httpx.post(
        f"{BACKEND_URL}/eval/run",
        json=payload,
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_evaluation_results_api() -> list[dict[str, Any]]:
    """Fetch previous evaluation runs via GET /eval/results."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/eval/results", timeout=15.0)
        if resp.status_code == 200:
            return resp.json().get("runs", [])
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sidebar Navigation & Backend Status
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 🎓 NoticeIQ")
page = st.sidebar.radio(
    "Navigation",
    ["💬 Chat", "📋 Notices", "📊 Evaluation"],
    index=0,
)

st.sidebar.markdown("---")
is_healthy = check_backend_health()
if is_healthy:
    st.sidebar.success(f"Backend Connected\n`{BACKEND_URL}`")
else:
    st.sidebar.error(
        f"Backend Unreachable\n`{BACKEND_URL}`\n"
        "Please start the API:\n"
        "`.\.run.ps1` (Windows) or `./run.sh` (Linux/macOS)"
    )

st.sidebar.markdown("---")
st.sidebar.caption("NoticeIQ • AI College Notice Assistant")


# ---------------------------------------------------------------------------
# Page: 💬 Chat
# ---------------------------------------------------------------------------

if page == "💬 Chat":
    st.markdown('<div class="main-title">💬 NoticeIQ Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">NoticeIQ: RAG-Based AI Assistant for Answering College Notice Queries with Citations. '
        'Every answer is grounded in official notices with citations.</div>',
        unsafe_allow_html=True,
    )

    if not is_healthy:
        st.error(
            f"⚠️ **Backend service is not running.**\n\n"
            f"Start both services with one command:\n"
            f"- **Windows:** `.\\run.ps1`\n"
            f"- **Linux/macOS:** `./run.sh`\n\n"
            f"The UI expects the API at `{BACKEND_URL}`."
        )

    # Category dropdown filled dynamically from cached GET /notices
    notices = get_notices() if is_healthy else []
    categories = ["All"]
    if notices:
        discovered_cats = sorted(list({n["category"] for n in notices if n.get("category")}))
        categories.extend(discovered_cats)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Answers are generated strictly from indexed notices. When revised notices exist, the newest notice is preferred.")
    with col2:
        selected_cat = st.selectbox("Category filter:", categories, index=0)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    all_messages = st.session_state.messages
    total_msgs = len(all_messages)

    # Render only the most recent 10 messages by default
    show_all = st.session_state.get("show_all_messages", False)
    if total_msgs > 10 and not show_all:
        if st.button(f"📜 Show earlier messages ({total_msgs - 10} hidden)", key="btn_show_earlier"):
            st.session_state.show_all_messages = True
            st.rerun()
        visible_messages = all_messages[-10:]
    elif total_msgs > 10 and show_all:
        if st.button("Collapse earlier messages", key="btn_collapse_earlier"):
            st.session_state.show_all_messages = False
            st.rerun()
        visible_messages = all_messages
    else:
        visible_messages = all_messages

    # Display visible chat messages
    for msg in visible_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("notes"):
                for note in msg["notes"]:
                    st.info(f"🔄 **Notice Update:** {note}")

            if msg.get("citations"):
                st.markdown("**Citations:**")
                citation_html = "".join(
                    [
                        f'<span class="citation-badge">📄 {c.get("title")} ({c.get("notice_date")}, p.{c.get("page")})</span>'
                        for c in msg["citations"]
                    ]
                )
                st.markdown(citation_html, unsafe_allow_html=True)

            if msg.get("chunks"):
                with st.expander(f"📚 Sources ({len(msg['chunks'])})", expanded=False):
                    for i, chunk in enumerate(msg["chunks"], start=1):
                        st.markdown(
                            f"**[{i}] {chunk.get('title')}** — *{chunk.get('notice_date')} (p.{chunk.get('page')})*"
                        )
                        st.text(chunk.get("text", ""))

    # User chat input
    user_query = st.chat_input("Ask a question about college notices (e.g. When do midterms start?)...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if not is_healthy:
                err_msg = (
                    "Cannot reach backend API. "
                    "Start it with `.\\run.ps1` (Windows) or `./run.sh` (Linux/macOS)."
                )
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            else:
                with st.spinner("Searching notices and generating verified answer …"):
                    try:
                        cat_arg = selected_cat if selected_cat != "All" else None
                        result = ask_question(question=user_query, category=cat_arg)
                        answer_text = result.get("answer", "No answer received.")
                        citations = result.get("citations", [])
                        chunks = result.get("chunks", [])
                        notes = result.get("notes", [])

                        st.markdown(answer_text)

                        if notes:
                            for note in notes:
                                st.info(f"🔄 **Notice Update:** {note}")

                        if citations:
                            st.markdown("**Citations:**")
                            citation_html = "".join(
                                [
                                    f'<span class="citation-badge">📄 {c.get("title")} ({c.get("notice_date")}, p.{c.get("page")})</span>'
                                    for c in citations
                                ]
                            )
                            st.markdown(citation_html, unsafe_allow_html=True)

                        if chunks:
                            with st.expander(f"📚 Sources ({len(chunks)})", expanded=False):
                                for i, chunk in enumerate(chunks, start=1):
                                    st.markdown(
                                        f"**[{i}] {chunk.get('title')}** — *{chunk.get('notice_date')} (p.{chunk.get('page')})*"
                                    )
                                    st.text(chunk.get("text", ""))

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations,
                                "chunks": chunks,
                                "notes": notes,
                            }
                        )
                    except httpx.HTTPStatusError as http_err:
                        err_text = f"Backend returned an error ({http_err.response.status_code}): {format_api_error_for_ui(http_err.response.text)}"
                        st.error(err_text)
                        st.session_state.messages.append({"role": "assistant", "content": err_text})
                    except Exception as exc:
                        err_text = f"An error occurred while querying notices: {format_api_error_for_ui(str(exc))}"
                        st.error(err_text)
                        st.session_state.messages.append({"role": "assistant", "content": err_text})

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.show_all_messages = False
        st.rerun()


# ---------------------------------------------------------------------------
# Page: 📋 Notices
# ---------------------------------------------------------------------------

elif page == "📋 Notices":
    st.markdown('<div class="main-title">📋 Notice Management & Library</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload new notices, synchronise files from data/inbox/, and inspect the indexed library.</div>',
        unsafe_allow_html=True,
    )

    if not is_healthy:
        st.error(f"⚠️ Backend is not running at `{BACKEND_URL}`. Start FastAPI to manage notices.")

    col_up, col_sync = st.columns([1, 1])

    with col_up:
        st.subheader("📤 Upload Notice")
        st.caption("Upload a single PDF or TXT notice to immediately extract, chunk, and index it.")
        uploaded_file = st.file_uploader("Select file", type=["pdf", "txt"], key="upload_notice_uploader")
        if st.button("Upload & Index", type="primary", disabled=not is_healthy or uploaded_file is None):
            if uploaded_file is not None:
                with st.spinner(f"Ingesting '{uploaded_file.name}' …"):
                    try:
                        mime = "application/pdf" if uploaded_file.name.lower().endswith(".pdf") else "text/plain"
                        res = upload_notice_file(
                            file_name=uploaded_file.name,
                            file_bytes=uploaded_file.getvalue(),
                            mime_type=mime,
                        )
                        status = res.get("status")
                        if status == "indexed":
                            msg = f"✅ Successfully indexed **{res.get('title')}** ({res.get('chunks')} chunks)."
                            if res.get("supersedes_id"):
                                msg += f" Supersedes older notice ID `{res.get('supersedes_id')}`."
                            st.success(msg)
                        elif status == "duplicate":
                            st.info(f"ℹ️ File is already indexed: **{res.get('title')}**.")
                        elif status == "empty":
                            st.warning(f"⚠️ No text could be extracted from **{res.get('title')}**.")
                        else:
                            st.write(res)
                    except Exception as e:
                        st.error(f"Failed to upload: {e}")

    with col_sync:
        st.subheader("📥 Sync Inbox")
        st.caption("Ingest all un-indexed notices currently sitting in `data/inbox/`.")
        if st.button("🔄 Sync Inbox Now", disabled=not is_healthy):
            with st.spinner("Processing files from inbox …"):
                try:
                    sync_res = sync_inbox_notices()
                    results = sync_res.get("results", [])
                    if not results:
                        st.info("No new files found in `data/inbox/`.")
                    else:
                        st.success(f"Processed {len(results)} file(s):")
                        for r in results:
                            st.markdown(
                                f"- **{r.get('title', 'Unknown')}**: Status `{r.get('status')}` "
                                f"({r.get('chunks', 0)} chunks)"
                            )
                except Exception as e:
                    st.error(f"Failed to sync inbox: {e}")

    st.markdown("---")

    # ── Notice Library Table with Fixed Height ────────────────────────────────
    st.subheader("📚 Indexed Notice Library")
    notices = get_notices() if is_healthy else []

    if not notices:
        st.info("No notices have been indexed yet. Upload a notice above or drop files into `data/inbox/` and click Sync.")
    else:
        title_map = {n["id"]: n["title"] for n in notices}

        table_rows = []
        for n in notices:
            supersedes_id = n.get("supersedes_id")
            superseded_title = title_map.get(supersedes_id) if supersedes_id else None
            supersedes_display = superseded_title if superseded_title else ("—" if not supersedes_id else supersedes_id)

            table_rows.append(
                {
                    "Title": n.get("title"),
                    "Notice Date": n.get("notice_date"),
                    "Category": n.get("category"),
                    "Supersedes": supersedes_display,
                    "Source File": n.get("source_file"),
                    "Indexed Time": n.get("indexed_at"),
                }
            )

        # Fixed height (360px) enables virtualized row scrolling for smooth performance
        st.dataframe(
            table_rows,
            height=360,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Showing {len(notices)} indexed notice(s).")

        # ── Optional Manual Override / Edit Notice Metadata ───────────────────
        with st.expander("✏️ Manual Override / Edit Notice Metadata", expanded=False):
            st.caption("Correct notice title, date, category, or manual supersedes relationship (calls PATCH /notices/{id}).")
            notice_options = {f"{n['title']} ({n['notice_date']}) [ID: {n['id'][:8]}…]": n for n in notices}
            selected_label = st.selectbox("Select notice to edit:", list(notice_options.keys()))
            selected_notice = notice_options[selected_label]

            edit_title = st.text_input("Title", value=selected_notice.get("title", ""))
            edit_date = st.text_input("Notice Date (YYYY-MM-DD)", value=selected_notice.get("notice_date", ""))
            edit_cat = st.text_input("Category", value=selected_notice.get("category", ""))

            other_notices = [n for n in notices if n["id"] != selected_notice["id"]]
            supersedes_choices = ["(None)"] + [f"{o['title']} ({o['notice_date']})" for o in other_notices]
            current_target = title_map.get(selected_notice.get("supersedes_id"))
            current_index = 0
            if current_target:
                for idx, o in enumerate(other_notices, start=1):
                    if o["id"] == selected_notice.get("supersedes_id"):
                        current_index = idx
                        break

            chosen_super = st.selectbox("Supersedes notice:", supersedes_choices, index=current_index)

            if st.button("Save Changes"):
                patch_payload: dict[str, Any] = {
                    "title": edit_title,
                    "notice_date": edit_date,
                    "category": edit_cat,
                }
                if chosen_super == "(None)":
                    patch_payload["supersedes_id"] = None
                else:
                    chosen_idx = supersedes_choices.index(chosen_super) - 1
                    patch_payload["supersedes_id"] = other_notices[chosen_idx]["id"]

                try:
                    update_notice_metadata(selected_notice["id"], patch_payload)
                    st.success("Notice updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update notice: {e}")

        # ── Delete Notice ──────────────────────────────────────────────────────
        with st.expander("🗑️ Delete Notice", expanded=False):
            st.caption("Permanently remove a notice from the index (calls DELETE /notices/{id}). This action cannot be undone.")
            del_options = {f"{n['title']} ({n['notice_date']}) [ID: {n['id'][:8]}…]": n for n in notices}
            del_label = st.selectbox("Select notice to delete:", list(del_options.keys()), key="del_notice_select")
            del_notice = del_options[del_label]
            st.warning(f"You are about to delete **{del_notice['title']}**. This will remove all its chunks from the vector index.")
            if st.button("🗑️ Confirm Delete", type="primary", key="btn_confirm_delete"):
                try:
                    delete_notice_api(del_notice["id"])
                    st.success(f"Deleted notice **{del_notice['title']}** and all its chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete notice: {e}")


# ---------------------------------------------------------------------------
# Page: 📊 Evaluation
# ---------------------------------------------------------------------------

elif page == "📊 Evaluation":
    st.markdown('<div class="main-title">📊 RAG Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Benchmark question-answering accuracy, citations, refusal precision, and latency across retrieval configs.</div>',
        unsafe_allow_html=True,
    )

    if not is_healthy:
        st.error(
            f"⚠️ **Backend service is not running.** Please start FastAPI at `{BACKEND_URL}` to run or view evaluations."
        )

    # ── Evaluation Run Controls ───────────────────────────────────────────────
    st.subheader("⚙️ Run Benchmark Evaluation")
    st.caption("Executes `eval/test_set.json` against the selected pipeline configurations. When a configuration uses a different chunk size, it is indexed into a separate Chroma collection without modifying the main database.")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        # Config selection: checkboxes for multi-config or single radio
        st.markdown("**Select configuration(s):**")
        run_mode_choice = st.radio(
            "Run mode:",
            ["Both configs (A & B)", "Config A only", "Config B only"],
            horizontal=True,
            key="radio_cfg_choice",
        )
        if run_mode_choice == "Both configs (A & B)":
            chosen_configs = ["A", "B"]
        elif run_mode_choice == "Config A only":
            chosen_configs = ["A"]
        else:
            chosen_configs = ["B"]

        retrieval_only_mode = st.checkbox(
            "🔍 Retrieval-only mode (no LLM calls — measures Hit@5 and MRR)",
            value=False,
            key="chk_retrieval_only",
        )
        rerun_failed = st.checkbox(
            "🔄 Re-run failed/error questions only",
            value=False,
            key="chk_rerun_failed",
            disabled=retrieval_only_mode,
        )
        delay_input = st.number_input(
            "Delay between LLM calls (seconds):",
            min_value=0.0,
            max_value=30.0,
            value=7.0,
            step=0.5,
            key="num_delay_sec",
            disabled=retrieval_only_mode,
        )
    with col_ctrl2:
        st.write("")
        st.write("")

        run_disabled = not is_healthy or len(chosen_configs) == 0
        if st.button("🚀 Run Evaluation", type="primary", disabled=run_disabled, key="btn_run_eval"):
            prev_run_id_val = None
            _current_run = st.session_state.get("latest_eval_run")
            if rerun_failed and _current_run:
                prev_run_id_val = _current_run.get("run_id")
            if retrieval_only_mode:
                spinner_msg = f"Running retrieval-only evaluation on Config(s) {chosen_configs} (Hit@5 + MRR)..."
            elif rerun_failed:
                spinner_msg = f"Re-running failed/error questions on Config(s) {chosen_configs} (delay={delay_input}s)..."
            else:
                spinner_msg = f"Running evaluation benchmark on Config(s) {chosen_configs} (delay={delay_input}s)..."
            with st.spinner(spinner_msg):
                try:
                    eval_result = run_evaluation_api(
                        configs=chosen_configs,
                        delay_seconds=float(delay_input),
                        rerun_failed_only=rerun_failed,
                        previous_run_id=prev_run_id_val,
                        retrieval_only=retrieval_only_mode,
                    )
                    st.session_state["latest_eval_run"] = eval_result
                    run_id_str = eval_result.get("run_id", "")
                    if eval_result.get("aborted_reason"):
                        st.warning(f"⚠️ {eval_result['aborted_reason']} (Run ID: `{run_id_str}`)")
                    else:
                        st.success(f"✅ Evaluation run completed! (Run ID: `{run_id_str}`)")
                except Exception as exc:
                    st.error(f"Evaluation run failed: {exc}")

    st.markdown("---")

    # ── Load Runs History ─────────────────────────────────────────────────────
    runs_history = get_evaluation_results_api() if is_healthy else []

    # If a run just executed in this session, prepend or ensure it's selected
    selected_run = None
    if "latest_eval_run" in st.session_state:
        selected_run = st.session_state["latest_eval_run"]

    if runs_history:
        run_options = {
            f"{r.get('run_id')} ({r.get('timestamp', '')[:19]}) — {len(r.get('configs', {}))} config(s)": r
            for r in runs_history
        }
        selected_label = st.selectbox("Select Evaluation Run to inspect:", list(run_options.keys()), index=0)
        selected_run = run_options[selected_label]
    elif selected_run is None:
        st.info("ℹ️ No saved evaluation runs found yet. Click **Run Evaluation** above to run the benchmark suite.")
        st.stop()

    if not selected_run or "configs" not in selected_run or not selected_run["configs"]:
        st.warning("Selected evaluation run contains no configuration results.")
        st.stop()

    # Banner for aborted runs
    if selected_run.get("aborted_reason"):
        st.warning(f"⚠️ **{selected_run['aborted_reason']}**")

    configs_data = selected_run["configs"]
    run_mode = selected_run.get("run_mode", "full")
    is_retrieval_only = run_mode == "retrieval_only"
    top_k_used = selected_run.get("top_k", 5)

    st.markdown(f"### 📈 Results for Run `{selected_run.get('run_id')}`")
    st.caption(f"Executed on: `{selected_run.get('timestamp')}` | Total test cases: {selected_run.get('total_test_cases', 'N/A')} | Mode: `{'Retrieval-only' if is_retrieval_only else 'Full RAG'}`")

    # ── Summary Comparison Table ──────────────────────────────────────────────
    st.subheader("📋 Configuration Comparison Table")

    comparison_rows = []
    if is_retrieval_only:
        metric_labels = [
            (f"Hit@{top_k_used}", "hit_at_k", "%"),
            ("MRR", "mrr", "ratio"),
            ("Answerable Questions", "total_answerable", " count"),
            ("Total Questions", "total_questions", " count"),
        ]
    else:
        metric_labels = [
            ("Answer Accuracy", "answer_accuracy", "%"),
            ("Citation Hit Rate", "citation_hit_rate", "%"),
            ("Refusal Accuracy", "refusal_accuracy", "%"),
            ("Average Latency", "average_latency_ms", " ms"),
            (f"Hit@5", "hit_at_5", "%"),
            ("MRR", "mrr", "ratio"),
            ("Passed Questions", "passed_count", " count"),
            ("Failed Questions", "failed_count", " count"),
            ("API Errors", "api_errors", " count"),
            ("Total Questions", "total_questions", " count"),
        ]

    for label, key, unit in metric_labels:
        row = {"Metric": label}
        for cfg_name, cfg_info in configs_data.items():
            metrics = cfg_info.get("metrics", {})
            val = metrics.get(key, 0)
            col_label = f"Config {cfg_name} ({cfg_info.get('retrieval_mode', '')})"
            if unit == "%":
                row[col_label] = f"{val * 100:.1f}%"
            elif unit == "ratio":
                row[col_label] = f"{val:.4f}"
            elif unit == " ms":
                row[col_label] = f"{val:.1f} ms"
            else:
                row[col_label] = f"{val}"
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    csv_bytes = df_comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Comparison Metrics (CSV)",
        data=csv_bytes,
        file_name=f"{selected_run.get('run_id')}_metrics.csv",
        mime="text/csv",
        key="btn_dl_metrics_csv",
    )

    # ── Bar Charts Comparing Configurations ───────────────────────────────────
    st.subheader("📊 Metric Comparison Charts")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        if is_retrieval_only:
            st.markdown(f"**Retrieval Quality (%)**")
            chart_data: dict[str, list[float]] = {"Metric": [f"Hit@{top_k_used}", "MRR (×100)"]}
            for cfg_name, cfg_info in configs_data.items():
                m = cfg_info.get("metrics", {})
                chart_data[f"Config {cfg_name}"] = [
                    m.get("hit_at_k", 0.0) * 100,
                    m.get("mrr", 0.0) * 100,
                ]
            df_acc_chart = pd.DataFrame(chart_data).set_index("Metric")
            st.bar_chart(df_acc_chart)
        else:
            st.markdown("**Accuracy & Precision Metrics (%)**")
            chart_data = {"Metric": ["Answer Accuracy", "Citation Hit Rate", "Refusal Accuracy", "Hit@5", "MRR (×100)"]}
            for cfg_name, cfg_info in configs_data.items():
                m = cfg_info.get("metrics", {})
                chart_data[f"Config {cfg_name}"] = [
                    m.get("answer_accuracy", 0.0) * 100,
                    m.get("citation_hit_rate", 0.0) * 100,
                    m.get("refusal_accuracy", 0.0) * 100,
                    m.get("hit_at_5", 0.0) * 100,
                    m.get("mrr", 0.0) * 100,
                ]
            df_acc_chart = pd.DataFrame(chart_data).set_index("Metric")
            st.bar_chart(df_acc_chart)

    with col_chart2:
        if not is_retrieval_only:
            st.markdown("**Average Latency (ms)**")
            latency_data = {
                f"Config {cfg_name}": [cfg_info.get("metrics", {}).get("average_latency_ms", 0.0)]
                for cfg_name, cfg_info in configs_data.items()
            }
            df_lat_chart = pd.DataFrame(latency_data, index=["Latency (ms)"])
            st.bar_chart(df_lat_chart)
        else:
            st.markdown("**Hit count vs answerable questions**")
            hit_data: dict[str, list[int]] = {"Metric": ["Hit count", "Answerable total"]}
            for cfg_name, cfg_info in configs_data.items():
                m = cfg_info.get("metrics", {})
                hit_data[f"Config {cfg_name}"] = [
                    m.get("hit_count", 0),
                    m.get("total_answerable", 0),
                ]
            df_hit_chart = pd.DataFrame(hit_data).set_index("Metric")
            st.bar_chart(df_hit_chart)

    st.markdown("---")

    # ── Per-Question Detailed Inspection Table ────────────────────────────────
    st.subheader("🔍 Per-Question Results & Failure Inspection")
    st.caption("Inspect individual test questions to verify answer generation, keyword presence, citation hits, and refusal accuracy.")

    cfg_names = list(configs_data.keys())
    selected_inspect_cfg = st.radio("Select Config to Inspect:", cfg_names, horizontal=True)

    per_q_list = configs_data[selected_inspect_cfg].get("per_question", [])
    if not per_q_list:
        st.info("No per-question data available for this configuration.")
    else:
        # Filter options
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            filter_choice = st.selectbox(
                "Filter Questions:",
                ["All Questions", "❌ Failures Only", "⚠️ API Errors Only", "✅ Passed Only"],
                index=0,
            )
        with col_f2:
            passed_c = sum(1 for q in per_q_list if q.get("passed"))
            error_c = sum(1 for q in per_q_list if q.get("status") == "ERROR")
            failed_c = sum(1 for q in per_q_list if q.get("status") == "FAIL")
            st.metric("Total Questions", f"{len(per_q_list)}", delta=f"{passed_c} passed / {failed_c} failed / {error_c} API errors")

        filtered_questions = []
        for q in per_q_list:
            if filter_choice == "❌ Failures Only" and q.get("status") != "FAIL":
                continue
            if filter_choice == "⚠️ API Errors Only" and q.get("status") != "ERROR":
                continue
            if filter_choice == "✅ Passed Only" and not q.get("passed"):
                continue
            filtered_questions.append(q)

        # Overview Table
        table_records = []
        for q in filtered_questions:
            reasons = "; ".join(q.get("failure_reasons", [])) if q.get("failure_reasons") else "—"
            status_display = "⚠️ ERROR" if q.get("status") == "ERROR" else ("✅ PASS" if q.get("passed") else "❌ FAIL")
            hit_val = q.get("hit")
            rank_val = q.get("first_correct_rank")
            row: dict[str, Any] = {
                "ID": q.get("id"),
                "Type": "Answerable" if q.get("answerable") else "Unanswerable",
                "Question": q.get("question"),
            }
            if is_retrieval_only:
                row["Expected Source"] = q.get("expected_source") or "—"
                row[f"Hit@{top_k_used}"] = "✅" if hit_val is True else ("❌" if hit_val is False else "—")
                row["First Correct Rank"] = str(rank_val) if rank_val is not None else "—"
                row["Reciprocal Rank"] = f"{q.get('reciprocal_rank', 0.0):.4f}" if q.get("reciprocal_rank") is not None else "—"
            else:
                row["Status"] = status_display
                row["Expected Answer"] = q.get("expected_answer")
                row["Actual Answer"] = q.get("actual_answer")
                row["Latency"] = f"{q.get('latency_ms', 0)} ms"
                row[f"Hit@5"] = "✅" if hit_val is True else ("❌" if hit_val is False else "—")
                row["First Correct Rank"] = str(rank_val) if rank_val is not None else "—"
                row["Failure Reason"] = reasons
            table_records.append(row)

        st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)

        # Expandable inspection cards
        with st.expander("🔎 Deep Inspection (Question-by-Question)", expanded=False):
            for q in filtered_questions:
                status_icon = "⚠️" if q.get("status") == "ERROR" else ("✅" if q.get("passed") else "❌")
                st.markdown(f"#### {status_icon} `{q.get('id')}`: {q.get('question')}")
                st.markdown(f"- **Expected Answer:** `{q.get('expected_answer')}`")
                st.markdown(f"- **Actual Output:** {q.get('actual_answer')}")
                if q.get("expected_source"):
                    st.markdown(f"- **Expected Source Notice:** `{q.get('expected_source')}`")
                if q.get("citations"):
                    c_titles = [f"{c.get('title')} (p.{c.get('page')})" for c in q["citations"]]
                    st.markdown(f"- **Returned Citations:** {', '.join(c_titles)}")
                if q.get("failure_reasons"):
                    st.error(f"Failure Reason(s): {', '.join(q['failure_reasons'])}")
                st.markdown("---")
