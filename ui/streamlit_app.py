"""Streamlit UI for College Notice Assistant (Milestone 5 - Performance Optimized).

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
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")

st.set_page_config(
    page_title="College Notice Assistant",
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


# ---------------------------------------------------------------------------
# Sidebar Navigation & Backend Status
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 🎓 Notice Assistant")
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
    st.sidebar.error(f"Backend Unreachable\n`{BACKEND_URL}`\nPlease start the API:\n`uvicorn app.main:app --reload`")

st.sidebar.markdown("---")
st.sidebar.caption("College Notice Assistant • Milestone 5")


# ---------------------------------------------------------------------------
# Page: 💬 Chat
# ---------------------------------------------------------------------------

if page == "💬 Chat":
    st.markdown('<div class="main-title">💬 Notice Assistant Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Ask any question about college exams, fees, hostels, or schedules. '
        'Every answer is grounded in official notices with citations.</div>',
        unsafe_allow_html=True,
    )

    if not is_healthy:
        st.error(
            f"⚠️ **Backend service is not running.** Please start the FastAPI backend server:\n\n"
            f"```powershell\nuvicorn app.main:app --reload\n```\n"
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
                err_msg = "Cannot reach backend API. Please make sure `uvicorn app.main:app --reload` is running."
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
                        err_text = f"Backend returned an error ({http_err.response.status_code}): {http_err.response.text}"
                        st.error(err_text)
                        st.session_state.messages.append({"role": "assistant", "content": err_text})
                    except Exception as exc:
                        err_text = f"An unexpected error occurred while querying the notices: {exc}"
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


# ---------------------------------------------------------------------------
# Page: 📊 Evaluation
# ---------------------------------------------------------------------------

elif page == "📊 Evaluation":
    st.markdown('<div class="main-title">📊 RAG Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Benchmark question-answering accuracy, citations, and latency across retrieval configs.</div>',
        unsafe_allow_html=True,
    )

    st.info("🚧 **Coming in Milestone 6**\n\nThe evaluation harness will run the benchmark suite (`eval/test_set.json`) and compare retrieval performance between Vector-only and Hybrid configurations.")

    st.markdown(
        """
        ### Planned Milestone 6 Features:
        - **Automated Test Set Runner**: Evaluates questions against ground truth notices.
        - **Configuration Comparison**:
          - **Config A**: Vector-only search with 800-token chunks.
          - **Config B**: Hybrid search (Vector + BM25 RRF) with 500-token chunks.
        - **Evaluation Metrics**:
          - **Answer Accuracy**: Keyword and semantic precision.
          - **Citation Hit Rate**: Verifies expected source notice is cited.
          - **Refusal Accuracy**: Verifies unanswerable questions are politely refused.
          - **Average Latency**: Response generation time comparison.
        """
    )
