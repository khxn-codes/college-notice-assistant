"""Prompt templates for the RAG system.

System prompt enforces TECH_SPEC §7 rules:
  - Answer only from provided passages.
  - Refuse with the exact phrase when context is insufficient.
  - Cite every claim as [Title, Date, p.N].
  - Never invent dates, amounts, or names.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are NoticeIQ, an AI assistant for answering college notice queries with citations. You answer student questions \
strictly based on the notice passages provided below. Follow these rules without exception:

1. Answer ONLY using information from the provided passages. Do not use any outside knowledge.
2. If the passages do not contain enough information to answer the question, reply with \
EXACTLY this sentence and nothing else:
   I could not find this in the available notices.
3. Every factual claim must be cited using this format: [Title, Date, p.N]
   where N is the page number from the passage header.
4. If notices conflict or when revision notes state that a newer notice supersedes an older one, always use the newest notice for the answer, and explicitly mention that it supersedes the older notice.
5. Never invent or guess dates, amounts, names, or rules.
6. Keep answers short and direct. Do not add preamble or filler.
"""


def build_user_message(
    question: str,
    passages: list[dict],
    supersedes_notes: list[str] | None = None,
) -> str:
    """Build the user turn that contains the numbered passages, optional revision notes, and question.

    Args:
        question:         The student's question.
        passages:         List of dicts with keys: index, title, notice_date, page, text.
        supersedes_notes: Optional list of supersedes notes
                          (e.g. "Notice B (date) supersedes Notice A (date)").

    Returns:
        The full user-turn string to send to the LLM.
    """
    if not passages:
        passage_block = "(No relevant passages were found.)"
    else:
        lines: list[str] = []
        for p in passages:
            header = (
                f"[{p['index']}] {p['title']} | {p['notice_date']} | p.{p['page']}"
            )
            lines.append(header)
            lines.append(p["text"])
            lines.append("")
        passage_block = "\n".join(lines).strip()

    notes_block = ""
    if supersedes_notes:
        note_lines = ["Revision Notes:"]
        for note in supersedes_notes:
            note_lines.append(f"- {note}")
        notes_block = "\n".join(note_lines) + "\n\n"

    return f"Passages:\n{passage_block}\n\n{notes_block}Question: {question}"
