# AGENTS.md

## Project
NoticeIQ: RAG-Based AI Assistant for Answering College Notice Queries with Citations: a RAG chatbot that answers student questions about college notices with citations. Read `PRD.md` and `TECH_SPEC.md` before doing anything. They are the source of truth.

## Stack
Python 3.11+, FastAPI, Streamlit, ChromaDB, sentence-transformers, rank_bm25, SQLite, pytest.

## How to work
- Work on one milestone at a time (see TECH_SPEC.md section 13). Stop after each milestone and wait for review.
- Do not add features, endpoints, or dependencies that are not in the PRD or tech spec. If something seems missing, ask first.
- Keep changes small and focused. Do not rewrite files that are unrelated to the current milestone.
- Run the tests after every change and fix failures before moving on.
- If a requirement is ambiguous, state your assumption briefly and proceed with the simplest option.

## Code rules
- Follow the folder structure in TECH_SPEC.md section 3.
- Keep the ingestion connector interface swappable (`Connector.fetch_new()`), so an ERP connector can replace the folder connector later.
- Never hardcode secrets, API keys, or paths. Read them from environment variables via `config.py`. Never commit `.env`.
- Use type hints and short docstrings for public functions.
- Handle errors gracefully (unreadable PDF, empty text, LLM failure) with clear messages instead of crashing.
- Add logging for ingestion and queries.

## RAG rules (must always hold)
- The assistant answers only from retrieved context.
- If the context does not contain the answer, it replies exactly: "I could not find this in the available notices."
- Every answer cites sources as `[Title, Date, p.N]`.
- When notices conflict, prefer the most recent and mention the one it supersedes.
- Never invent dates, amounts, or names. Use low temperature (0 to 0.2).

## Data and privacy
- Use only public notices. Do not ingest or store personal student data (marks, attendance, individual fees).
- No ERP login, scraping, or credentials anywhere in the code.

## Testing
- Unit tests for chunking, metadata parsing, dedupe, and the latest-wins logic.
- One end-to-end test for `/ask`.
- Evaluation test set lives in `eval/test_set.json`. Do not edit expected answers to make tests pass.

## When finishing a milestone
1. Summarize what was built and which files changed.
2. List how to run and verify it.
3. Note any assumptions or open questions.
4. Stop and wait for the next instruction.
