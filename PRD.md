# PRD: NoticeIQ (RAG-Based AI Assistant for Answering College Notice Queries with Citations)

## 1. Overview
NoticeIQ is a chatbot that answers student questions about college notices (exam schedules, fees, hostel, events, rules) using Retrieval-Augmented Generation (RAG). Every answer cites the source notice (title, date, page). If the notices do not contain the answer, the assistant says so instead of guessing.

## 2. Problem
Students waste time searching the college ERP for notices, and revised notices are easy to miss. There is no quick way to ask a question and get a verified, cited answer.

## 3. Target users
- Primary: college students asking questions about notices and rules.
- Secondary (demo): the project evaluator, who will try the app live.

## 4. Goals
- Answer notice-related questions accurately, with citations.
- Prefer the newest notice when notices conflict, and say which notice it supersedes.
- Refuse gracefully when the answer is not in the documents.
- Measure quality with a small evaluation suite and show the results.

## 5. Non-goals (out of scope)
- No automatic ERP login, scraping, or sync (the college does not provide API/DB access).
- No personal student data (marks, attendance, fees of individuals).
- No user accounts or authentication.
- No fine-tuning of models.
- No mobile app.

## 6. Data source
Notices are exported manually from the ERP (PDF or text) and placed in `data/inbox/` or uploaded through the UI. Only public notices are used. The ingestion layer is a swappable connector so that automatic ERP sync can be added later as future work.

## 7. Features

### Must-have
1. Upload and index notices (PDF, TXT).
2. Ask a question and receive an answer with citations (notice title, date, page).
3. "I don't know" response when retrieved context does not support an answer.
4. Source viewer: show the retrieved chunks used for the answer.
5. Notice library: table of indexed notices with title, date, category, and last-updated time.

### Should-have (stand-out)
6. "Latest wins" logic: when notices conflict, prefer the most recent and mention the superseded one.
7. Category filter (exam, fees, hostel, events, general).
8. Evaluation dashboard: run a test set against two retrieval configurations and display accuracy and citation hit rate.
9. Duplicate handling: re-uploading a notice does not create duplicates; a revised notice updates or supersedes the old one.

### Nice-to-have (only if time remains)
10. Voice input.
11. Hindi and English answers.

## 8. User stories
- As a student, I want to ask "What is the revised date of the mid-sem exam?" and get the latest date with its source.
- As a student, I want to see which notice an answer came from so I can verify it.
- As a student, I want the assistant to tell me when it does not know, instead of guessing.
- As an admin or maintainer, I want to upload a new notice and have it searchable immediately.
- As an evaluator, I want to see how well the system performs on a test set.

## 9. Success criteria
- At least 80% of answerable test questions answered correctly with the correct source notice cited.
- At least 90% of unanswerable test questions correctly refused.
- Revised-notice test cases return the newest information.
- App runs with a single documented command (Docker or `make run`).
- Response time under about 10 seconds per question on a laptop.

## 10. Constraints
- Solo project, 7-10 days, built with an AI coding agent.
- Must run locally and demo without depending on the college network.
- LLM provider is configurable through environment variables; no keys in the repo.

## 11. Risks
- Scanned or table-heavy PDFs extract poorly: start with text-based PDFs.
- LLM hallucination: mitigated by strict prompt rules and the "I don't know" behavior.
- Demo failure from network/API issues: keep a recorded backup demo and cached data.

## 12. Future work
- Automatic ERP sync if access is granted.
- OCR for scanned notices.
- Multilingual answers and voice.
- Notification when new notices are indexed.
