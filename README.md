# College Notice Assistant

A RAG chatbot that answers student questions about college notices (exam schedules, fees, hostel, events, rules) with citations. Built with FastAPI, Streamlit, ChromaDB, and rank_bm25.

---

## Quickstart: Running Backend & UI Together

To run the application locally, run the FastAPI backend and Streamlit UI in two separate terminal windows.

### 1. Start the FastAPI Backend

In Terminal 1 (from the project root with the virtual environment activated):

```powershell
# Activate virtual environment (if not already active)
.venv\Scripts\Activate.ps1

# Start the FastAPI server on port 8000
uvicorn app.main:app --reload
```

The backend API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2. Start the Streamlit UI

In Terminal 2 (with the virtual environment activated):

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run the Streamlit web application
streamlit run ui/streamlit_app.py
```

The UI will automatically open in your default browser at [http://localhost:8501](http://localhost:8501).

> **Note on Backend URL:**  
> By default, the Streamlit UI connects to `http://127.0.0.1:8000`. To point to a different host or port, set the `BACKEND_URL` environment variable:
> ```powershell
> $env:BACKEND_URL="http://127.0.0.1:8000"
> streamlit run ui/streamlit_app.py
> ```

---

## Running Tests

Run the test suite using pytest:

```powershell
.venv\Scripts\pytest
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your LLM API credentials:

```bash
cp .env.example .env
```
