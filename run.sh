#!/usr/bin/env bash
# run.sh — start NoticeIQ backend + Streamlit UI on Linux / macOS
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
# Requirements:
#   - Python 3.11+
#   - .venv (optional but recommended): python3 -m venv .venv && source .venv/bin/activate
#   - pip install -r requirements.txt
#   - .env (copied from .env.example with your LLM credentials)

set -euo pipefail

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Detect virtual environment or system binaries ───────────────────────────
if [ -d "$ROOT_DIR/.venv/bin" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
    UVICORN="$ROOT_DIR/.venv/bin/uvicorn"
    STREAMLIT="$ROOT_DIR/.venv/bin/streamlit"
elif [ -d "$ROOT_DIR/venv/bin" ]; then
    PYTHON="$ROOT_DIR/venv/bin/python"
    UVICORN="$ROOT_DIR/venv/bin/uvicorn"
    STREAMLIT="$ROOT_DIR/venv/bin/streamlit"
else
    PYTHON="$(command -v python3 || command -v python)"
    UVICORN="$(command -v uvicorn || true)"
    STREAMLIT="$(command -v streamlit || true)"
    if [ -z "$UVICORN" ]; then
        UVICORN="$PYTHON -m uvicorn"
    fi
    if [ -z "$STREAMLIT" ]; then
        STREAMLIT="$PYTHON -m streamlit"
    fi
fi

# ── Ensure .env exists ───────────────────────────────────────────────────────
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example ..."
    if [ -f "$ROOT_DIR/.env.example" ]; then
        cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
        echo "ℹ️  Created .env. Please edit it and set your LLM_API_KEY before using LLM features."
    else
        echo "❌ .env.example not found. Cannot start." >&2
        exit 1
    fi
fi

# ── Ensure required data directories exist ───────────────────────────────────
mkdir -p "$ROOT_DIR/data/inbox" "$ROOT_DIR/data/processed" "$ROOT_DIR/data/chroma" "$ROOT_DIR/eval/results"

# ── Cleanup child processes on exit / Ctrl+C ─────────────────────────────────
BACKEND_PID=""
UI_PID=""

cleanup() {
    echo ""
    echo "🛑 Shutting down NoticeIQ services..."
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Stopping backend (PID $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$UI_PID" ] && kill -0 "$UI_PID" 2>/dev/null; then
        echo "Stopping Streamlit UI (PID $UI_PID)..."
        kill "$UI_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "Done. Goodbye!"
}

trap cleanup INT TERM EXIT

# ── Start backend ────────────────────────────────────────────────────────────
echo "🚀 Starting NoticeIQ backend on http://${BACKEND_HOST}:${BACKEND_PORT} ..."
$UVICORN app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait for backend health check
echo "Waiting for backend to be healthy..."
HEALTH_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/health"
RETRIES=30
until curl -s "$HEALTH_URL" > /dev/null 2>&1 || [ "$RETRIES" -le 0 ]; do
    sleep 0.5
    RETRIES=$((RETRIES - 1))
done

if [ "$RETRIES" -le 0 ]; then
    echo "⚠️  Backend didn't respond to health check in time; starting UI anyway."
else
    echo "✅ Backend is healthy!"
fi

# ── Start Streamlit UI ───────────────────────────────────────────────────────
echo "🎨 Starting NoticeIQ Streamlit UI on http://localhost:${UI_PORT} ..."
export BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
$STREAMLIT run ui/streamlit_app.py --server.port "$UI_PORT" --server.headless false &
UI_PID=$!

echo ""
echo "=========================================================="
echo " NoticeIQ is running!"
echo " • Web UI:     http://localhost:${UI_PORT}"
echo " • API Docs:   http://${BACKEND_HOST}:${BACKEND_PORT}/docs"
echo " • Health:     http://${BACKEND_HOST}:${BACKEND_PORT}/health"
echo " Press Ctrl+C in this terminal to stop both services."
echo "=========================================================="
echo ""

# Wait on background processes
wait "$BACKEND_PID" "$UI_PID"
