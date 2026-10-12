.PHONY: all run run-backend run-ui install test eval clean docker-build docker-up docker-down help

PYTHON ?= python3
VENV ?= .venv
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest
UVICORN ?= $(VENV)/bin/uvicorn
STREAMLIT ?= $(VENV)/bin/streamlit

help:
	@echo "NoticeIQ Makefile Commands:"
	@echo "  make install       - Create venv (if needed) and install dependencies"
	@echo "  make run           - Run both backend and UI via run.sh"
	@echo "  make run-backend   - Run FastAPI backend only (port 8000)"
	@echo "  make run-ui        - Run Streamlit UI only (port 8501)"
	@echo "  make test          - Run pytest test suite"
	@echo "  make eval          - Run evaluation script"
	@echo "  make docker-up     - Start services via docker-compose"
	@echo "  make docker-down   - Stop docker-compose services"
	@echo "  make clean         - Clean temporary files and caches"

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	@chmod +x run.sh && ./run.sh

run-backend:
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 --reload

run-ui:
	$(STREAMLIT) run ui/streamlit_app.py --server.port 8501

test:
	$(PYTEST) -v

eval:
	$(PYTHON) eval/runner.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
