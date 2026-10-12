# NoticeIQ — Dockerfile
# Multi-purpose slim image for both FastAPI backend and Streamlit UI services

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

WORKDIR /app

# Install minimal OS dependencies for network & compilation if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create persistent cache and data directories
RUN mkdir -p /cache/huggingface /cache/torch /app/data/inbox /app/data/processed /app/data/chroma /app/eval/results

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching wheel files in the image layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code (secrets in .env are excluded by .dockerignore)
COPY app/ ./app/
COPY ui/ ./ui/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY .env.example ./

# Expose standard FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command starts the FastAPI backend; overridden in docker-compose.yml for UI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
