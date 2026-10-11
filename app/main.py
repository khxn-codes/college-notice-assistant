"""FastAPI application entry point.

Routes defined here (Milestone 1):
  GET /health  – liveness check

Further routes will be added in later milestones:
  /notices/*, /ask, /eval/*
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks before the app accepts requests."""
    logger.info("Starting College Notice Assistant API …")
    init_db()
    yield
    # teardown actions (if any) go here


app = FastAPI(
    title="College Notice Assistant",
    description="RAG chatbot for college notices — FastAPI backend.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns ``{"status": "ok"}`` when the service is running.
    """
    return {"status": "ok"}
