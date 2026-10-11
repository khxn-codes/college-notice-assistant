"""Application settings loaded from environment variables.

All settings have safe defaults so the app starts without a .env file,
but LLM calls will fail if LLM_API_KEY is not set.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object. Values come from environment variables
    or a .env file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider settings
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    llm_api_key: str = ""

    # Embedding model (runs locally via sentence-transformers)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 80

    # Retrieval settings
    top_k: int = 5

    # Retrieval mode: "hybrid" (vector + BM25) or "vector" (vector-only)
    # Config A = vector-only (chunk_size 800), Config B = hybrid (chunk_size 500)
    retrieval_mode: str = "hybrid"   # "hybrid" | "vector"

    # k values for the two retrieval legs (used by retriever.py)
    vector_k: int = 8
    bm25_k: int = 8

    # Data directory (relative to project root)
    data_dir: str = "./data"


# Module-level singleton — import this everywhere
settings = Settings()
