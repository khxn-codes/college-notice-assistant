"""Provider-agnostic LLM client.

Currently implements a Gemini backend via ``google-generativeai``.
Add new providers by subclassing :class:`BaseLLMClient` and registering
in :func:`get_llm_client`.

Configuration (from environment via config.py):
  LLM_PROVIDER  – "gemini" (default)
  LLM_MODEL     – model name, e.g. "gemini-2.0-flash"
  LLM_API_KEY   – secret key; never logged or hardcoded

Temperature is fixed at 0.1 (low, per TECH_SPEC §7).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.config import settings
from app.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_TEMPERATURE = 0.1   # TECH_SPEC §7: 0 to 0.2


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Minimal interface every LLM provider must implement."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a system + user message and return the model's text reply.

        Implementations must:
          - Never log the API key.
          - Return a clear human-readable error string on failure (do not raise).
        """
        ...


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


class GeminiClient(BaseLLMClient):
    """Google Gemini via the ``google-generativeai`` SDK."""

    def __init__(self) -> None:
        if not settings.llm_api_key:
            logger.warning(
                "LLM_API_KEY is not set. Gemini calls will fail. "
                "Set it in your .env file."
            )
        self._model_name = settings.llm_model

    def complete(self, system: str, user: str) -> str:
        """Call the Gemini Chat API and return the text response."""
        try:
            import google.generativeai as genai

            # Configure with the key (never logged)
            genai.configure(api_key=settings.llm_api_key)

            model = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
                generation_config=genai.GenerationConfig(
                    temperature=_TEMPERATURE,
                    max_output_tokens=1024,
                ),
            )
            response = model.generate_content(user)
            return response.text

        except Exception as exc:  # noqa: BLE001
            # Strip any accidental key echo from the exception message
            err = str(exc)
            if settings.llm_api_key and settings.llm_api_key in err:
                err = err.replace(settings.llm_api_key, "***")
            logger.error("Gemini API error: %s", err)
            return (
                "I could not generate a response due to an API error. "
                "Please check your LLM_API_KEY and try again."
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_client() -> BaseLLMClient:
    """Return the configured LLM client based on ``LLM_PROVIDER``.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return GeminiClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Supported: 'gemini'."
    )
