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
import re
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


def classify_error_cause(err: str | Exception) -> str:
    """Classify an API error into: 'rate limit', 'quota', 'model not found', or 'auth'."""
    text = str(err).lower()

    # Check for authentication / API key issues
    if not settings.llm_api_key:
        return "auth"
    if any(k in text for k in [
        "api key not valid",
        "api_key_invalid",
        "invalid api key",
        "unauthenticated",
        "unauthorized",
        "permission_denied",
        "permissiondenied",
        "401",
        "403",
        "forbidden",
        "invalid_argument",
    ]) and not ("429" in text or "quota" in text):
        return "auth"

    # Rate limit vs Quota
    if any(k in text for k in ["rate limit", "rate_limit", "too many requests"]):
        return "rate limit"
    if any(k in text for k in ["quota", "resourceexhausted", "resource exhausted"]):
        return "quota"
    if "429" in text:
        return "rate limit"

    # Model not found
    if any(k in text for k in ["not found", "notfound", "404", "model not found"]):
        return "model not found"

    if any(k in text for k in ["auth", "credential", "api key", "api_key"]):
        return "auth"

    return "unknown API error"


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
        if not settings.llm_api_key:
            cause = "auth"
            logger.error("Gemini API error (%s): LLM_API_KEY is not configured.", cause)
            return f"I could not generate a response due to an API error ({cause})."

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
            err = re.sub(r"AIza[0-9A-Za-z-_]{35}", "***", err)
            cause = classify_error_cause(err)
            logger.error("Gemini API error (%s): %s", cause, err)
            return f"I could not generate a response due to an API error ({cause})."


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
