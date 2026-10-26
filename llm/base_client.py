"""
llm/base_client.py
==================
Shared typed exceptions and abstract base class for all LLM provider adapters.

All provider-specific clients (GeminiClient, LMStudioClient, …) inherit from
``BaseLLMClient`` and implement ``extract()``.  The agent orchestrator depends
only on this interface, not on any concrete client.

Typed exceptions are defined here so they can be raised and caught uniformly
regardless of which provider is active.  ``llm/gemini_client.py`` re-exports
them for backward compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Typed exceptions  (provider-agnostic)
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all LLM adapter errors."""


class LLMAuthError(LLMError):
    """API key missing, invalid, or lacking permissions."""


class LLMRateLimitError(LLMError):
    """Rate limit or quota exceeded; the caller may retry later."""


class LLMTimeoutError(LLMError):
    """Request timed out."""


class LLMResponseFormatError(LLMError):
    """Response could not be parsed as valid JSON / schema."""


class LLMUnavailableError(LLMError):
    """Service unavailable or generic server error."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """
    Shared interface that every LLM provider adapter must implement.

    The agent orchestrator calls only ``extract()``; it never touches
    provider-specific SDK types.
    """

    #: Human-readable label for the provider (shown to the user at startup).
    provider_label: str = "Unknown provider"

    #: Model identifier resolved after connection/availability check.
    model: str = "unknown"

    @abstractmethod
    def extract(
        self,
        system_instruction: str,
        user_content: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Send a structured extraction request and return parsed JSON.

        Parameters
        ----------
        system_instruction : str
            The system-role prompt.
        user_content : str
            The user-turn prompt.

        Returns
        -------
        (parsed_dict, usage_metadata)
            ``parsed_dict``      — JSON object from the model (must be a dict).
            ``usage_metadata``   — Token counts etc., or ``None``.

        Raises
        ------
        LLMAuthError
        LLMRateLimitError
        LLMTimeoutError
        LLMResponseFormatError
        LLMUnavailableError
        """

    def check_available(self) -> None:
        """
        Verify that the provider is reachable and a model is loaded/ready.

        Raises ``LLMUnavailableError`` or a more specific subclass on failure.
        The default implementation is a no-op; providers that support a
        connectivity probe should override this method.
        """
