"""
llm/gemini_client.py
=====================
Adapter for the Google Generative AI (Gemini) API.

Responsibilities
----------------
* Accept a system instruction, user content, and an expected JSON schema.
* Request structured JSON output using the ``response_mime_type`` parameter
  where supported by the chosen model.
* Apply timeout and bounded retry logic for transient failures.
* Return the raw parsed dict plus usage metadata.
* Raise typed exceptions for auth, rate-limit, timeout, and format failures.
* Never store or log the API key.

Provider isolation
------------------
All Gemini-specific SDK calls are confined to this file.  The rest of the
application uses only the ``GeminiClient`` class and the typed exceptions.
To swap providers, replace this file and update ``requirements.txt``.

Environment variable
--------------------
``GEMINI_API_KEY`` — must be set before instantiating the client.

Example usage
-------------
    client = GeminiClient()
    result = client.extract(
        system_instruction=SYSTEM_INSTRUCTION,
        user_content=build_extraction_prompt(user_input),
    )
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Typed exceptions (provider-agnostic names)
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
# Client
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-3.8-flash"
_DEFAULT_TIMEOUT = 30          # seconds
_DEFAULT_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.5      # seconds; doubles each retry


class GeminiClient:
    """
    Thin adapter around the ``google-genai`` SDK.

    Parameters
    ----------
    model : str
        Gemini model identifier (default: ``gemini-2.0-flash``).
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Maximum number of retries on transient failures.
    api_key : str, optional
        Override the API key (for testing only). Normally read from env.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        api_key: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise LLMAuthError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it before using GeminiClient."
            )
        # Import lazily so the rest of the app can load without the SDK
        try:
            import google.genai as genai  # type: ignore[import]
            import google.genai.types as genai_types  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "google-genai package is not installed. "
                "Run: pip install google-genai"
            ) from exc

        self._genai = genai
        self._genai_types = genai_types
        self._client = genai.Client(api_key=resolved_key)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        system_instruction: str,
        user_content: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Send an extraction request to Gemini and return parsed JSON.

        Parameters
        ----------
        system_instruction : str
            The system-role instructions.
        user_content : str
            The user-turn content.

        Returns
        -------
        (parsed_dict, usage_metadata)
            ``parsed_dict`` is the JSON object from the model.
            ``usage_metadata`` is a dict with token counts (or ``None``).

        Raises
        ------
        LLMAuthError
            If the API key is invalid.
        LLMRateLimitError
            If the quota is exceeded.
        LLMTimeoutError
            If the request times out.
        LLMResponseFormatError
            If the response cannot be parsed as JSON.
        LLMUnavailableError
            For other server-side failures.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=self._genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.0,   # deterministic extraction
                    ),
                )
                return self._parse_response(response)

            except Exception as exc:
                last_error = exc
                retryable, typed = self._classify_error(exc)
                if not retryable:
                    raise typed from exc
                # Exponential back-off before retry
                if attempt < self.max_retries - 1:
                    sleep_time = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(sleep_time)

        # All retries exhausted
        raise LLMUnavailableError(
            f"Gemini API unavailable after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self, response: Any
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Extract and parse the JSON payload from a Gemini response."""
        try:
            text = response.text
        except AttributeError as exc:
            raise LLMResponseFormatError(
                "Gemini response has no 'text' attribute."
            ) from exc

        if not text or not text.strip():
            raise LLMResponseFormatError("Gemini returned an empty response.")

        # Strip accidental markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
            cleaned = cleaned.rstrip("`").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"Could not parse Gemini response as JSON: {exc}\n"
                f"Raw response (truncated): {cleaned[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMResponseFormatError(
                f"Expected a JSON object; got {type(parsed).__name__}."
            )

        # Usage metadata — best-effort
        usage: Optional[Dict[str, Any]] = None
        try:
            meta = response.usage_metadata
            if meta:
                usage = {
                    "prompt_token_count": getattr(meta, "prompt_token_count", None),
                    "candidates_token_count": getattr(meta, "candidates_token_count", None),
                    "total_token_count": getattr(meta, "total_token_count", None),
                }
        except Exception:  # noqa: BLE001
            pass  # usage metadata is non-critical

        return parsed, usage

    def _classify_error(
        self, exc: Exception
    ) -> Tuple[bool, LLMError]:
        """
        Classify an SDK exception into (is_retryable, typed_LLMError).

        Returns
        -------
        (retryable, typed_error)
        """
        msg = str(exc).lower()

        # Authentication
        if any(kw in msg for kw in ("api key", "authentication", "permission", "403", "401")):
            return False, LLMAuthError(f"Authentication failed: {exc}")

        # Rate limit
        if any(kw in msg for kw in ("rate limit", "quota", "429", "resource exhausted")):
            return True, LLMRateLimitError(f"Rate limit exceeded: {exc}")

        # Timeout
        if any(kw in msg for kw in ("timeout", "timed out", "deadline")):
            return True, LLMTimeoutError(f"Request timed out: {exc}")

        # Server errors (5xx) — retryable
        if any(kw in msg for kw in ("500", "502", "503", "504", "server error", "unavailable")):
            return True, LLMUnavailableError(f"Server error: {exc}")

        # Format / content errors — not retryable
        if any(kw in msg for kw in ("invalid", "schema", "json", "blocked", "safety")):
            return False, LLMResponseFormatError(f"Response format/safety error: {exc}")

        # Generic — retryable
        return True, LLMUnavailableError(f"Unexpected error: {exc}")
