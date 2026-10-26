"""
llm/lm_studio_client.py
========================
Adapter for LM Studio's OpenAI-compatible local inference server.

LM Studio exposes an HTTP API that mirrors the OpenAI Chat Completions
endpoint.  This adapter uses ``urllib`` from the standard library so that no
extra SDK dependency is required.  The only required runtime dependency is
``pydantic`` (already in requirements.txt).

Default configuration
---------------------
``LM_STUDIO_BASE_URL``  (env)  — default: ``http://localhost:1234/v1``
``LM_STUDIO_MODEL``     (env)  — default: detected from /v1/models
``LM_STUDIO_TIMEOUT``   (env)  — default: 60 seconds

Structured output
-----------------
LM Studio (current builds) accepts only ``response_format.type`` values of
``"json_schema"`` or ``"text"``.  The earlier ``"json_object"`` type is not
supported and returns HTTP 400.

This client uses a two-stage strategy:

  Stage 1 — ``json_schema`` (preferred)
    Sends ``response_format: {type: "json_schema", json_schema: {...}}`` with
    a hand-written JSON Schema derived from ``CandidateRequirements``.
    ``strict`` is set to ``false`` because strict mode requires every property
    to be listed in ``required`` and forbids ``additionalProperties``, which
    conflicts with our deeply-optional schema and some GGUF backends.

  Stage 2 — ``text`` (fallback)
    If the server returns HTTP 400 on the ``json_schema`` request (meaning the
    loaded model's backend does not support grammar-constrained decoding), the
    client retries once with ``response_format: {type: "text"}`` and relies on
    the existing prompt instructions plus the markdown-fence-stripping parser.

The fallback switch happens at most once per ``extract()`` call and is not
retried in a loop.  Retries of transient errors (timeout, 5xx) use whichever
format last succeeded.

Error handling
--------------
Each public method raises only the typed exceptions defined in
``llm.base_client``, matching the contract used by GeminiClient.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from llm.base_client import (
    BaseLLMClient,
    LLMResponseFormatError,
    LLMTimeoutError,
    LLMUnavailableError,
)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_DEFAULT_TIMEOUT = 60        # seconds — local models can be slow on first token
_DEFAULT_MAX_RETRIES = 2     # retries for transient errors (timeout / 5xx)
_RETRY_BACKOFF_BASE = 2.0

# ---------------------------------------------------------------------------
# JSON Schema for CandidateRequirements
#
# This is a hand-maintained schema that mirrors schemas.CandidateRequirements.
# It is intentionally permissive (no "required" constraints, no
# "additionalProperties": false) so that:
#   a) strict=false works without issues on llama.cpp backends, and
#   b) partially-populated responses (the normal case) are accepted by the
#      grammar engine without forcing the model to fabricate values.
#
# The Pydantic validation in _parse_candidate() is the real enforcement point.
# ---------------------------------------------------------------------------

_CANDIDATE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "building_type": {
            "anyOf": [
                {"type": "string", "enum": ["residential", "commercial", "mixed", "other"]},
                {"type": "null"},
            ]
        },
        "floor_count": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
        },
        "rooms": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "room_id":      {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "room_type":    {"type": "string"},
                            "count":        {"type": "integer", "minimum": 1},
                            "window_count": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                            "door_count":   {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                            "width_m":      {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "length_m":     {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "notes":        {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                    },
                },
                {"type": "null"},
            ]
        },
        "overall_dimensions": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "width_m":          {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "length_m":         {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "floor_height_m":   {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "wall_thickness_m": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    },
                },
                {"type": "null"},
            ]
        },
        "relationships": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "room_a":       {"type": "string"},
                            "room_b":       {"type": "string"},
                            "relationship": {
                                "type": "string",
                                "enum": [
                                    "adjacent_to",
                                    "connects_to",
                                    "shares_wall_with",
                                    "shared_facility",
                                    "opposite_to",
                                ],
                            },
                            "negative":     {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                            "notes":        {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                    },
                },
                {"type": "null"},
            ]
        },
        "units":        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "constraints":  {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "preferences":  {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "ambiguities":  {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
    },
}

_RESPONSE_FORMAT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidate_requirements",
        "strict": False,
        "schema": _CANDIDATE_JSON_SCHEMA,
    },
}

_RESPONSE_FORMAT_TEXT: Dict[str, Any] = {"type": "text"}


class LMStudioClient(BaseLLMClient):
    """
    Adapter for LM Studio's OpenAI-compatible REST API.

    Parameters
    ----------
    base_url : str
        Base URL for the LM Studio server (without trailing slash).
        Reads from ``LM_STUDIO_BASE_URL`` env var if not supplied.
    model : str, optional
        Model identifier to use for inference.  If not supplied, the client
        queries ``/v1/models`` and uses the first available model.
        Reads from ``LM_STUDIO_MODEL`` env var if not supplied.
    timeout : int
        Per-request timeout in seconds.
        Reads from ``LM_STUDIO_TIMEOUT`` env var if not supplied.
    max_retries : int
        Maximum retries on transient errors (timeout / 5xx).
        Format-level fallback (json_schema → text) is separate and does not
        consume retry budget.
    """

    provider_label: str = "Local inference"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("LM_STUDIO_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")

        self.timeout = int(
            timeout
            or os.environ.get("LM_STUDIO_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )
        self.max_retries = max_retries

        # model may be resolved later in check_available()
        _env_model = os.environ.get("LM_STUDIO_MODEL", "")
        self.model: str = model or _env_model or ""

        # Tracks which response_format worked for this server/model combination.
        # Starts as json_schema; falls back to text on first HTTP 400.
        # Once set to text, all subsequent calls skip the json_schema attempt.
        self._response_format: Dict[str, Any] = _RESPONSE_FORMAT_JSON_SCHEMA

        # Endpoint URLs
        self._chat_url = f"{self._base_url}/chat/completions"
        self._models_url = f"{self._base_url}/models"

    # ------------------------------------------------------------------
    # BaseLLMClient interface
    # ------------------------------------------------------------------

    def check_available(self) -> None:
        """
        Query ``/v1/models`` to confirm the server is running and a model is
        loaded.  If ``self.model`` is not set, auto-selects the first loaded
        model.

        Raises
        ------
        LLMUnavailableError
            If the server is not reachable, no model is loaded, or the
            configured model identifier is not found.
        """
        try:
            raw = self._http_get(self._models_url)
        except LLMUnavailableError:
            raise LLMUnavailableError(
                "LM Studio server is not reachable at "
                f"{self._base_url}.\n"
                "  Please:\n"
                "    1. Open LM Studio.\n"
                "    2. Load a model (e.g. Qwen3.5-2B).\n"
                "    3. Go to the 'Local Server' tab and click 'Start Server'.\n"
                "    4. Verify the server is running at "
                f"{self._base_url.split('/v1')[0]}.\n"
                "  Then restart this application."
            )
        except LLMTimeoutError:
            raise LLMUnavailableError(
                f"Connection to LM Studio timed out ({self.timeout}s). "
                "The server may be starting up — please wait and try again."
            )

        # Parse the models list
        available_models: List[str] = []
        try:
            data = json.loads(raw)
            available_models = [
                m.get("id", "") for m in data.get("data", []) if m.get("id")
            ]
        except (json.JSONDecodeError, AttributeError):
            raise LLMUnavailableError(
                "LM Studio returned an unexpected response from /v1/models. "
                "Please ensure LM Studio is fully started."
            )

        if not available_models:
            raise LLMUnavailableError(
                "LM Studio is running but no model is currently loaded.\n"
                "  Please:\n"
                "    1. Open LM Studio.\n"
                "    2. Download a compatible model (e.g. Qwen3.5-2B).\n"
                "    3. Load the model.\n"
                "    4. Confirm the server is still running.\n"
                "  Then restart this application."
            )

        if self.model:
            # Check the user-configured model is actually available
            if self.model not in available_models:
                raise LLMUnavailableError(
                    f"Configured model '{self.model}' is not loaded in LM Studio.\n"
                    f"  Available models: {', '.join(available_models)}\n"
                    "  Either load the correct model in LM Studio, or update\n"
                    "  LM_STUDIO_MODEL in your .env to match an available model."
                )
        else:
            # Auto-select the first available model
            self.model = available_models[0]

    def extract(
        self,
        system_instruction: str,
        user_content: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Send an extraction request to LM Studio and return parsed JSON.

        Tries ``response_format: json_schema`` first.  If the server returns
        HTTP 400 (meaning the loaded model's backend does not support
        grammar-constrained decoding), falls back to ``response_format: text``
        and parses the model's free-text JSON output.  The fallback decision is
        cached on the client instance so subsequent calls skip the failing
        format without consuming retry budget.

        Parameters
        ----------
        system_instruction : str
        user_content : str

        Returns
        -------
        (parsed_dict, usage_metadata)

        Raises
        ------
        LLMUnavailableError, LLMTimeoutError, LLMResponseFormatError
        """
        if not self.model:
            raise LLMUnavailableError(
                "No model has been selected. Call check_available() first."
            )

        # --- Stage 1: try the currently-preferred format ---
        try:
            return self._attempt_extract(system_instruction, user_content, self._response_format)
        except _HTTP400Error as exc:
            # The server rejected the format — not a transient error.
            # Do NOT retry with the same payload.
            if self._response_format is _RESPONSE_FORMAT_JSON_SCHEMA:
                # Stage 2: fall back to plain text and cache the decision.
                self._response_format = _RESPONSE_FORMAT_TEXT
                return self._attempt_extract(
                    system_instruction, user_content, _RESPONSE_FORMAT_TEXT
                )
            # Already on text mode and still getting 400 — surface the error.
            raise LLMUnavailableError(
                f"LM Studio returned HTTP 400 even with response_format=text: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """Return the base URL (without /v1 suffix) for display purposes."""
        return self._base_url.replace("/v1", "") if "/v1" in self._base_url else self._base_url

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attempt_extract(
        self,
        system_instruction: str,
        user_content: str,
        response_format: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Build the payload with the given ``response_format`` and attempt up to
        ``self.max_retries`` times for transient failures (timeout / 5xx).

        Raises ``_HTTP400Error`` on HTTP 400 so the caller can switch format.
        Raises ``LLMResponseFormatError`` for content-level parse failures
        (not retryable).
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "response_format": response_format,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw = self._http_post(self._chat_url, payload)
                return self._parse_chat_response(raw)
            except _HTTP400Error:
                raise  # propagate immediately; not retryable
            except LLMResponseFormatError:
                raise  # content parse failure; not retryable
            except (LLMUnavailableError, LLMTimeoutError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))

        raise LLMUnavailableError(
            f"LM Studio request failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    def _http_get(self, url: str) -> str:
        """Perform a GET request and return the response body as a string."""
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
            if "timed out" in reason.lower() or isinstance(exc.reason, socket.timeout):
                raise LLMTimeoutError(f"GET {url} timed out: {exc}") from exc
            raise LLMUnavailableError(
                f"Cannot reach LM Studio at {url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError(f"GET {url} timed out: {exc}") from exc

    def _http_post(self, url: str, payload: Dict[str, Any]) -> str:
        """Perform a JSON POST request and return the response body."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 400:
                raise _HTTP400Error(
                    f"HTTP 400: {exc.reason}. Body: {body_text}"
                ) from exc
            raise LLMUnavailableError(
                f"LM Studio returned HTTP {exc.code}: {exc.reason}. "
                f"Response body: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
            if "timed out" in reason.lower() or isinstance(exc.reason, socket.timeout):
                raise LLMTimeoutError(
                    f"LM Studio request timed out after {self.timeout}s. "
                    "Consider increasing LM_STUDIO_TIMEOUT."
                ) from exc
            raise LLMUnavailableError(
                f"LM Studio is not reachable at {url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"LM Studio request timed out after {self.timeout}s."
            ) from exc

    def _parse_chat_response(
        self, raw: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Parse an OpenAI-compatible chat completions response."""
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"LM Studio returned non-JSON response: {exc}\n"
                f"Raw (truncated): {raw[:500]}"
            ) from exc

        # Extract content from first choice
        try:
            content = outer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseFormatError(
                f"Unexpected LM Studio response structure: {exc}\n"
                f"Response (truncated): {raw[:500]}"
            ) from exc

        if not content or not content.strip():
            raise LLMResponseFormatError(
                "LM Studio returned an empty message content."
            )

        # Strip <think>...</think> blocks that Qwen3 thinking variants emit
        # before outputting the JSON.
        cleaned = _strip_think_tags(content.strip())

        # Strip markdown fences if the model added them despite the prompt
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (``` or ```json) and trailing ```
            lines = lines[1:]
            cleaned = "\n".join(lines).rstrip("`").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"Could not parse LM Studio message content as JSON: {exc}\n"
                f"Content (truncated): {cleaned[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMResponseFormatError(
                f"Expected a JSON object from LM Studio; got {type(parsed).__name__}."
            )

        # Usage metadata — best-effort
        usage: Optional[Dict[str, Any]] = None
        try:
            u = outer.get("usage", {})
            if u:
                usage = {
                    "prompt_token_count": u.get("prompt_tokens"),
                    "candidates_token_count": u.get("completion_tokens"),
                    "total_token_count": u.get("total_tokens"),
                }
        except Exception:  # noqa: BLE001
            pass

        return parsed, usage


# ---------------------------------------------------------------------------
# Internal sentinel exception — not part of the public LLM error hierarchy
# ---------------------------------------------------------------------------


class _HTTP400Error(Exception):
    """
    Raised internally when the server returns HTTP 400.

    This is a sentinel used by ``_http_post`` → ``_attempt_extract`` →
    ``extract`` to trigger the json_schema → text fallback without consuming
    the transient-error retry budget.  It is never raised to the caller of
    ``extract()``.
    """


# ---------------------------------------------------------------------------
# Utility: strip Qwen3 <think> blocks
# ---------------------------------------------------------------------------


def _strip_think_tags(text: str) -> str:
    """
    Remove ``<think>...</think>`` reasoning blocks that Qwen3 thinking-mode
    models emit before the actual JSON output.

    The match is case-insensitive and handles multi-line blocks.  If no tags
    are found the original string is returned unchanged.
    """
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()
