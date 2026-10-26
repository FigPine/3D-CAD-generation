"""
llm/provider_factory.py
========================
CLI provider-selection prompt and factory function.

This module is the only place that knows about the two concrete provider
implementations.  Everything else (agent, orchestrator, CLI) depends only on
``BaseLLMClient``.

Public API
----------
    select_provider_interactive() -> BaseLLMClient
        Prompt the user to choose a provider at startup, initialize it, run
        the availability check, and return the ready client.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from llm.base_client import BaseLLMClient, LLMUnavailableError, LLMAuthError


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

_MENU = """\
Choose your AI model provider:

  1. Gemini API (online)
  2. Local Qwen3.5-2B via LM Studio (offline)\
"""


def select_provider_interactive() -> BaseLLMClient:
    """
    Display the provider menu, initialize the selected provider, run a
    connectivity check, and return the ready ``BaseLLMClient`` instance.

    This function is called once at application startup.  It exits the
    process with a clear error message if:
    - The user enters an invalid choice.
    - The selected provider fails its availability check.
    - The Gemini API key is missing when Gemini is chosen.

    Returns
    -------
    BaseLLMClient
        A connected, ready-to-use client.
    """
    print(_MENU)
    choice = _prompt_choice()

    if choice == "1":
        return _init_gemini()
    else:
        return _init_lm_studio()


# ---------------------------------------------------------------------------
# Provider initializers
# ---------------------------------------------------------------------------


def _init_gemini() -> "BaseLLMClient":
    """Initialize and verify the Gemini API client."""
    from llm.gemini_client import GeminiClient, LLMAuthError as _LLMAuthError

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "\n❌  GEMINI_API_KEY is not set.\n"
            "    Set it in your environment or create a .env file.\n"
            "    See .env.example for instructions.\n"
            "\n    To use the local model instead, restart and select option 2."
        )
        sys.exit(1)

    print("\n⏳  Initializing Gemini client…")
    try:
        client = GeminiClient()
    except _LLMAuthError as exc:
        print(f"\n❌  Gemini authentication error: {exc}")
        sys.exit(1)
    except ImportError as exc:
        print(
            f"\n❌  Could not load the google-genai SDK: {exc}\n"
            "    Run:  pip install google-genai"
        )
        sys.exit(1)

    # Availability probe — the Gemini SDK validates the key on first real call.
    # We skip the model-list probe here because it is not strictly needed and
    # saves a round-trip; the first extract() call will surface auth errors.
    _print_provider_header(
        label="Gemini API",
        model=client.model,
        note="API key loaded ✓",
    )
    return client


def _init_lm_studio() -> "BaseLLMClient":
    """Initialize and verify the LM Studio client."""
    from llm.lm_studio_client import LMStudioClient

    print("\n⏳  Connecting to LM Studio…")
    client = LMStudioClient()

    try:
        client.check_available()
    except LLMUnavailableError as exc:
        print(f"\n❌  LM Studio is not available:\n\n{exc}")
        print(
            "\n    If you intended to use Gemini instead, restart the "
            "application and select option 1."
        )
        sys.exit(1)

    _print_provider_header(
        label=client.provider_label,
        model=client.model,
        note=f"Endpoint: {client.endpoint}",
    )
    return client


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_provider_header(label: str, model: str, note: str) -> None:
    print(f"\n  Provider : {label}")
    print(f"  Model    : {model}")
    print(f"  {note}")


def _prompt_choice() -> str:
    """Read a provider choice from stdin.  Returns '1' or '2'."""
    while True:
        try:
            raw = input("\nSelect provider (1 or 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)
        if raw in ("1", "2"):
            return raw
        print("  Please enter 1 or 2.")
