"""
app.py
======
Command-line interface for the Requirements Extraction Agent.

Usage
-----
    python app.py

The CLI starts an interactive session:
  1. Provider selection (Gemini API or LM Studio local model).
  2. Accepts the initial building description.
  3. Displays clarification questions (up to 3 rounds).
  4. Shows the final requirements summary for confirmation.
  5. On confirmation, saves the validated JSON and exits.

Environment
-----------
For Gemini:
    Set ``GEMINI_API_KEY`` before running:
    Windows PowerShell: $env:GEMINI_API_KEY = "your-key"
    Linux/macOS:        export GEMINI_API_KEY="your-key"
    Or copy .env.example to .env and fill in the key.

For LM Studio:
    Optionally set ``LM_STUDIO_BASE_URL`` (default: http://localhost:1234/v1)
    and ``LM_STUDIO_TIMEOUT`` (default: 60).
    No API key required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Optional: load .env file if python-dotenv is installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so relative imports work when the
# script is run from any directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from schemas import SessionStatus
from agents.req_ext_agent import (
    set_llm_client,
    start_session,
    process_user_reply,
    get_session_state,
    build_ir_from_file,
)
from IR_builder import IRBuilderError
from llm.base_client import LLMAuthError


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    print("\n" + "=" * 60)
    print("  AI-Assisted 3D Building Model Generator")
    print("  Requirements Extraction Agent — v1.0")
    print("=" * 60)


def _user_input(prompt: str) -> str:
    """Read a line of input, stripping whitespace."""
    try:
        line = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSession interrupted. Goodbye.")
        sys.exit(0)
    if line.lower() in ("quit", "exit"):
        print("Goodbye.")
        sys.exit(0)
    return line


def _display_turn(turn) -> None:
    """Pretty-print an AgentTurn."""
    print("\n" + "-" * 60)
    print(turn.message)
    if turn.warnings:
        for w in turn.warnings:
            print(f"\n  ⚠ {w}")
    if turn.errors:
        for e in turn.errors:
            print(f"\n  ❌ {e}")
    print()


def run_build_ir_cli(requirements_file: str, output_file: Optional[str] = None) -> None:
    """Manually run Building IR construction on an existing requirements JSON file."""
    print("\n" + "=" * 60)
    print("  AI-Assisted 3D Building Model Generator")
    print("  Building IR v1 Manual Generator")
    print("=" * 60)
    print(f"\n📂  Loading requirements from: {requirements_file}")

    try:
        ir_doc, saved_path = build_ir_from_file(requirements_file, output_file)
    except FileNotFoundError as exc:
        print(f"\n❌  File error: {exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"\n❌  JSON parse error in '{requirements_file}': {exc}")
        sys.exit(1)
    except IRBuilderError as exc:
        print(f"\n❌  Building IR validation error: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌  Unexpected error: {exc}")
        sys.exit(1)

    print("\n✅  Building IR v1 constructed and saved successfully.")
    print(f"    IR Version  : {ir_doc.get('ir_version')}")
    print(f"    Session ID  : {ir_doc.get('session_id')}")
    print(f"    Building ID : {ir_doc.get('building', {}).get('id')}")
    print(f"    Rooms       : {len(ir_doc.get('requirements', {}).get('rooms', []))}")
    print(f"    Edges       : {len(ir_doc.get('requirements', {}).get('edges', []))}")
    print(f"    Constraints : {len(ir_doc.get('requirements', {}).get('constraints', []))}")
    print(f"    Saved to    : {saved_path}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Assisted 3D Building Model Generator",
    )
    parser.add_argument(
        "--build-ir",
        "-b",
        metavar="REQUIREMENTS_FILE",
        help="Path to an existing requirements JSON file to build Building IR without LLM calls.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT_FILE",
        help="Optional destination path for the generated Building IR JSON file.",
    )

    args = parser.parse_args()

    if args.build_ir:
        run_build_ir_cli(args.build_ir, args.output)
        return

    _print_banner()

    # --- Provider selection ---
    # This must happen before any LLM call.  The factory handles its own
    # availability check and exits with a clear error message on failure.
    from llm.provider_factory import select_provider_interactive
    client = select_provider_interactive()

    # Register the chosen client with the agent orchestrator
    set_llm_client(client)

    print("\n" + "-" * 60)
    print("Describe your building and I will extract the requirements.")
    print("Type 'quit' or 'exit' at any prompt to stop.")
    print("-" * 60)

    # --- Initial building description prompt ---
    initial_prompt = _user_input("\nDescribe your building: ")
    if not initial_prompt:
        print("No input provided. Goodbye.")
        sys.exit(0)

    print(f"\n⏳  Extracting requirements ({client.provider_label})…")
    try:
        turn = start_session(initial_prompt)
    except LLMAuthError as exc:
        print(f"\n❌  Authentication error: {exc}")
        sys.exit(1)

    session_id = turn.session_id
    _display_turn(turn)

    # --- Clarification / confirmation loop ---
    while not turn.is_finalized:
        if turn.status == SessionStatus.ERROR:
            print("Session ended with an error. Please check the message above.")
            sys.exit(1)

        if turn.status == SessionStatus.CONFLICT:
            print("⚠  Conflicts must be resolved before finalization.")
            reply = _user_input("Your response: ")
        elif turn.status == SessionStatus.AWAITING_CONFIRMATION:
            reply = _user_input("Your response (type 'confirm' to finalize): ")
        elif turn.status == SessionStatus.NEEDS_CLARIFICATION:
            reply = _user_input("Your answer: ")
        else:
            reply = _user_input("Your response: ")

        if not reply:
            print("Empty response — please provide an answer.")
            continue

        print(f"\n⏳  Processing ({client.provider_label})…")
        turn = process_user_reply(session_id, reply)
        _display_turn(turn)

    # --- Session finalized ---
    print("\n✅  Session complete.")
    print(f"    Session ID : {session_id}")
    print(f"    Output     : outputs/{session_id}.json")
    print(f"    Building IR: outputs/{session_id}_ir.json")


if __name__ == "__main__":
    main()
