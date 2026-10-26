"""
agents/req_ext_agent.py
========================
Requirements Extraction Agent orchestrator.

This module ties together all the subsystems:
  GeminiClient → CandidateRequirements → RequirementState →
  CompletenessChecker → DialogueManager → RequirementsValidator →
  ExtractionResult

Public interface (matches the spec §3.1)
-----------------------------------------
    start_session(prompt)            -> AgentTurn
    process_user_reply(sid, reply)   -> AgentTurn
    get_session_state(sid)           -> dict
    finalize_requirements(sid)       -> ExtractionResult

Design principles
-----------------
* The orchestrator decides WHEN to call Gemini (only for language
  interpretation) and NEVER for checklist evaluation, round counting, or
  default application.
* All session state is held in an in-memory dict keyed by session_id.
  Sessions survive for the lifetime of the process.
* The three-round limit is enforced by ``DialogueManager``.
* "Confirm" / "yes" / "looks good" replies trigger finalization.
* Invalid LLM responses are retried once via ``GeminiClient``'s own retry
  logic; if they still fail, an ``AgentTurn`` with status ERROR is returned.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from schemas import (
    AgentTurn,
    BuildingType,
    CandidateRequirements,
    Dimensions,
    ExtractionResult,
    FieldSource,
    FieldStatus,
    SessionStatus,
)
from requirements.requirement_state import RequirementState
from requirements.completeness_checker import CompletenessChecker
from requirements.dialogue_manager import (
    DEFAULTS_PREFERENCE_PROMPT,
    DialogueManager,
    parse_preference_reply,
)
from requirements.defaults import DEFAULTS
from requirements.unit_normalizer import (
    canonicalize_unit_name,
    detect_unit_in_text,
    normalize_to_metres,
)
from validators.requirements_validator import RequirementsValidator, ValidationError
from llm.gemini_client import GeminiClient, LLMError, LLMAuthError
from llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_extraction_prompt,
    build_followup_prompt,
)

# ---------------------------------------------------------------------------
# Simple keyword detection for user confirmation replies
# ---------------------------------------------------------------------------

_CONFIRM_PATTERNS = [
    r"\bconfirm\b",
    r"\byes\b",
    r"\byep\b",
    r"\bapprove\b",
    r"\blooks?\s+good\b",
    r"\bthat['']?s?\s+(correct|right|fine|ok|okay)\b",
    r"\bproceed\b",
    r"\bfinalize\b",
    r"\baccept\b",
]
_CONFIRM_RE = re.compile("|".join(_CONFIRM_PATTERNS), re.IGNORECASE)

# In-memory session store: session_id -> RequirementState
_SESSIONS: Dict[str, RequirementState] = {}

# Shared subsystem instances (stateless — safe to share)
_checker = CompletenessChecker()
_dialogue = DialogueManager()
_validator = RequirementsValidator()


# ---------------------------------------------------------------------------
# Helper: build GeminiClient only when needed (deferred so tests can mock)
# ---------------------------------------------------------------------------

def _get_gemini_client() -> GeminiClient:
    """Return a ``GeminiClient`` configured from environment variables."""
    return GeminiClient()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_session(prompt: str) -> AgentTurn:
    """
    Start a new extraction session from the initial user prompt.

    1. Creates a fresh ``RequirementState``.
    2. Detects explicit units in the user prompt if present.
    3. Calls Gemini to extract initial candidate requirements (explicit user values).
    4. Merges the candidate into the state.
    5. Returns an AgentTurn asking how the user wants unspecified details handled.
    """
    state = RequirementState()
    state._initial_prompt = prompt
    _SESSIONS[state.session_id] = state

    # Detect unit mentions in the prompt text
    det_unit, is_ambig = detect_unit_in_text(prompt)
    if is_ambig:
        state.ambiguities.append(
            "Ambiguous unit mention detected in building description. Please clarify measurement units."
        )
    elif det_unit:
        state.user_supplied_units = det_unit
        state._update_scalar("units", "m", FieldSource.USER_EXPLICIT)
        state.checklist["units"].notes = f"Canonical unit: metres (user specified: {det_unit})"

    # --- Call Gemini for initial extraction ---
    try:
        client = _get_gemini_client()
        raw_dict, usage = client.extract(
            system_instruction=SYSTEM_INSTRUCTION,
            user_content=build_extraction_prompt(prompt),
        )
        candidate = _parse_candidate(raw_dict, state)
        state.merge_candidate(candidate, source=FieldSource.USER_EXPLICIT)
        state.normalize_rooms()
        state.recalculate_warnings()

    except LLMAuthError as exc:
        return _error_turn(state, f"Authentication error: {exc}")
    except LLMError as exc:
        return _error_turn(state, f"LLM error during extraction: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _error_turn(state, f"Unexpected error during extraction: {exc}")

    # Return the initial defaults preference prompt
    return AgentTurn(
        status=SessionStatus.NEEDS_CLARIFICATION,
        session_id=state.session_id,
        clarification_round=state.clarification_round,
        message=DEFAULTS_PREFERENCE_PROMPT,
        pending_fields=["defaults_preference"],
        requirements=_compact_current_values(state),
        errors=[],
        warnings=list(state.warnings),
        is_finalized=False,
    )


def process_user_reply(session_id: str, reply: str) -> AgentTurn:
    """
    Process a user's reply and advance the session.

    1. Retrieve session state.
    2. Check for confirmation intent (finalize only if awaiting confirmation).
    3. Check and apply initial defaults preference if pending.
    4. Check for 'use defaults' intent at any round.
    5. Interpret reply (LLM or deterministic parsing).
    6. Merge updates without overwriting explicit user values.
    7. Increment round (if not initial preference turn) and return next turn.
    """
    state = _SESSIONS.get(session_id)
    if state is None:
        return AgentTurn(
            status=SessionStatus.ERROR,
            session_id=session_id,
            message=f"Session '{session_id}' not found. Please start a new session.",
            errors=["session_not_found"],
        )

    if not reply or not reply.strip():
        return _build_agent_turn(state, extra_warning="Empty reply received; please provide an answer.")

    # Check for resolution of specific ambiguities in user reply
    state.check_user_reply_resolutions(reply)

    # --- Confirmation intent ---
    # Only finalize if the session was actively awaiting confirmation of the latest summary!
    if state.status == SessionStatus.AWAITING_CONFIRMATION and _is_confirmation(reply):
        return _finalize_session(state)

    # --- Initial Defaults Preference Handling ---
    if state.defaults_preference is None:
        pref, cats = parse_preference_reply(reply)
        clean = reply.strip().lower()
        if pref == "hybrid" and not cats and clean in ("3", "3.", "hybrid", "3. hybrid", "option 3"):
            state.set_defaults_preference("hybrid", [])
            return AgentTurn(
                status=SessionStatus.NEEDS_CLARIFICATION,
                session_id=state.session_id,
                clarification_round=state.clarification_round,
                message=(
                    "Which categories would you like to default, and which would you like to specify manually?\n"
                    "(e.g. 'use default dimensions, but ask me about room connections')"
                ),
                pending_fields=["defaults_preference_categories"],
                requirements=_compact_current_values(state),
                errors=[],
                warnings=list(state.warnings),
                is_finalized=False,
            )
        if pref:
            state.set_defaults_preference(pref, cats)
            return _build_agent_turn(state)

    # If hybrid mode is waiting for category specification
    if state.defaults_preference == "hybrid" and not state.hybrid_default_categories:
        pref, cats = parse_preference_reply(reply)
        if cats:
            state.hybrid_default_categories = cats
            return _build_agent_turn(state)

    # --- "Use defaults" intent at any time ---
    use_defaults = _dialogue.is_use_defaults_reply(reply)
    if use_defaults:
        state.set_defaults_preference("use_defaults")

    # --- Determine whether we need Gemini to interpret the reply ---
    current_check = _checker.check(state)
    pending = current_check.missing_fields

    if _needs_llm_interpretation(reply, pending, use_defaults):
        try:
            client = _get_gemini_client()
            current_known = _compact_current_values(state)
            raw_dict, _ = client.extract(
                system_instruction=SYSTEM_INSTRUCTION,
                user_content=build_followup_prompt(reply, pending, current_known),
            )
            candidate = _parse_candidate(raw_dict, state)
            state.merge_candidate(candidate, source=FieldSource.USER_EXPLICIT)
        except LLMAuthError as exc:
            return _error_turn(state, f"Authentication error: {exc}")
        except LLMError as exc:
            return _error_turn(state, f"LLM error processing reply: {exc}")
        except Exception as exc:  # noqa: BLE001
            return _error_turn(state, f"Unexpected error: {exc}")
    else:
        # Try deterministic parsing for simple structured replies
        _parse_simple_reply(reply, state)

    from requirements.dialogue_manager import MAX_ROUNDS
    if state.clarification_round < MAX_ROUNDS:
        state.increment_round()
    return _build_agent_turn(state, use_defaults=use_defaults)


def get_session_state(session_id: str) -> Dict[str, Any]:
    """
    Return the serialisable state for a session (no secrets).

    Parameters
    ----------
    session_id : str

    Returns
    -------
    dict  (JSON-serialisable)
    """
    state = _SESSIONS.get(session_id)
    if state is None:
        return {"error": f"Session '{session_id}' not found."}
    return state.to_dict()


def finalize_requirements(session_id: str) -> ExtractionResult:
    """
    Validate and package the final requirements for the Building IR module.

    Raises
    ------
    ValidationError
        If the state does not pass final validation.
    KeyError
        If the session is not found.
    """
    state = _SESSIONS.get(session_id)
    if state is None:
        raise KeyError(f"Session '{session_id}' not found.")
    state.normalize_rooms()
    _validator.validate_or_raise(state)
    return _build_extraction_result(state)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_agent_turn(
    state: RequirementState,
    extra_warning: Optional[str] = None,
    use_defaults: bool = False,
) -> AgentTurn:
    """Run completeness check and build the appropriate AgentTurn."""
    if extra_warning:
        if extra_warning not in state.warnings:
            state.warnings.append(extra_warning)

    state.recalculate_warnings()
    check = _checker.check(state)
    status, message, pending = _dialogue.decide_next_action(
        state, check, user_said_use_defaults=use_defaults
    )
    state.status = status

    return AgentTurn(
        status=status,
        session_id=state.session_id,
        clarification_round=state.clarification_round,
        message=message,
        pending_fields=pending,
        requirements=_compact_current_values(state),
        errors=[],
        warnings=list(state.warnings),
        is_finalized=False,
    )


def _finalize_session(state: RequirementState) -> AgentTurn:
    """
    Run final validation and return a FINALIZED AgentTurn with the result.
    """
    state.normalize_rooms()

    try:
        _validator.validate_or_raise(state)
    except ValidationError as exc:
        return AgentTurn(
            status=SessionStatus.ERROR,
            session_id=state.session_id,
            clarification_round=state.clarification_round,
            message=(
                "Finalization failed due to validation errors:\n"
                + "\n".join(f"  • {e}" for e in exc.errors)
            ),
            errors=exc.errors,
            is_finalized=False,
        )

    state.confirm()

    try:
        result = _build_extraction_result(state)
    except ValidationError as exc:
        return AgentTurn(
            status=SessionStatus.ERROR,
            session_id=state.session_id,
            clarification_round=state.clarification_round,
            message="Could not build final result: " + "; ".join(exc.errors),
            errors=exc.errors,
            is_finalized=False,
        )

    # Save JSON output
    _save_output(state.session_id, result)

    return AgentTurn(
        status=SessionStatus.FINALIZED,
        session_id=state.session_id,
        clarification_round=state.clarification_round,
        message=(
            "✅ Requirements finalized! The structured requirements have been saved.\n"
            f"Session ID: {state.session_id}"
        ),
        requirements=result.model_dump(),
        is_finalized=True,
    )


def _build_extraction_result(state: RequirementState) -> ExtractionResult:
    """
    Build and validate an ``ExtractionResult`` from the current state.

    Raises ``ValidationError`` if the state is not ready.
    """
    _validator.validate_or_raise(state)

    # Assemble provenance map
    provenance: Dict[str, str] = {}
    for field_name, entry in state.checklist.items():
        if entry.source:
            provenance[field_name] = entry.source.value
    for r_id, fields in state.room_field_sources.items():
        for f_name, src in fields.items():
            provenance[f"{r_id}.{f_name}"] = src.value

    defaults_applied = [
        f for f, entry in state.checklist.items()
        if entry.source == FieldSource.SYSTEM_DEFAULT
    ]
    for r_id, fields in state.room_field_sources.items():
        for f_name, src in fields.items():
            if src == FieldSource.SYSTEM_DEFAULT:
                defaults_applied.append(f"{r_id}_{f_name}")
    for rel in state.relationships:
        if getattr(rel, "source", None) == FieldSource.SYSTEM_DEFAULT:
            defaults_applied.append(f"rel_{rel.room_a}_{rel.relationship.value}_{rel.room_b}")

    # Build overall_dimensions from state defaults in canonical metres
    wall_thickness = 0.2
    if "wall_thickness_m" in DEFAULTS:
        wall_thickness = DEFAULTS["wall_thickness_m"].value
    elif "wall_thickness_mm" in DEFAULTS:
        wall_thickness = DEFAULTS["wall_thickness_mm"].value / 1000.0

    floor_height = 3.0
    if "floor_height_m" in DEFAULTS:
        floor_height = DEFAULTS["floor_height_m"].value

    dims = Dimensions(
        width_m=None,
        length_m=None,
        floor_height_m=floor_height,
        wall_thickness_m=wall_thickness,
    )

    try:
        result = ExtractionResult(
            session_id=state.session_id,
            building_type=BuildingType(state.checklist["building_type"].value),
            floor_count=state.checklist["floor_count"].value,
            units="m",
            user_supplied_units=state.user_supplied_units,
            rooms=list(state.rooms),
            overall_dimensions=dims,
            relationships=list(state.relationships),
            constraints=list(state.constraints),
            preferences=list(state.preferences),
            provenance=provenance,
            defaults_applied=defaults_applied,
            warnings=list(state.warnings),
            confirmed_by_user=state.confirmed_by_user,
        )
    except Exception as exc:
        raise ValidationError([str(exc)]) from exc

    return result


def _parse_candidate(
    raw_dict: Dict[str, Any], state: RequirementState
) -> CandidateRequirements:
    """
    Parse the raw LLM response dict into a ``CandidateRequirements``.

    Raises ``LLMResponseFormatError`` on schema mismatch.
    """
    from llm.gemini_client import LLMResponseFormatError

    try:
        return CandidateRequirements(**raw_dict)
    except Exception as exc:
        raise LLMResponseFormatError(
            f"LLM response does not match CandidateRequirements schema: {exc}"
        ) from exc


def _error_turn(state: RequirementState, message: str) -> AgentTurn:
    """Build an error AgentTurn without changing state."""
    return AgentTurn(
        status=SessionStatus.ERROR,
        session_id=state.session_id,
        clarification_round=state.clarification_round,
        message=message,
        errors=[message],
        is_finalized=False,
    )


def _is_confirmation(reply: str) -> bool:
    """Return True if the reply is a clear confirmation."""
    return bool(_CONFIRM_RE.search(reply.strip()))


def _needs_llm_interpretation(
    reply: str, pending: list, use_defaults: bool
) -> bool:
    """
    Decide whether this reply needs Gemini to interpret it.

    Heuristic: call Gemini if the reply is longer than a few words or
    mentions room types (suggesting free-form natural language).
    Skip Gemini for very short replies like "2", "yes", "use defaults".
    """
    if use_defaults:
        return False  # handled deterministically
    word_count = len(reply.split())
    if word_count <= 2:
        return False  # likely a simple value like "2" or "residential"
    # If rooms are pending, any multi-word reply probably needs NL parsing
    if "rooms" in pending or any("room" in p for p in pending):
        return True
    # Multi-word reply for other fields — let Gemini interpret
    return word_count > 4


def _parse_simple_reply(reply: str, state: RequirementState) -> None:
    """
    Attempt deterministic parsing of very simple replies.

    Handles patterns like:
    - "residential" / "commercial" → building_type
    - "1" / "2" → floor_count (if it's pending)
    - "mm" / "m" → units
    """
    clean = reply.strip().lower()

    bt_entry = state.checklist["building_type"]
    if bt_entry.status == FieldStatus.MISSING:
        for bt in ("residential", "commercial", "mixed", "other"):
            if bt in clean:
                state._update_scalar("building_type", bt, FieldSource.USER_EXPLICIT)
                return

    fc_entry = state.checklist["floor_count"]
    if fc_entry.status == FieldStatus.MISSING:
        m = re.search(r"\b(\d+)\s*floor", clean)
        if m:
            state._update_scalar("floor_count", int(m.group(1)), FieldSource.USER_EXPLICIT)
            return
        if re.fullmatch(r"\d+", clean):
            state._update_scalar("floor_count", int(clean), FieldSource.USER_EXPLICIT)
            return

    units_entry = state.checklist["units"]
    if units_entry.status == FieldStatus.MISSING:
        for u in ("mm", "cm", "m", "ft", "in"):
            if re.search(rf"\b{u}\b", clean):
                state.user_supplied_units = u
                state._update_scalar("units", "m", FieldSource.USER_EXPLICIT)
                state.checklist["units"].notes = f"Canonical unit: metres (user specified: {u})"
                return

    # Deterministic room dimension updates or corrections
    for r in state.rooms:
        r_id = (r.room_id or r.room_type).lower()
        if r_id in clean or r.room_type.lower() in clean:
            dim_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|metres?)?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:m|metres?)?", clean)
            if dim_match:
                w = float(dim_match.group(1))
                l = float(dim_match.group(2))
                target_id = r.room_id or r.room_type
                r.width_m = w
                r.length_m = l
                state.set_room_field_source(target_id, "width_m", FieldSource.USER_EXPLICIT)
                state.set_room_field_source(target_id, "length_m", FieldSource.USER_EXPLICIT)
                return


def _compact_current_values(state: RequirementState) -> Dict[str, Any]:
    """Build a compact dict of currently-known values for context."""
    result: Dict[str, Any] = {}
    for k, entry in state.checklist.items():
        if entry.value is not None:
            result[k] = entry.value
    if state.user_supplied_units:
        result["user_supplied_units"] = state.user_supplied_units
    if state.defaults_preference:
        result["defaults_preference"] = state.defaults_preference
    if state.rooms:
        result["rooms"] = [
            {
                "room_id": r.room_id,
                "room_type": r.room_type,
                "count": r.count,
                "width_m": r.width_m,
                "length_m": r.length_m,
                "window_count": r.window_count,
                "door_count": r.door_count,
            }
            for r in state.rooms
        ]
    if state.relationships:
        result["relationships"] = [
            {
                "room_a": r.room_a,
                "room_b": r.room_b,
                "relationship": r.relationship.value,
                "negative": getattr(r, "negative", False),
            }
            for r in state.relationships
        ]
    return result


def _save_output(session_id: str, result: ExtractionResult) -> None:
    """Save the final ExtractionResult as JSON to the outputs/ directory."""
    try:
        outputs_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "outputs"
        )
        os.makedirs(outputs_dir, exist_ok=True)
        out_path = os.path.join(outputs_dir, f"{session_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
    except Exception:  # noqa: BLE001
        pass  # saving is best-effort; don't crash finalization
