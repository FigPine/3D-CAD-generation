"""
agents/req_ext_agent.py
========================
Requirements Extraction Agent orchestrator.

This module ties together all the subsystems:
  LLM provider → CandidateRequirements → RequirementState →
  CompletenessChecker → DialogueManager → RequirementsValidator →
  ExtractionResult

Public interface (matches the spec §3.1)
-----------------------------------------
    set_llm_client(client)           -> None         (call once at startup)
    start_session(prompt)            -> AgentTurn
    process_user_reply(sid, reply)   -> AgentTurn
    get_session_state(sid)           -> dict
    finalize_requirements(sid)       -> ExtractionResult

Design principles
-----------------
* The orchestrator decides WHEN to call the LLM (only for language
  interpretation) and NEVER for checklist evaluation, round counting, or
  default application.
* The active LLM client is set once at startup via ``set_llm_client()``.
  This allows any ``BaseLLMClient`` implementation (Gemini, LM Studio, …)
  to be used without modifying this module.
* All session state is held in an in-memory dict keyed by session_id.
  Sessions survive for the lifetime of the process.
* The three-round limit is enforced by ``DialogueManager``.
* "Confirm" / "yes" / "looks good" replies trigger finalization.
* Invalid LLM responses are retried by the client's own retry logic;
  if they still fail, an ``AgentTurn`` with status ERROR is returned.
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
from llm.base_client import BaseLLMClient, LLMError, LLMAuthError
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
# Module-level LLM client — set once at startup via set_llm_client()
# ---------------------------------------------------------------------------

_llm_client: Optional[BaseLLMClient] = None


def set_llm_client(client: BaseLLMClient) -> None:
    """
    Set the LLM client to use for all extraction calls in this session.

    Must be called once at application startup, before ``start_session()``.
    Accepts any ``BaseLLMClient`` implementation (GeminiClient, LMStudioClient…).
    """
    global _llm_client
    _llm_client = client


def _get_gemini_client() -> BaseLLMClient:
    """Construct a default Gemini client (legacy path / test hook)."""
    from llm.gemini_client import GeminiClient
    return GeminiClient()


def _get_llm_client() -> BaseLLMClient:
    """
    Return the active LLM client.

    Falls back to constructing a ``GeminiClient`` from the environment if
    ``set_llm_client()`` was never called (preserves backward compatibility
    for direct module imports in tests).
    """
    if _llm_client is not None:
        return _llm_client
    return _get_gemini_client()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_session(prompt: str) -> AgentTurn:
    """
    Start a new extraction session from the initial user prompt.

    1. Creates a fresh ``RequirementState``.
    2. Detects explicit units in the user prompt if present.
    3. Calls the active LLM provider to extract initial candidate requirements.
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

    # Detect explicit whole-building footprint in initial prompt
    fp_match = re.search(
        rf"\b(?:overall\s+footprint|building\s+footprint|overall\s+dimensions?|total\s+building\s+(?:size|dimensions?)|outer\s+dimensions?|entire\s+building\s+(?:dimensions?|size)|whole\s+building\s+(?:dimensions?|size))\b[:\s]+(?:width\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*,?\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?|(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?)",
        prompt,
        re.IGNORECASE,
    )
    if fp_match:
        state.has_user_explicit_footprint = True

    # --- Call LLM for initial extraction ---
    try:
        client = _get_llm_client()
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

    # Deterministically extract user dimensions, openings, and relationships
    det_updated = _extract_deterministic_user_inputs(reply, state)

    # --- Confirmation intent ---
    # Only finalize if the session was actively awaiting confirmation of the latest summary!
    if state.status == SessionStatus.AWAITING_CONFIRMATION and _is_confirmation(reply):
        return _finalize_session(state)

    # --- Initial Defaults Preference Handling ---
    clean_low = reply.strip().lower()
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
            has_explicit_content = det_updated or any(
                k in clean_low for k in (
                    "bedroom", "bathroom", "hall", "kitchen", "living", "room",
                    "width", "length", "dimension", "window", "door",
                    "share", "connect", "adjacent", "wall",
                )
            )
            if not has_explicit_content:
                return _build_agent_turn(state)

    # If hybrid mode is waiting for category specification
    if state.defaults_preference == "hybrid" and not state.hybrid_default_categories:
        pref, cats = parse_preference_reply(reply)
        if cats:
            state.hybrid_default_categories = cats
            has_explicit_content = det_updated or any(
                k in clean_low for k in (
                    "bedroom", "bathroom", "hall", "kitchen", "living", "room",
                    "width", "length", "dimension", "window", "door",
                    "share", "connect", "adjacent", "wall",
                )
            )
            if not has_explicit_content:
                return _build_agent_turn(state)

    # --- "Use defaults" intent at any time ---
    use_defaults = _dialogue.is_use_defaults_reply(reply)
    if use_defaults:
        state.set_defaults_preference("use_defaults")

    # --- Recompute pending fields from current state ---
    current_check = _checker.check(state)
    pending = list(current_check.missing_fields)
    if not state.allows_default("dimensions"):
        missing_dims = [r.room_id or r.room_type for r in state.rooms if r.width_m is None or r.length_m is None]
        if missing_dims and not any("room_dimensions" in p for p in pending):
            pending.append(f"room_dimensions ({', '.join(missing_dims)})")
    if not state.allows_default("openings"):
        missing_openings = [r.room_id or r.room_type for r in state.rooms if r.window_count is None or r.door_count is None]
        if missing_openings and not any("room_openings" in p for p in pending):
            pending.append(f"room_openings ({', '.join(missing_openings)})")
    if state.ambiguities:
        pending.extend(state.ambiguities)

    if _needs_llm_interpretation(reply, pending, use_defaults):
        try:
            client = _get_llm_client()
            current_known = _compact_current_values(state)
            raw_dict, _ = client.extract(
                system_instruction=SYSTEM_INSTRUCTION,
                user_content=build_followup_prompt(reply, pending, current_known),
            )
            candidate = _parse_candidate(raw_dict, state)
            state.merge_candidate(candidate, source=FieldSource.USER_EXPLICIT)
            # Re-apply deterministic inputs to ensure LLM nulls never overwrite explicit user inputs
            _extract_deterministic_user_inputs(reply, state)
        except LLMAuthError as exc:
            return _error_turn(state, f"Authentication error: {exc}")
        except LLMError as exc:
            check_after = _checker.check(state)
            if check_after.missing_fields and not any(r.width_m is not None for r in state.rooms):
                return _error_turn(state, f"LLM error processing reply: {exc}")
        except Exception as exc:  # noqa: BLE001
            return _error_turn(state, f"Unexpected error: {exc}")
    else:
        # Try deterministic parsing for simple structured replies
        _parse_simple_reply(reply, state)
        _extract_deterministic_user_inputs(reply, state)

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

    state.normalize_rooms()
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
        src = getattr(rel, "source", None) or FieldSource.USER_EXPLICIT
        provenance[f"rel_{rel.room_a}_{rel.relationship.value}_{rel.room_b}"] = src.value
        if getattr(rel, "source", None) == FieldSource.SYSTEM_DEFAULT:
            defaults_applied.append(f"rel_{rel.room_a}_{rel.relationship.value}_{rel.room_b}")

    # Build overall_dimensions in canonical metres
    wall_thickness = 0.2
    if "wall_thickness_m" in DEFAULTS:
        wall_thickness = DEFAULTS["wall_thickness_m"].value
    elif "wall_thickness_mm" in DEFAULTS:
        wall_thickness = DEFAULTS["wall_thickness_mm"].value / 1000.0

    floor_height = 3.0
    if "floor_height_m" in DEFAULTS:
        floor_height = DEFAULTS["floor_height_m"].value

    has_footprint = getattr(state, "has_user_explicit_footprint", False)
    if (
        has_footprint
        and state.overall_dimensions
        and state.overall_dimensions.width_m is not None
        and state.overall_dimensions.length_m is not None
    ):
        dims = Dimensions(
            width_m=state.overall_dimensions.width_m,
            length_m=state.overall_dimensions.length_m,
            floor_height_m=state.overall_dimensions.floor_height_m or floor_height,
            wall_thickness_m=state.overall_dimensions.wall_thickness_m or wall_thickness,
        )
        provenance["overall_dimensions.width_m"] = FieldSource.USER_EXPLICIT.value
        provenance["overall_dimensions.length_m"] = FieldSource.USER_EXPLICIT.value
    else:
        # Footprint width and length remain unset (null) pending layout engine computation
        floor_h = state.overall_dimensions.floor_height_m if state.overall_dimensions and state.overall_dimensions.floor_height_m else floor_height
        wall_th = state.overall_dimensions.wall_thickness_m if state.overall_dimensions and state.overall_dimensions.wall_thickness_m else wall_thickness
        dims = Dimensions(
            width_m=None,
            length_m=None,
            floor_height_m=floor_h,
            wall_thickness_m=wall_th,
        )

    state.synchronize_constraints_with_relationships()
    final_prefs: List[str] = []
    known_modes = {"use_defaults", "specify_details", "hybrid"}
    if state.defaults_preference in known_modes:
        final_prefs.append(state.defaults_preference)
    for p in state.preferences:
        if p not in final_prefs:
            final_prefs.append(p)

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
            preferences=final_prefs,
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
    from llm.base_client import LLMResponseFormatError

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
    clean = reply.strip().lower()
    # If the reply contains any words indicating a change, requirement, dimension,
    # opening, relationship, or room specification, it is NOT a confirmation.
    change_indicators = (
        "share", "wall", "connect", "adjacent", "door", "window",
        "width", "length", "dimension", "size", "floor", "height",
        "change", "modify", "update", "make", "set", "add", "remove",
        "replace", "instead", "not", "no ", "don't", "dont", "except",
        "bedroom", "bathroom", "hall", "kitchen", "living", "room",
    )
    if any(re.search(rf"\b{re.escape(w)}", clean) for w in change_indicators):
        return False
    return bool(_CONFIRM_RE.search(clean))


_UNIT_RE_STR = r"(?:m(?:etre[s]?|eter[s]?)?|cm|centimetre[s]?|centimeter[s]?|mm|millimetre[s]?|millimeter[s]?|ft|feet|foot|'|in|inch(?:es)?|\")"
_COPULA = r"(?:[:=]|to\s+be|should\s+be|must\s+be|will\s+be|can\s+be|shall\s+be|has\s+to\s+be|is|are)"


def _safe_normalize_dim(val: float, raw_unit: Optional[str], default_unit: str = "m") -> float:
    """Safely normalize a dimension to metres, falling back to default_unit on unsupported strings."""
    from requirements.unit_normalizer import canonicalize_unit_name, normalize_to_metres
    unit = default_unit
    if raw_unit:
        canon = canonicalize_unit_name(raw_unit)
        if canon:
            unit = canon
    try:
        return normalize_to_metres(val, unit)
    except Exception:
        return val


def _extract_deterministic_user_inputs(reply: str, state: RequirementState) -> bool:
    """
    Deterministically extract user-supplied room dimensions, openings,
    relationships, and footprint from clarification replies.

    Returns True if any user requirements were updated.
    """
    from requirements.requirement_state import (
        _extract_relationships_from_text,
        resolve_room_reference,
    )
    from schemas import RelationshipType, RoomRelationship

    updated = False
    clean_reply = reply.strip()
    if not clean_reply:
        return False

    def_u = state.user_supplied_units or "m"

    # Check explicit whole-building footprint dimensions (must specifically refer to whole-building footprint)
    footprint_match = re.search(
        rf"\b(?:overall\s+footprint|building\s+footprint|overall\s+dimensions?|total\s+building\s+(?:size|dimensions?)|outer\s+dimensions?|entire\s+building\s+(?:dimensions?|size)|whole\s+building\s+(?:dimensions?|size))\b[:\s]+(?:width\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*,?\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?|(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?)",
        clean_reply,
        re.IGNORECASE,
    )
    if footprint_match:
        if footprint_match.group(1) and footprint_match.group(3):
            w_val = float(footprint_match.group(1))
            w_u = footprint_match.group(2) or def_u
            l_val = float(footprint_match.group(3))
            l_u = footprint_match.group(4) or w_u
            w_m = _safe_normalize_dim(w_val, w_u, def_u)
            l_m = _safe_normalize_dim(l_val, l_u, def_u)
        elif footprint_match.group(5) and footprint_match.group(7):
            w_val = float(footprint_match.group(5))
            w_u = footprint_match.group(6) or def_u
            l_val = float(footprint_match.group(7))
            l_u = footprint_match.group(8) or w_u
            w_m = _safe_normalize_dim(w_val, w_u, def_u)
            l_m = _safe_normalize_dim(l_val, l_u, def_u)
        else:
            w_m = None
            l_m = None
        if w_m is not None and l_m is not None:
            state.overall_dimensions = Dimensions(
                width_m=w_m,
                length_m=l_m,
                floor_height_m=3.0,
                wall_thickness_m=0.2,
            )
            state.has_user_explicit_footprint = True
            updated = True

    # Split reply into lines, sentences, or semi-colon separated clauses
    lines = [ln.strip() for ln in re.split(r"[\r\n;]+|(?<!\d)\.\s*", clean_reply) if ln.strip()]

    for line in lines:
        clean_line = line.lstrip("-*• \t").strip()
        if not clean_line:
            continue

        target_rooms = []
        rest = clean_line

        # Check for room prefix before a colon, e.g. "bedroom_1: width 3 m, length 4 m"
        colon_parts = clean_line.split(":", 1)
        if len(colon_parts) == 2:
            room_prefix = colon_parts[0].strip()
            rest = colon_parts[1].strip()
            resolved_ids = resolve_room_reference(room_prefix, state)
            if resolved_ids:
                target_rooms = [r for r in state.rooms if r.room_id in resolved_ids or r.room_type in resolved_ids]

        # If not resolved by colon prefix, try to resolve from line content
        if not target_rooms:
            from requirements.requirement_state import parse_room_name

            # 1. Check exact canonical IDs or indexed patterns (e.g. bedroom_1, bedroom 1, bedroom1, bedroom-1)
            for r in state.rooms:
                r_id = r.room_id or r.room_type
                base, idx = parse_room_name(r_id)
                if idx is not None:
                    pat = rf"\b{re.escape(base)}[\s_#-]*{idx}\b"
                    if re.search(pat, clean_line, re.IGNORECASE):
                        if r not in target_rooms:
                            target_rooms.append(r)

            # 2. Check collective or plural references (e.g. "both bedrooms", "all bedrooms", "the bedrooms", "bedrooms")
            if not target_rooms:
                for r in state.rooms:
                    base, _ = parse_room_name(r.room_id or r.room_type)
                    coll_pat = rf"\b(?:both|all|the|each|every)?\s*{re.escape(base)}s\b"
                    if re.search(coll_pat, clean_line, re.IGNORECASE):
                        resolved_ids = resolve_room_reference(f"both {base}s", state)
                        if resolved_ids:
                            for rm in state.rooms:
                                if (rm.room_id in resolved_ids or rm.room_type in resolved_ids) and rm not in target_rooms:
                                    target_rooms.append(rm)
                            break

            # 3. Check unique single room type mentions (e.g. "bathroom", "the hall")
            if not target_rooms:
                type_counts: Dict[str, int] = {}
                for rm in state.rooms:
                    b, _ = parse_room_name(rm.room_id or rm.room_type)
                    type_counts[b] = type_counts.get(b, 0) + 1
                for r in state.rooms:
                    base, _ = parse_room_name(r.room_id or r.room_type)
                    if type_counts.get(base, 0) == 1:
                        if re.search(rf"\b{re.escape(base)}\b", clean_line, re.IGNORECASE):
                            if r not in target_rooms:
                                target_rooms.append(r)

        if not target_rooms:
            continue

        # --- Dimensions Extraction ---
        width_m = None
        length_m = None

        w_match = (
            re.search(rf"\b(?:width|wide)\b\s*(?:of\s+[a-zA-Z0-9_\s#-]+?)?\s*(?:of|is|are|to\s+be|[:=])?\s*(\d+(?:\.\d+)?)(?:\s*({_UNIT_RE_STR}))?\b", rest, re.IGNORECASE)
            or re.search(rf"\bwidth\b.*?(?:{_COPULA})\s*(\d+(?:\.\d+)?)(?:\s*({_UNIT_RE_STR}))?\b", rest, re.IGNORECASE)
            or re.search(rf"(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*(?:in\s+)?(?:width|wide)\b", rest, re.IGNORECASE)
        )
        l_match = (
            re.search(rf"\b(?:length|long)\b\s*(?:of\s+[a-zA-Z0-9_\s#-]+?)?\s*(?:of|is|are|to\s+be|[:=])?\s*(\d+(?:\.\d+)?)(?:\s*({_UNIT_RE_STR}))?\b", rest, re.IGNORECASE)
            or re.search(rf"\blength\b.*?(?:{_COPULA})\s*(\d+(?:\.\d+)?)(?:\s*({_UNIT_RE_STR}))?\b", rest, re.IGNORECASE)
            or re.search(rf"(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*(?:in\s+)?(?:length|long)\b", rest, re.IGNORECASE)
        )
        if w_match:
            val = float(w_match.group(1))
            unit = w_match.group(2) or def_u
            width_m = _safe_normalize_dim(val, unit, def_u)
        if l_match:
            val = float(l_match.group(1))
            unit = l_match.group(2) or def_u
            length_m = _safe_normalize_dim(val, unit, def_u)

        # Pattern: X [m] x Y [m] or X by Y
        if width_m is None and length_m is None:
            dim_match = re.search(
                rf"(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*({_UNIT_RE_STR})?",
                rest,
                re.IGNORECASE,
            )
            if dim_match:
                w_val = float(dim_match.group(1))
                w_unit = dim_match.group(2) or def_u
                l_val = float(dim_match.group(3))
                l_unit = dim_match.group(4) or w_unit
                width_m = _safe_normalize_dim(w_val, w_unit, def_u)
                length_m = _safe_normalize_dim(l_val, l_unit, def_u)

        if width_m is not None:
            for tr in target_rooms:
                tr.width_m = width_m
                target_key = tr.room_id or tr.room_type
                state.set_room_field_source(target_key, "width_m", FieldSource.USER_EXPLICIT)
            updated = True

        if length_m is not None:
            for tr in target_rooms:
                tr.length_m = length_m
                target_key = tr.room_id or tr.room_type
                state.set_room_field_source(target_key, "length_m", FieldSource.USER_EXPLICIT)
            updated = True

        # --- Openings Extraction ---
        win_match = re.search(r"\b(\d+|no|zero|one|two|three)\s*windows?\b", rest, re.IGNORECASE)
        if win_match:
            txt = win_match.group(1).lower()
            word_map = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3}
            w_cnt = word_map.get(txt, int(txt) if txt.isdigit() else 0)
            for tr in target_rooms:
                tr.window_count = w_cnt
                target_key = tr.room_id or tr.room_type
                state.set_room_field_source(target_key, "window_count", FieldSource.USER_EXPLICIT)
            updated = True

        door_match = re.search(r"\b(\d+|no|zero|one|two|three)\s*(?:main\s+entrance\s+)?doors?\b", rest, re.IGNORECASE)
        if door_match:
            txt = door_match.group(1).lower()
            word_map = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3}
            d_cnt = word_map.get(txt, int(txt) if txt.isdigit() else 1)
            for tr in target_rooms:
                tr.door_count = d_cnt
                target_key = tr.room_id or tr.room_type
                state.set_room_field_source(target_key, "door_count", FieldSource.USER_EXPLICIT)
                if "main entrance" in rest.lower() or "entrance" in rest.lower():
                    if not tr.notes:
                        tr.notes = "main entrance door"
                    elif "main entrance" not in tr.notes:
                        tr.notes += ", main entrance door"
            updated = True

    # Extract any relationships explicitly stated in reply
    extracted_rels = _extract_relationships_from_text(reply, state)
    if extracted_rels:
        for r in extracted_rels:
            r.source = FieldSource.USER_EXPLICIT
        state._merge_relationships(extracted_rels, source=FieldSource.USER_EXPLICIT)
        updated = True

    # Extract any user-explicit constraint statements from reply
    constraint_split_pattern = r"[\r\n;.]+|\band\s+(?=(?:the\s+)?[a-zA-Z0-9_-]+\s+(?:must|must\s+not|should|shall|cannot|never)\b)"
    constraint_lines = [ln.strip() for ln in re.split(constraint_split_pattern, clean_reply) if ln.strip()]
    for cl in constraint_lines:
        cl_clean = cl.lstrip("-*• \t").strip()
        cl_low = cl_clean.lower()
        if any(w in cl_low for w in ("must", "must not", "should", "shall", "cannot", "never", "require", "required")):
            if any(k in cl_low for k in ("share", "wall", "connect", "adjacent", "door", "window", "room", "dimension", "floor", "level")):
                if not any(ign in cl_low for ign in ("use default", "default value", "everything else", "rest of")):
                    if cl_clean not in state.constraints:
                        state.constraints.append(cl_clean)
                        updated = True

    state.synchronize_constraints_with_relationships()

    return updated


def _needs_llm_interpretation(
    reply: str, pending: list, use_defaults: bool
) -> bool:
    """
    Decide whether this reply needs the LLM to interpret it.
    """
    if use_defaults:
        return False
    word_count = len(reply.split())
    if word_count <= 2:
        clean = reply.strip().lower()
        if not any(k in clean for k in ("bedroom", "bathroom", "hall", "kitchen", "living", "room", "connect", "adjacent", "wall")):
            return False
    return True


def _parse_simple_reply(reply: str, state: RequirementState) -> None:
    """
    Attempt deterministic parsing of very simple replies.

    Handles patterns like:
    - "residential" / "commercial" -> building_type
    - "1" / "2" -> floor_count (if it's pending)
    - "mm" / "m" -> units
    """
    clean = reply.strip().lower()

    bt_entry = state.checklist["building_type"]
    if bt_entry.status == FieldStatus.MISSING:
        for bt in ("residential", "commercial", "mixed", "other"):
            if bt in clean:
                state._update_scalar("building_type", bt, FieldSource.USER_EXPLICIT)
                break

    fc_entry = state.checklist["floor_count"]
    if fc_entry.status == FieldStatus.MISSING:
        m = re.search(r"\b(\d+)\s*floor", clean)
        if m:
            state._update_scalar("floor_count", int(m.group(1)), FieldSource.USER_EXPLICIT)
        elif re.fullmatch(r"\d+", clean):
            state._update_scalar("floor_count", int(clean), FieldSource.USER_EXPLICIT)

    units_entry = state.checklist["units"]
    if units_entry.status == FieldStatus.MISSING:
        for u in ("mm", "cm", "m", "ft", "in"):
            if re.search(rf"\b{u}\b", clean):
                state.user_supplied_units = u
                state._update_scalar("units", "m", FieldSource.USER_EXPLICIT)
                state.checklist["units"].notes = f"Canonical unit: metres (user specified: {u})"
                break

    # Deterministic room dimension updates or corrections
    for r in state.rooms:
        r_id = (r.room_id or r.room_type).lower()
        if r_id in clean or r.room_type.lower() in clean:
            target_id = r.room_id or r.room_type
            w_m = re.search(r"\b(?:width|wide)\b\s*(?:of|is|are|to\s+be|[:=])?\s*(\d+(?:\.\d+)?)\s*(?:m|metres?)?\b", clean)
            l_m = re.search(r"\b(?:length|long)\b\s*(?:of|is|are|to\s+be|[:=])?\s*(\d+(?:\.\d+)?)\s*(?:m|metres?)?\b", clean)
            if w_m:
                r.width_m = float(w_m.group(1))
                state.set_room_field_source(target_id, "width_m", FieldSource.USER_EXPLICIT)
            if l_m:
                r.length_m = float(l_m.group(1))
                state.set_room_field_source(target_id, "length_m", FieldSource.USER_EXPLICIT)
            if not w_m and not l_m:
                dim_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|metres?)?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:m|metres?)?", clean)
                if dim_match:
                    r.width_m = float(dim_match.group(1))
                    r.length_m = float(dim_match.group(2))
                    state.set_room_field_source(target_id, "width_m", FieldSource.USER_EXPLICIT)
                    state.set_room_field_source(target_id, "length_m", FieldSource.USER_EXPLICIT)


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
