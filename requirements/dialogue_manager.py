"""
requirements/dialogue_manager.py
=================================
Manages the clarification dialogue: round counting, question generation,
preference handling, default-application decisions, and summary building.

Responsibilities (all deterministic — no LLM calls here)
---------------------------------------------------------
* Enforce initial preference for handling unspecified details (Defaults, Specify, Hybrid).
* Track and enforce the 3-round maximum for clarifications.
* Generate focused, template-based clarification questions from missing fields.
* Detect "use defaults" intent or preference selections from user replies.
* Apply eligible defaults only according to the user's preference and allowed categories.
* Build the human-readable final summary for user confirmation disclosing all assumptions.

The agent orchestrator calls this manager to decide what happens after each
state update, keeping the LLM call decision logic outside the manager.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from schemas import FieldSource, FieldStatus, SessionStatus
from requirements.requirement_state import RequirementState
from requirements.completeness_checker import CheckResult, CompletenessChecker
from requirements.defaults import (
    DEFAULTS,
    NO_DEFAULT_FIELDS,
    format_defaults_summary,
    get_room_dimension_defaults,
    get_room_opening_defaults,
    propose_residential_relationship_defaults,
)


MAX_ROUNDS: int = 3

DEFAULTS_PREFERENCE_PROMPT = """\
How would you like to handle details you haven't specified?

1. Use defaults
2. I'll specify the details
3. Hybrid"""

# Keywords that indicate the user wants system defaults applied
_USE_DEFAULTS_PATTERNS = [
    r"\buse\s+defaults?\b",
    r"\bapply\s+defaults?\b",
    r"\bdefault\s+values?\b",
    r"\bjust\s+use\s+defaults?\b",
    r"\bgo\s+with\s+defaults?\b",
    r"\bdefaults?\s+are\s+(fine|ok|okay|good)\b",
    r"\bkeep\s+defaults?\b",
]
_USE_DEFAULTS_RE = re.compile(
    "|".join(_USE_DEFAULTS_PATTERNS), re.IGNORECASE
)


def parse_preference_reply(reply: str) -> Tuple[Optional[str], List[str]]:
    """
    Parse the user's response to the initial defaults preference prompt.

    Returns
    -------
    (preference, hybrid_categories)
        preference: 'use_defaults', 'specify_details', 'hybrid', or None
        hybrid_categories: list of categories permitted to default in hybrid mode
    """
    clean = reply.strip().lower()

    # Exact or leading number matches
    if clean in ("1", "1.", "1. use defaults", "use defaults", "defaults", "option 1", "use default", "default"):
        return "use_defaults", []

    if clean in ("2", "2.", "2. i'll specify the details", "i'll specify the details", "specify details", "specify", "i will specify", "option 2", "ask me"):
        return "specify_details", []

    if clean in ("3", "3.", "3. hybrid", "hybrid", "option 3"):
        # Hybrid requested without immediate categories — default to empty categories (prompts user or default category set)
        return "hybrid", []

    # Check for hybrid instructions such as:
    # "use default dimensions, but ask me about room connections"
    # "default dimensions, specify connections"
    # "hybrid: use default dimensions and openings"
    is_hybrid = False
    categories: List[str] = []

    if "hybrid" in clean or "but" in clean or ("default" in clean and any(w in clean for w in ("ask", "specify", "manual", "connect"))):
        is_hybrid = True

    # 1. Dimensions
    if re.search(r"\b(default\s+dimensions?|dimensions?\s+default|use\s+default\s+dimensions?)\b", clean):
        is_hybrid = True
        categories.append("dimensions")
    elif "dimension" in clean and "default" in clean and not re.search(r"(ask|specify).*dimension", clean):
        is_hybrid = True
        categories.append("dimensions")

    # 2. Openings (doors/windows)
    if re.search(r"\b(default\s+(?:openings?|doors?|windows?)|(?:openings?|doors?|windows?)\s+default)\b", clean):
        is_hybrid = True
        categories.append("openings")

    # 3. Relationships / Connections
    if re.search(r"\b(default\s+(?:connections?|relationships?)|(?:connections?|relationships?)\s+default)\b", clean):
        is_hybrid = True
        categories.append("relationships")

    # 4. Units
    if re.search(r"\b(default\s+units?|units?\s+default)\b", clean):
        is_hybrid = True
        categories.append("units")

    if is_hybrid:
        return "hybrid", categories

    # Fallback pattern matches
    if _USE_DEFAULTS_RE.search(clean):
        return "use_defaults", []
    if any(p in clean for p in ("specify", "ask me", "i will provide")):
        return "specify_details", []

    return None, []


class DialogueManager:
    """
    Stateless helper called by the agent orchestrator after each state update.

    Methods return (next_status, message, pending_fields).
    """

    def __init__(self) -> None:
        self._checker = CompletenessChecker()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def is_use_defaults_reply(self, reply: str) -> bool:
        """Return True if the user's reply clearly requests system defaults."""
        return bool(_USE_DEFAULTS_RE.search(reply))

    def apply_all_eligible_defaults(
        self, state: RequirementState
    ) -> List[str]:
        """
        Apply system defaults to MISSING eligible fields in ``state``,
        strictly respecting the selected defaults preference and permitted categories.

        Returns a list of default keys / descriptions that were applied.
        Never overwrites explicit user values or CONFLICT fields.
        """
        applied: List[str] = []

        # --- 1. Units default ---
        if state.allows_default("units"):
            entry = state.checklist.get("units")
            if entry and entry.status == FieldStatus.MISSING:
                state.apply_default(
                    "units",
                    "m",
                    unit="m",
                    notes="Metres are the canonical storage unit for linear dimensions.",
                )
                applied.append("units")

        # --- 2. Dimension defaults ---
        if state.allows_default("dimensions"):
            # Floor height
            if "floor_height_m" not in applied:
                applied.append("floor_height_m")
            # Wall thickness (0.2 m)
            if "wall_thickness_m" not in applied:
                applied.append("wall_thickness_m")

            # Room dimension defaults
            for room in state.rooms:
                from requirements.requirement_state import parse_room_name
                base_type, _ = parse_room_name(room.room_id or room.room_type)
                dim_defs = get_room_dimension_defaults(base_type)
                if dim_defs["width_m"] is not None or dim_defs["length_m"] is not None:
                    res = state.apply_room_dimension_defaults(
                        room.room_id or room.room_type,
                        width_m=dim_defs["width_m"],
                        length_m=dim_defs["length_m"],
                    )
                    applied.extend(res)

        # --- 3. Room opening defaults ---
        if state.allows_default("openings"):
            for room in state.rooms:
                from requirements.requirement_state import parse_room_name
                base_type, _ = parse_room_name(room.room_id or room.room_type)
                opening_defaults = get_room_opening_defaults(base_type)
                if opening_defaults["window_count"] is None and opening_defaults["door_count"] is None:
                    opening_defaults = get_room_opening_defaults(room.room_type)

                r_key = room.room_id or room.room_type
                if room.window_count is None and opening_defaults["window_count"] is not None:
                    state.apply_room_opening_defaults(
                        r_key,
                        window_count=opening_defaults["window_count"],
                        door_count=None,
                    )
                    applied.append(f"{r_key}_window_count")
                if room.door_count is None and opening_defaults["door_count"] is not None:
                    state.apply_room_opening_defaults(
                        r_key,
                        window_count=None,
                        door_count=opening_defaults["door_count"],
                    )
                    applied.append(f"{r_key}_door_count")

        # --- 4. Relationship defaults (propose hall-to-bedrooms/bathroom) ---
        if state.allows_default("relationships"):
            bt = state.get_scalar("building_type")
            if bt == "residential":
                proposed = propose_residential_relationship_defaults(
                    rooms=state.rooms,
                    existing_relationships=state.relationships,
                    constraints=state.constraints,
                )
                if proposed:
                    rel_applied = state.apply_relationship_defaults(proposed)
                    applied.extend(rel_applied)

        return applied

    def _has_unspecified_manual_details(self, state: RequirementState) -> bool:
        """
        Return True if there are room properties or relationships that the user
        has elected to specify manually but has not yet provided.
        """
        # If dimensions are not defaulted, check if any room lacks dimensions
        if not state.allows_default("dimensions"):
            for r in state.rooms:
                if r.width_m is None or r.length_m is None:
                    return True

        # If openings are not defaulted, check if any room lacks window/door counts
        if not state.allows_default("openings"):
            for r in state.rooms:
                if r.window_count is None or r.door_count is None:
                    return True

        # If relationships are not defaulted, check if relationships are empty for multi-room building
        if not state.allows_default("relationships") and len(state.rooms) > 1 and not state.relationships:
            return True

        return False

    def decide_next_action(
        self,
        state: RequirementState,
        check: CheckResult,
        user_said_use_defaults: bool = False,
    ) -> Tuple[SessionStatus, str, List[str]]:
        """
        Decide the next session action based on the current state and check.

        Parameters
        ----------
        state : RequirementState
        check : CheckResult
            Result of the latest completeness check.
        user_said_use_defaults : bool
            True if the user's latest reply requested defaults.

        Returns
        -------
        (status, message, pending_fields)
        """
        # ---- Unresolvable conflicts: surface them, cannot finalize ----
        if check.conflict_fields:
            message = self._format_conflict_message(check.conflict_fields)
            return SessionStatus.CONFLICT, message, check.conflict_fields

        # ---- Preference not yet chosen: prompt for preference ----
        if state.defaults_preference is None and not user_said_use_defaults:
            return (
                SessionStatus.NEEDS_CLARIFICATION,
                DEFAULTS_PREFERENCE_PROMPT,
                ["defaults_preference"],
            )

        if user_said_use_defaults:
            state.set_defaults_preference("use_defaults")

        # ---- Option 1: USE_DEFAULTS ----
        if state.defaults_preference == "use_defaults":
            applied = self.apply_all_eligible_defaults(state)
            state.normalize_rooms()
            state.recalculate_warnings()
            check = self._checker.check(state)
            if not check.missing_fields and not check.conflict_fields:
                return (
                    SessionStatus.AWAITING_CONFIRMATION,
                    self.build_confirmation_summary(state),
                    [],
                )
            # Mandatory fields still missing (building_type, floor_count, rooms)
            message = (
                "I've applied configured defaults where possible. "
                "However, the following fields still need your input:\n"
                + self._format_missing_fields(check.missing_fields)
            )
            if state.clarification_round < MAX_ROUNDS:
                return SessionStatus.NEEDS_CLARIFICATION, message, check.missing_fields
            return SessionStatus.CONFLICT, message + "\n\n⚠ Max clarification rounds reached.", check.missing_fields

        # ---- Option 3: HYBRID ----
        if state.defaults_preference == "hybrid":
            applied = self.apply_all_eligible_defaults(state)
            state.normalize_rooms()
            state.recalculate_warnings()
            check = self._checker.check(state)
            has_manual_pending = self._has_unspecified_manual_details(state)

            if not check.missing_fields and not check.conflict_fields and not has_manual_pending:
                return (
                    SessionStatus.AWAITING_CONFIRMATION,
                    self.build_confirmation_summary(state),
                    [],
                )
            if state.clarification_round >= MAX_ROUNDS:
                return (
                    SessionStatus.AWAITING_CONFIRMATION,
                    self.build_confirmation_summary(state),
                    [],
                )
            questions = self.generate_questions(check, state)
            round_num = state.clarification_round + 1
            header = f"[Round {round_num}/{MAX_ROUNDS}] "
            return SessionStatus.NEEDS_CLARIFICATION, header + questions, check.missing_fields

        # ---- Option 2: SPECIFY_DETAILS ----
        # Do not automatically apply ordinary optional defaults!
        has_manual_pending = self._has_unspecified_manual_details(state)
        if check.complete and not has_manual_pending:
            state.normalize_rooms()
            state.recalculate_warnings()
            return (
                SessionStatus.AWAITING_CONFIRMATION,
                self.build_confirmation_summary(state),
                [],
            )

        # After round 3: apply defaults to remaining eligible fields as fallback
        if state.clarification_round >= MAX_ROUNDS:
            applied = self.apply_all_eligible_defaults(state)
            state.normalize_rooms()
            state.recalculate_warnings()
            check = self._checker.check(state)
            if not check.missing_fields and not check.conflict_fields:
                note = ""
                if applied:
                    note = "\n\n" + format_defaults_summary(applied)
                return (
                    SessionStatus.AWAITING_CONFIRMATION,
                    self.build_confirmation_summary(state) + note,
                    [],
                )
            conflict_msg = (
                "⚠ Maximum clarification rounds reached. "
                "The following required fields have no safe defaults and must be provided:\n"
                + self._format_missing_fields(check.missing_fields)
            )
            return SessionStatus.CONFLICT, conflict_msg, check.missing_fields

        # Still have clarification rounds left: ask targeted questions
        questions = self.generate_questions(check, state)
        round_num = state.clarification_round + 1
        header = f"[Round {round_num}/{MAX_ROUNDS}] "
        return SessionStatus.NEEDS_CLARIFICATION, header + questions, check.missing_fields

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def generate_questions(
        self, check: CheckResult, state: RequirementState
    ) -> str:
        """
        Build a compact, focused question string from the missing fields
        and unspecified manual categories.
        """
        parts: List[str] = []

        missing = check.missing_fields

        if "building_type" in missing:
            parts.append(
                "• What type of building is this? "
                "(e.g. residential, commercial, mixed)"
            )
        if "floor_count" in missing:
            parts.append(
                "• How many floors does the building have? "
                "(Note: the current prototype supports single-floor buildings only.)"
            )
        if "rooms" in missing:
            parts.append(
                "• What rooms do you need? Please list types and counts, "
                "e.g. '2 bedrooms, 1 bathroom, 1 hall'."
            )
        if "units" in missing:
            parts.append(
                "• What unit system should I use? (e.g. m, cm, mm, ft, in). "
                "Canonical storage will be normalized to metres (m)."
            )

        # Room openings question (if not permitted to default or explicitly requested)
        if not state.allows_default("openings"):
            rooms_missing_openings: List[str] = [
                r.room_id or r.room_type for r in state.rooms
                if r.window_count is None or r.door_count is None
            ]
            if rooms_missing_openings:
                room_list = ", ".join(rooms_missing_openings)
                parts.append(
                    f"• How many windows and doors should each of these rooms have: "
                    f"{room_list}?"
                )

        # Room dimensions question (if not permitted to default or explicitly requested)
        if not state.allows_default("dimensions"):
            rooms_missing_dims: List[str] = [
                r.room_id or r.room_type for r in state.rooms
                if r.width_m is None or r.length_m is None
            ]
            if rooms_missing_dims:
                room_list = ", ".join(rooms_missing_dims)
                parts.append(
                    f"• What dimensions (width and length) should these rooms have: "
                    f"{room_list}?"
                )

        # Room relationships question (if not permitted to default and none exist)
        if not state.allows_default("relationships") and len(state.rooms) > 1 and not state.relationships:
            parts.append(
                "• How should the rooms connect to or relate to each other? "
                "(e.g. 'the hall connects to both bedrooms and the bathroom')"
            )

        if not parts:
            parts.append(
                "• Please provide more details about the unspecified fields: "
                + ", ".join(missing) + "."
            )

        intro = "I need a few more details to complete the requirements:"
        return intro + "\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # Summary building
    # ------------------------------------------------------------------

    def build_confirmation_summary(self, state: RequirementState) -> str:
        """
        Build a human-readable final summary disclosing:
          1. User-supplied values (and original user units)
          2. Applied system defaults (dimensions, relationships, wall thickness, floor height)
          3. Overall footprint status (pending layout generation)
          4. Warnings / assumptions
        """
        user_lines: List[str] = []
        default_lines: List[str] = []

        # --- Scalar fields ---
        field_labels = {
            "building_type": "Building type",
            "floor_count": "Floor count",
        }
        for key, label in field_labels.items():
            entry = state.checklist[key]
            if entry.value is None:
                continue
            line = f"  {label}: {entry.value}"
            if entry.source in (FieldSource.USER_EXPLICIT, FieldSource.USER_CONFIRMED_DEFAULT):
                user_lines.append(line)
            else:
                default_lines.append(line)

        # Units display
        units_entry = state.checklist.get("units")
        if units_entry and units_entry.value:
            if state.user_supplied_units and state.user_supplied_units != "m":
                user_lines.append(f"  Units: m (canonical storage; user supplied: {state.user_supplied_units})")
            elif units_entry.source == FieldSource.USER_EXPLICIT:
                user_lines.append(f"  Units: {units_entry.value}")
            else:
                default_lines.append(f"  Units: {units_entry.value} (canonical storage, system default)")

        # Overall Dimensions
        default_lines.append("  Overall building footprint: width and length pending layout generation")
        default_lines.append("  Wall thickness: 0.2 m (system default)")
        default_lines.append("  Floor height: 3.0 m (system default)")

        # --- Rooms ---
        for room in state.rooms:
            r_id = room.room_id or room.room_type
            type_str = f" ({room.room_type})" if room.room_id and room.room_id != room.room_type else ""
            room_line = f"  • {r_id}{type_str}"

            details = []
            if room.width_m is not None and room.length_m is not None:
                w_src = state.get_room_field_source(r_id, "width_m")
                l_src = state.get_room_field_source(r_id, "length_m")
                if w_src == FieldSource.SYSTEM_DEFAULT or l_src == FieldSource.SYSTEM_DEFAULT:
                    details.append(f"{room.width_m} m × {room.length_m} m (defaulted)")
                    default_lines.append(f"  {r_id} dimensions: {room.width_m} m × {room.length_m} m (prototype default)")
                else:
                    details.append(f"{room.width_m} m × {room.length_m} m")
            elif room.width_m is not None:
                details.append(f"width: {room.width_m} m")
            elif room.length_m is not None:
                details.append(f"length: {room.length_m} m")

            if room.window_count is not None:
                w_src = state.get_room_field_source(r_id, "window_count")
                if w_src == FieldSource.SYSTEM_DEFAULT:
                    details.append(f"{room.window_count} window(s) (defaulted)")
                else:
                    details.append(f"{room.window_count} window(s)")
            if room.door_count is not None:
                d_src = state.get_room_field_source(r_id, "door_count")
                if d_src == FieldSource.SYSTEM_DEFAULT:
                    details.append(f"{room.door_count} door(s) (defaulted)")
                else:
                    details.append(f"{room.door_count} door(s)")

            if room.notes:
                details.append(f"notes: {room.notes}")
            if details:
                room_line += " (" + ", ".join(details) + ")"
            user_lines.append(room_line)

        # --- Relationships ---
        if state.relationships:
            for rel in state.relationships:
                rel_desc = rel.relationship.value
                if getattr(rel, "negative", False):
                    rel_desc = f"must NOT {rel_desc}"
                is_default = (getattr(rel, "source", None) == FieldSource.SYSTEM_DEFAULT)
                if is_default:
                    default_lines.append(f"  Assumed relationship: {rel.room_a} {rel_desc} {rel.room_b} (prototype default)")
                else:
                    user_lines.append(f"  {rel.room_a} {rel_desc} {rel.room_b}")

        # --- Constraints ---
        if state.constraints:
            for c in state.constraints:
                user_lines.append(f"  Constraint: {c}")

        # --- Preferences ---
        if state.preferences:
            for p in state.preferences:
                user_lines.append(f"  Preference: {p}")

        # --- Assemble ---
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  REQUIREMENTS SUMMARY")
        lines.append("=" * 60)

        if user_lines:
            lines.append("\nUser-supplied requirements:")
            lines.extend(user_lines)
        else:
            lines.append("\n(No user-supplied requirements recorded.)")

        if default_lines:
            lines.append("\nSystem defaults applied / Assumptions:")
            lines.extend(default_lines)

        if state.warnings:
            lines.append("\nWarnings / assumptions:")
            for w in state.warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("\n" + "=" * 60)
        lines.append("Please review the above and reply 'confirm' to finalize,")
        lines.append("or describe any changes you'd like to make.")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_missing_fields(fields: List[str]) -> str:
        return "\n".join(f"  • {f}" for f in fields)

    @staticmethod
    def _format_conflict_message(fields: List[str]) -> str:
        lines = [
            "⚠ The following fields have conflicting values that must be resolved "
            "before I can finalize the requirements:"
        ]
        for f in fields:
            lines.append(f"  • {f}")
        lines.append(
            "\nPlease clarify the correct value for each conflict. "
            "System defaults cannot be used to resolve explicit contradictions."
        )
        return "\n".join(lines)
