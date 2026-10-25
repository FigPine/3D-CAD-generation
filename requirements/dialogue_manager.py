"""
requirements/dialogue_manager.py
=================================
Manages the clarification dialogue: round counting, question generation,
default-application decisions, and summary building.

Responsibilities (all deterministic — no LLM calls here)
---------------------------------------------------------
* Track and enforce the 3-round maximum.
* Generate focused, template-based clarification questions from missing fields.
* Detect "use defaults" intent from user replies (simple keyword matching).
* Apply eligible defaults after round 3 (or immediately on "use defaults").
* Build the human-readable final summary for user confirmation.

The agent orchestrator calls this manager to decide what happens after each
state update, keeping the LLM call decision logic outside the manager.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from schemas import FieldStatus, SessionStatus
from requirements.requirement_state import RequirementState
from requirements.completeness_checker import CheckResult, CompletenessChecker
from requirements.defaults import (
    DEFAULTS,
    NO_DEFAULT_FIELDS,
    format_defaults_summary,
    get_room_opening_defaults,
)


MAX_ROUNDS: int = 3

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
        Apply all system defaults to MISSING eligible fields in ``state``.

        Returns a list of default keys that were applied (for provenance).
        Never applies defaults to NO_DEFAULT_FIELDS or CONFLICT fields.
        """
        applied: List[str] = []

        # --- Scalar defaults ---
        for field_name, entry in state.checklist.items():
            if entry.status == FieldStatus.MISSING and field_name not in NO_DEFAULT_FIELDS:
                default = DEFAULTS.get(field_name)
                if default is not None:
                    state.apply_default(
                        field_name,
                        default.value,
                        unit=default.unit,
                        notes=default.rationale,
                    )
                    applied.append(field_name)

        # --- Room-level opening defaults ---
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

        return applied

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

        # ---- "Use defaults" requested: apply immediately ----
        if user_said_use_defaults:
            applied = self.apply_all_eligible_defaults(state)
            state.normalize_rooms()
            state.recalculate_warnings()
            check = self._checker.check(state)  # recheck after defaults
            if not check.missing_fields and not check.conflict_fields:
                return (
                    SessionStatus.AWAITING_CONFIRMATION,
                    self.build_confirmation_summary(state),
                    [],
                )
            # Some fields still missing (those without safe defaults)
            message = (
                "I've applied system defaults where possible. "
                "However, the following fields still need your input:\n"
                + self._format_missing_fields(check.missing_fields)
            )
            if state.clarification_round < MAX_ROUNDS:
                return SessionStatus.NEEDS_CLARIFICATION, message, check.missing_fields
            # Already at max rounds — block finalization
            return SessionStatus.CONFLICT, message + "\n\n⚠ Max clarification rounds reached.", check.missing_fields

        # ---- Complete: normalize and move to confirmation ----
        if check.complete:
            state.normalize_rooms()
            self.apply_all_eligible_defaults(state)
            state.recalculate_warnings()
            return (
                SessionStatus.AWAITING_CONFIRMATION,
                self.build_confirmation_summary(state),
                [],
            )

        # ---- After round 3: apply defaults to remaining eligible fields ----
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
            # Still blocking fields with no default — surface them
            conflict_msg = (
                "⚠ Maximum clarification rounds reached. "
                "The following required fields have no safe defaults and must be provided:\n"
                + self._format_missing_fields(check.missing_fields)
            )
            return SessionStatus.CONFLICT, conflict_msg, check.missing_fields

        # ---- Still have rounds left: ask clarification questions ----
        questions = self.generate_questions(check, state)
        round_num = state.clarification_round + 1  # upcoming round number
        header = f"[Round {round_num}/{MAX_ROUNDS}] "
        message = header + questions
        return SessionStatus.NEEDS_CLARIFICATION, message, check.missing_fields

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def generate_questions(
        self, check: CheckResult, state: RequirementState
    ) -> str:
        """
        Build a compact, focused question string from the missing fields.

        Questions are grouped so the user can answer them in one reply.
        """
        parts: List[str] = []

        missing = check.missing_fields
        if not missing:
            return ""

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
                "• What unit system should I use? (e.g. mm, m, ft) "
                "I can use millimetres (mm) by default."
            )

        # Room-specific: openings not yet provided
        rooms_missing_openings: List[str] = []
        for room in state.rooms:
            if room.window_count is None or room.door_count is None:
                rooms_missing_openings.append(room.room_type)

        if rooms_missing_openings:
            room_list = ", ".join(rooms_missing_openings)
            parts.append(
                f"• How many windows and doors should each of these rooms have: "
                f"{room_list}? (Or say 'use defaults' to apply standard values.)"
            )

        if not parts:
            # Generic fallback — should not normally be reached
            parts.append(
                "• Please provide more details about: "
                + ", ".join(missing) + "."
            )

        intro = "I need a few more details to complete the requirements:"
        return intro + "\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # Summary building
    # ------------------------------------------------------------------

    def build_confirmation_summary(self, state: RequirementState) -> str:
        """
        Build a human-readable final summary grouped into:
          1. User-supplied values
          2. System defaults
          3. Warnings / assumptions
        """
        user_lines: List[str] = []
        default_lines: List[str] = []

        from schemas import FieldSource  # avoid circular at module level

        # --- Scalar fields ---
        field_labels = {
            "building_type": "Building type",
            "floor_count": "Floor count",
            "units": "Units",
        }
        for key, label in field_labels.items():
            entry = state.checklist[key]
            if entry.value is None:
                continue
            line = f"  {label}: {entry.value}"
            if entry.unit:
                line += f" {entry.unit}"
            if entry.source in (FieldSource.USER_EXPLICIT, FieldSource.USER_CONFIRMED_DEFAULT):
                user_lines.append(line)
            else:
                default_lines.append(line)

        # --- Rooms ---
        for room in state.rooms:
            label = room.room_id or f"{room.count}× {room.room_type}"
            type_str = f" ({room.room_type})" if room.room_id and room.room_id != room.room_type else ""
            room_line = f"  • {label}{type_str}"
            details = []
            if room.window_count is not None:
                details.append(f"{room.window_count} window(s)")
            if room.door_count is not None:
                details.append(f"{room.door_count} door(s)")
            if room.notes:
                details.append(f"notes: {room.notes}")
            if details:
                room_line += " (" + ", ".join(details) + ")"
            user_lines.append(room_line)

        # --- Relationships ---
        if state.relationships:
            for rel in state.relationships:
                user_lines.append(
                    f"  {rel.room_a} {rel.relationship.value} {rel.room_b}"
                )

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
            lines.append("\nSystem defaults applied:")
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
