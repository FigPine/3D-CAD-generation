"""
requirements/completeness_checker.py
=====================================
Deterministic completeness and consistency checker.

The checker owns all decisions about whether the current state is ready to
finalize.  It never calls Gemini; all rules are explicit Python logic.

Returned ``CheckResult`` contains:
  - ``complete``          : True only if all blocking fields are filled and
                            no unresolved conflicts remain.
  - ``missing_fields``    : Blocking fields with status MISSING.
  - ``conflict_fields``   : Fields in CONFLICT status.
  - ``warnings``          : Non-blocking issues.
  - ``pending_defaults``  : Fields that are MISSING but have a registered
                            default (eligible for auto-fill after round 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from schemas import FieldStatus
from requirements.requirement_state import RequirementState
from requirements.defaults import DEFAULTS, NO_DEFAULT_FIELDS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a completeness and consistency check."""

    complete: bool
    missing_fields: List[str] = field(default_factory=list)
    conflict_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pending_defaults: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # noqa: D105
        return self.complete


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class CompletenessChecker:
    """
    Evaluates a ``RequirementState`` against the defined completion policy.

    The completion policy:
    1. All blocking scalar fields must have status != MISSING and != CONFLICT.
    2. At least one room must be present.
    3. No blocking conflicts may be unresolved (conflicts list is empty).
    4. Room counts must be positive integers (enforced by schema, but
       double-checked here as a safety net).
    5. The ``floor_count`` must be 1 (prototype scope).
    """

    # Scalar checklist fields that MUST be present before finalization
    REQUIRED_SCALARS = ["building_type", "floor_count", "units"]

    def check(self, state: RequirementState) -> CheckResult:
        """
        Run all completeness rules and return a ``CheckResult``.

        Does NOT mutate ``state``.
        """
        missing: List[str] = []
        conflicts: List[str] = []
        state_warnings = state.recalculate_warnings()
        warnings: List[str] = list(state_warnings)
        pending_defaults: List[str] = []

        # ----- 1. Scalar fields -----
        for field_name in self.REQUIRED_SCALARS:
            entry = state.checklist.get(field_name)
            if entry is None:
                missing.append(field_name)
                continue

            if entry.status == FieldStatus.MISSING:
                missing.append(field_name)
                # Is there a default available?
                if field_name in DEFAULTS and field_name not in NO_DEFAULT_FIELDS:
                    pending_defaults.append(field_name)

            elif entry.status == FieldStatus.CONFLICT:
                conflicts.append(field_name)

        # ----- 2. Rooms -----
        if not state.rooms:
            missing.append("rooms")
        else:
            for room in state.rooms:
                if room.count < 1:
                    warnings.append(
                        f"Room '{room.room_id or room.room_type}' has count={room.count}; "
                        "must be at least 1."
                    )

        # ----- 3. Room dimensions and openings (when manual specification is required) -----
        if state.defaults_preference is not None:
            if not state.allows_default("dimensions") and state.rooms:
                missing_dims = [
                    r.room_id or r.room_type for r in state.rooms
                    if r.width_m is None or r.length_m is None
                ]
                if missing_dims:
                    missing.append(f"room_dimensions: {', '.join(missing_dims)}")

            if not state.allows_default("openings") and state.rooms:
                missing_openings = [
                    r.room_id or r.room_type for r in state.rooms
                    if r.window_count is None or r.door_count is None
                ]
                if missing_openings:
                    missing.append(f"room_openings: {', '.join(missing_openings)}")


        # ----- 4. Unresolved conflicts from state -----
        for conflict_msg in state.conflicts:
            if conflict_msg not in [c for c in conflicts]:
                # Scalar conflicts already captured above; add free-form ones
                conflicts.append(conflict_msg)

        # ----- 5. Floor count prototype constraint -----
        fc_entry = state.checklist.get("floor_count")
        if fc_entry and fc_entry.value is not None and fc_entry.value != 1:
            conflicts.append(
                f"floor_count={fc_entry.value} is not supported. "
                "This prototype supports single-floor buildings only."
            )

        # ----- 6. Domain Validation (relationship integrity, room limits, counts) -----
        from validators.requirements_validator import RequirementsValidator
        val_result = RequirementsValidator().validate(state)
        for err in val_result.errors:
            if not any(m in err for m in missing) and err not in conflicts:
                conflicts.append(err)

        complete = not missing and not conflicts

        return CheckResult(
            complete=complete,
            missing_fields=missing,
            conflict_fields=conflicts,
            warnings=warnings,
            pending_defaults=pending_defaults,
        )

    def blocking_missing(self, state: RequirementState) -> List[str]:
        """Convenience: return only the blocking missing field names."""
        return self.check(state).missing_fields

    def has_conflicts(self, state: RequirementState) -> bool:
        """Return True if there are any unresolved conflicts."""
        return bool(self.check(state).conflict_fields)
