"""
validators/requirements_validator.py
======================================
Deterministic validation of the full requirement state before finalization.

Responsibilities
----------------
* Type and range checks (counts, dimensions, units).
* Relationship reference validation (rooms referenced must exist).
* Allowed relationship vocabulary enforcement.
* Contradiction detection (e.g. a room marked adjacent to itself).
* Final pre-finalization gate: raises ``ValidationError`` if any blocking
  issue remains.

Design note
-----------
Pydantic handles schema-level validation; this validator applies
domain-level rules that depend on the interaction between fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from schemas import FieldStatus, RelationshipType
from requirements.requirement_state import RequirementState


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Aggregated validation outcome."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # noqa: D105
        return self.valid


class ValidationError(Exception):
    """Raised when the state fails final validation."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("Validation failed: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class RequirementsValidator:
    """
    Validates a ``RequirementState`` before it is packaged into an
    ``ExtractionResult``.

    Call ``validate(state)`` to get a ``ValidationResult``.
    Call ``validate_or_raise(state)`` for a hard gate before finalization.
    """

    # Allowed unit strings (case-insensitive)
    ALLOWED_UNITS = {"mm", "cm", "m", "ft", "in"}

    # Max reasonable values for prototype scope
    MAX_ROOMS_OF_ONE_TYPE = 20
    MAX_TOTAL_ROOMS = 50
    MAX_WINDOWS_PER_ROOM = 20
    MAX_DOORS_PER_ROOM = 10
    MAX_DIMENSION_M = 500.0  # 500 m footprint is already unreasonably large

    def validate(self, state: RequirementState) -> ValidationResult:
        """Run all validation rules and return a ``ValidationResult``."""
        errors: List[str] = []
        warnings: List[str] = []

        # ----- 1. Scalar fields -----
        self._validate_building_type(state, errors)
        self._validate_floor_count(state, errors)
        self._validate_units(state, errors, warnings)

        # ----- 2. Rooms -----
        self._validate_rooms(state, errors, warnings)

        # ----- 3. Relationships -----
        self._validate_relationships(state, errors)

        # ----- 4. Constraints (basic check) -----
        # Constraints are free-form strings; no deep validation in v1.

        # ----- 5. No unresolved conflicts -----
        if state.conflicts:
            for c in state.conflicts:
                errors.append(f"Unresolved conflict: {c}")

        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)

    def validate_or_raise(self, state: RequirementState) -> None:
        """Validate and raise ``ValidationError`` if there are any errors."""
        result = self.validate(state)
        if not result.valid:
            raise ValidationError(result.errors)

    # ------------------------------------------------------------------
    # Private validation sub-methods
    # ------------------------------------------------------------------

    def _validate_building_type(
        self, state: RequirementState, errors: List[str]
    ) -> None:
        entry = state.checklist.get("building_type")
        if not entry or entry.status == FieldStatus.MISSING:
            errors.append("building_type is required but not provided.")
        elif entry.status == FieldStatus.CONFLICT:
            errors.append(f"building_type has a conflict: {entry.notes}")

    def _validate_floor_count(
        self, state: RequirementState, errors: List[str]
    ) -> None:
        entry = state.checklist.get("floor_count")
        if not entry or entry.status == FieldStatus.MISSING:
            errors.append("floor_count is required but not provided.")
            return
        if entry.status == FieldStatus.CONFLICT:
            errors.append(f"floor_count has a conflict: {entry.notes}")
            return
        val = entry.value
        if not isinstance(val, int) or val < 1:
            errors.append(f"floor_count must be a positive integer; got {val!r}.")
        elif val != 1:
            errors.append(
                f"floor_count={val} is not supported. "
                "This prototype supports single-floor buildings only."
            )

    def _validate_units(
        self,
        state: RequirementState,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        entry = state.checklist.get("units")
        if not entry or entry.status == FieldStatus.MISSING:
            # units can be defaulted; this is not blocking if defaulted
            warnings.append("units not specified; a default will be applied.")
            return
        val = entry.value
        if isinstance(val, str) and val.lower() not in self.ALLOWED_UNITS:
            errors.append(
                f"Unit '{val}' is not supported. "
                f"Allowed: {', '.join(sorted(self.ALLOWED_UNITS))}."
            )

    def _validate_rooms(
        self,
        state: RequirementState,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        if not state.rooms:
            errors.append("At least one room must be specified.")
            return

        seen_ids: set[str] = set()
        seen_types: dict[str, int] = {}
        total_count = 0

        for room in state.rooms:
            # Validate stable room ID
            if not room.room_id:
                errors.append(f"Room '{room.room_type}' is missing a stable room_id.")
            elif room.room_id.lower() in seen_ids:
                errors.append(f"Duplicate room ID '{room.room_id}' found in room list.")
            else:
                seen_ids.add(room.room_id.lower())

            # Validate room type
            if not room.room_type or not room.room_type.strip():
                errors.append("Room has an empty or invalid room_type.")
            rt = room.room_type.strip().lower()
            seen_types[rt] = seen_types.get(rt, 0) + 1

            # Count range - each canonical physical room must have count=1
            if room.count < 1:
                errors.append(
                    f"Room '{room.room_id or room.room_type}' has count={room.count}; "
                    "must be at least 1."
                )
            elif room.count != 1:
                errors.append(
                    f"Room '{room.room_id or room.room_type}' has count={room.count}; "
                    "canonical physical room objects must have count=1."
                )

            total_count += room.count

            # Window/door counts
            if room.window_count is not None:
                if room.window_count < 0:
                    errors.append(
                        f"Room '{room.room_id or room.room_type}': window_count cannot be negative."
                    )
                elif room.window_count > self.MAX_WINDOWS_PER_ROOM:
                    errors.append(
                        f"Room '{room.room_id or room.room_type}': window_count={room.window_count} "
                        f"exceeds maximum of {self.MAX_WINDOWS_PER_ROOM}."
                    )

            if room.door_count is not None:
                if room.door_count < 0:
                    errors.append(
                        f"Room '{room.room_id or room.room_type}': door_count cannot be negative."
                    )
                elif room.door_count > self.MAX_DOORS_PER_ROOM:
                    errors.append(
                        f"Room '{room.room_id or room.room_type}': door_count={room.door_count} "
                        f"exceeds maximum of {self.MAX_DOORS_PER_ROOM}."
                    )

            # Dimension sanity
            for dim_name, dim_val in [("width_m", room.width_m), ("length_m", room.length_m)]:
                if dim_val is not None:
                    if dim_val <= 0:
                        errors.append(
                            f"Room '{room.room_id or room.room_type}': {dim_name} must be positive."
                        )
                    elif dim_val > self.MAX_DIMENSION_M:
                        errors.append(
                            f"Room '{room.room_id or room.room_type}': {dim_name}={dim_val} m is unreasonably large."
                        )

        if total_count > self.MAX_TOTAL_ROOMS:
            errors.append(
                f"Total room count {total_count} exceeds maximum of {self.MAX_TOTAL_ROOMS}."
            )

    def _validate_relationships(
        self, state: RequirementState, errors: List[str]
    ) -> None:
        valid_room_ids = {r.room_id.lower() for r in state.rooms if r.room_id}

        for rel in state.relationships:
            a = rel.room_a.lower()
            b = rel.room_b.lower()

            # Self-relationship
            if a == b:
                errors.append(
                    f"Relationship '{rel.relationship.value}' is self-referential: "
                    f"'{rel.room_a}' cannot be related to itself."
                )

            # Referenced room IDs must exist
            if a not in valid_room_ids:
                errors.append(
                    f"Relationship references room ID '{rel.room_a}' which does not exist in the room list."
                )
            if b not in valid_room_ids:
                errors.append(
                    f"Relationship references room ID '{rel.room_b}' which does not exist in the room list."
                )
