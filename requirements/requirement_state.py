"""
requirements/requirement_state.py
==================================
Session state for the Requirements Extraction Agent.

``RequirementState`` is the single mutable object that tracks:
  - Session identity and clarification-round count
  - Per-field checklist entries (value, status, provenance)
  - Accumulated rooms, relationships, constraints, and preferences
  - Pending questions for the user
  - Conversation history snippet (for targeted follow-up LLM calls)

Design principles
-----------------
* Python owns state; the LLM never resets or replaces it.
* Updating a field with a new valid value preserves prior values for other fields.
* ``merge_candidate`` is the only public method that writes to field values, and
  it refuses to overwrite a USER_EXPLICIT value with an inferred or defaulted one
  unless a conflict is deliberately flagged.
* The state can be serialised to / from a plain dict for optional JSON persistence.
"""

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from schemas import (
    BuildingType,
    CandidateRequirements,
    Dimensions,
    FieldEntry,
    FieldSource,
    FieldStatus,
    RoomRelationship,
    RoomSpec,
    SessionStatus,
)


# ---------------------------------------------------------------------------
# RequirementState
# ---------------------------------------------------------------------------


class RequirementState:
    """
    Mutable session state for one extraction session.

    Attributes
    ----------
    session_id : str
        UUID4 string identifying this session.
    clarification_round : int
        Number of completed clarification rounds (0 = initial extraction done,
        no clarification yet).
    checklist : Dict[str, FieldEntry]
        Top-level scalar fields tracked individually.
    rooms : List[RoomSpec]
        Room-level requirements (keyed by room_type internally).
    relationships : List[RoomRelationship]
        Spatial / functional relationships between rooms.
    constraints : List[str]
        Explicit must / must-not statements.
    preferences : List[str]
        Non-blocking aesthetic or functional preferences.
    pending_questions : List[str]
        Questions queued for the current clarification turn.
    warnings : List[str]
        Non-blocking concerns about the current state.
    conflicts : List[str]
        Blocking contradictions that prevent finalization.
    created_at : str
        ISO-8601 timestamp.
    confirmed_by_user : bool
        Set to True after the user explicitly confirms the final summary.
    """

    # Fields managed individually in ``checklist``
    TRACKED_FIELDS = ("building_type", "floor_count", "units")

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.clarification_round: int = 0
        self.confirmed_by_user: bool = False
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.status: SessionStatus = SessionStatus.NEEDS_CLARIFICATION

        # Scalar checklist fields
        self.checklist: Dict[str, FieldEntry] = {
            "building_type": FieldEntry(blocking=True),
            "floor_count": FieldEntry(blocking=True),
            "units": FieldEntry(blocking=False),  # can always be defaulted
        }

        # Structured sub-collections
        self.rooms: List[RoomSpec] = []
        self.relationships: List[RoomRelationship] = []
        self.constraints: List[str] = []
        self.preferences: List[str] = []

        # Dialogue metadata
        self.pending_questions: List[str] = []
        self.ambiguities: List[str] = []
        self.warnings: List[str] = []
        self.conflicts: List[str] = []

        # Optional: keep a small snippet for follow-up LLM calls
        self._initial_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def merge_candidate(
        self,
        candidate: CandidateRequirements,
        source: FieldSource = FieldSource.USER_EXPLICIT,
    ) -> None:
        """
        Merge a ``CandidateRequirements`` object into the current state.

        Rules
        -----
        * ``None`` in the candidate means "not mentioned" — it never clears
          an existing value.
        * A USER_EXPLICIT value is never overwritten by INFERRED or SYSTEM_DEFAULT.
        * If a new USER_EXPLICIT value conflicts with an existing USER_EXPLICIT
          value, mark the field as CONFLICT rather than silently overwriting.
        * Rooms are merged by ``room_type``; a new count for the same type
          from the user overwrites the old one.
        """
        # --- Scalar fields ---
        if candidate.building_type is not None:
            self._update_scalar(
                "building_type", candidate.building_type.value, source
            )

        if candidate.floor_count is not None:
            self._update_scalar("floor_count", candidate.floor_count, source)

        if candidate.units is not None:
            self._update_scalar("units", candidate.units, source)

        # --- Rooms ---
        if candidate.rooms:
            self._merge_rooms(candidate.rooms, source)

        # --- Relationships ---
        if candidate.relationships:
            self._merge_relationships(candidate.relationships)

        # --- Constraints ---
        if candidate.constraints:
            for c in candidate.constraints:
                if c not in self.constraints:
                    self.constraints.append(c)

        # --- Preferences ---
        if candidate.preferences:
            for p in candidate.preferences:
                if p not in self.preferences:
                    self.preferences.append(p)

        # --- LLM ambiguity notes ---
        if candidate.ambiguities:
            for note in candidate.ambiguities:
                if note not in self.ambiguities:
                    self.ambiguities.append(note)

        self.recalculate_warnings()

    def _update_scalar(
        self, field: str, new_value: Any, source: FieldSource
    ) -> None:
        """Update a scalar checklist field following the provenance rules."""
        entry = self.checklist[field]
        current_status = entry.status

        # If we have no value yet, simply set it
        if current_status == FieldStatus.MISSING:
            entry.value = new_value
            entry.status = FieldStatus.PROVIDED
            entry.source = source
            return

        # If the incoming source is SYSTEM_DEFAULT or INFERRED, never overwrite
        # a user-supplied or already-confirmed value
        if source in (FieldSource.SYSTEM_DEFAULT, FieldSource.INFERRED):
            if entry.source in (FieldSource.USER_EXPLICIT, FieldSource.USER_CONFIRMED_DEFAULT):
                return  # user value wins; silently skip
            # Overwrite a previous default/inferred with new default/inferred
            entry.value = new_value
            entry.status = FieldStatus.DEFAULTED
            entry.source = source
            return

        # Both old and new are USER_EXPLICIT — check for conflict
        if (
            source == FieldSource.USER_EXPLICIT
            and entry.source == FieldSource.USER_EXPLICIT
            and entry.value != new_value
        ):
            conflict_msg = (
                f"Conflicting values for '{field}': "
                f"previously '{entry.value}', now '{new_value}'."
            )
            entry.status = FieldStatus.CONFLICT
            entry.notes = conflict_msg
            if conflict_msg not in self.conflicts:
                self.conflicts.append(conflict_msg)
            return

        # Same or compatible value — update normally
        entry.value = new_value
        entry.status = FieldStatus.PROVIDED
        entry.source = source

    def _merge_rooms(
        self, new_rooms: List[RoomSpec], source: FieldSource
    ) -> None:
        """
        Merge new room specs into existing canonical state without appending duplicates.
        """
        if not self.rooms:
            self.rooms = canonicalize_rooms(new_rooms)
            return

        # Ensure existing rooms are canonical
        self.rooms = canonicalize_rooms(self.rooms)

        for new_room in new_rooms:
            raw_name = new_room.room_id or new_room.room_type
            base, idx = parse_room_name(raw_name)
            base = base or "room"

            matched_rooms: List[RoomSpec] = []
            if idx is not None:
                target_id = f"{base}_{idx}"
                matched_rooms = [r for r in self.rooms if r.room_id == target_id]
            else:
                existing_of_type = [r for r in self.rooms if r.room_type == base]
                if existing_of_type:
                    matched_rooms = existing_of_type

            if matched_rooms:
                for target in matched_rooms:
                    if new_room.window_count is not None:
                        target.window_count = new_room.window_count
                    if new_room.door_count is not None:
                        target.door_count = new_room.door_count
                    if new_room.width_m is not None:
                        target.width_m = new_room.width_m
                    if new_room.length_m is not None:
                        target.length_m = new_room.length_m
                    if new_room.notes:
                        target.notes = new_room.notes
            else:
                if new_room.count > 1 and idx is None:
                    for i in range(1, new_room.count + 1):
                        self.rooms.append(
                            RoomSpec(
                                room_id=f"{base}_{i}",
                                room_type=base,
                                count=1,
                                window_count=new_room.window_count,
                                door_count=new_room.door_count,
                                width_m=new_room.width_m,
                                length_m=new_room.length_m,
                                notes=new_room.notes,
                            )
                        )
                else:
                    self.rooms.append(
                        RoomSpec(
                            room_id=f"{base}_{idx or 1}",
                            room_type=base,
                            count=1,
                            window_count=new_room.window_count,
                            door_count=new_room.door_count,
                            width_m=new_room.width_m,
                            length_m=new_room.length_m,
                            notes=new_room.notes,
                        )
                    )

        self.rooms = canonicalize_rooms(self.rooms)

    def _merge_relationships(
        self, new_rels: List[RoomRelationship]
    ) -> None:
        """Merge relationships and normalize room endpoints to canonical room IDs."""
        self.normalize_rooms()
        valid_ids = {r.room_id: r.room_id for r in self.rooms if r.room_id}

        type_counts: Dict[str, int] = {}
        for r in self.rooms:
            type_counts[r.room_type] = type_counts.get(r.room_type, 0) + 1
        unique_type_to_id = {
            r.room_type: r.room_id for r in self.rooms if type_counts.get(r.room_type) == 1
        }

        existing = {(r.room_a, r.room_b, r.relationship) for r in self.relationships}

        for rel in new_rels:
            a_norm = normalize_room_id(rel.room_a)
            b_norm = normalize_room_id(rel.room_b)

            norm_a = a_norm if a_norm in valid_ids else unique_type_to_id.get(rel.room_a.lower(), a_norm)
            norm_b = b_norm if b_norm in valid_ids else unique_type_to_id.get(rel.room_b.lower(), b_norm)

            resolved_rel = RoomRelationship(
                room_a=norm_a,
                room_b=norm_b,
                relationship=rel.relationship,
                notes=rel.notes,
            )
            key = (resolved_rel.room_a, resolved_rel.room_b, resolved_rel.relationship)
            if key not in existing:
                self.relationships.append(resolved_rel)
                existing.add(key)

    # ------------------------------------------------------------------
    # Default application
    # ------------------------------------------------------------------

    def apply_default(
        self,
        field: str,
        value: Any,
        unit: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Apply a system default to a scalar checklist field.

        Only applied when the field is currently MISSING.  Never overwrites
        USER_EXPLICIT or CONFLICT fields.
        """
        if field not in self.checklist:
            raise KeyError(f"Unknown checklist field: '{field}'")
        entry = self.checklist[field]
        if entry.status == FieldStatus.MISSING:
            entry.value = value
            entry.unit = unit
            entry.status = FieldStatus.DEFAULTED
            entry.source = FieldSource.SYSTEM_DEFAULT
            if notes:
                entry.notes = notes

    def apply_room_opening_defaults(
        self,
        room_type: str,
        window_count: Optional[int],
        door_count: Optional[int],
        notes: Optional[str] = None,
    ) -> None:
        """Apply opening defaults to all rooms matching room_type that lack them."""
        clean_target = room_type.strip().lower().replace("-", "_").replace(" ", "_")
        target_base, _ = parse_room_name(clean_target)
        for room in self.rooms:
            r_base, _ = parse_room_name(room.room_id or room.room_type)
            matches = (
                room.room_id == room_type
                or room.room_type == room_type
                or r_base == clean_target
                or r_base == target_base
            )
            if matches:
                if room.window_count is None and window_count is not None:
                    object.__setattr__(room, "window_count", window_count)
                if room.door_count is None and door_count is not None:
                    object.__setattr__(room, "door_count", door_count)

    def check_user_reply_resolutions(self, reply: str) -> None:
        """
        Check user reply for explicit resolutions of ambiguities.
        E.g. if the user says 'hall has 4 doors in total' or confirms 4 doors,
        resolve the hall doors ambiguity.
        """
        clean = reply.strip().lower()

        # Hall doors ambiguity
        if ("hall" in clean and any(d in clean for d in ("4", "four", "total", "doors"))) or ("4 doors" in clean):
            self.resolve_ambiguity(r"hall.*door")
            for r in self.rooms:
                if r.room_type == "hall":
                    r.door_count = 4

        # Generic rooms ambiguity
        if "generic" in clean or ("room" in clean and any(w in clean for w in ("bedroom", "living", "function", "type"))):
            self.resolve_ambiguity(r"generic\s+rooms?.*function|room.*type")

    def resolve_ambiguity(self, pattern: str) -> None:
        """Resolve any ambiguity matching the regex pattern and remove from warnings."""
        regex = re.compile(pattern, re.IGNORECASE)
        self.ambiguities = [a for a in self.ambiguities if not regex.search(a)]
        self.warnings = [w for w in self.warnings if not regex.search(w)]

    def recalculate_warnings(self) -> List[str]:
        """
        Recalculate warnings reflecting the current state only.
        Removes obsolete warnings and retains only active ambiguities.
        """
        warnings: List[str] = []

        # 1. Active ambiguities (excluding any whose subject is already resolved)
        for amb in self.ambiguities:
            if re.search(r"hall.*door", amb, re.IGNORECASE):
                hall = next((r for r in self.rooms if r.room_type == "hall"), None)
                if hall and hall.door_count is not None and hall.door_count > 1:
                    continue  # Resolved!
            if re.search(r"generic\s+rooms?.*function|room.*type", amb, re.IGNORECASE):
                if self.clarification_round > 0 or all(r.room_type != "room" for r in self.rooms):
                    continue
            if amb not in warnings:
                warnings.append(amb)

        # 2. Room opening warnings for rooms that still lack door/window counts
        for room in self.rooms:
            r_id = room.room_id or room.room_type
            if room.window_count is None:
                msg = f"Window count for '{r_id}' is not specified (will use default or remain unset)."
                if msg not in warnings:
                    warnings.append(msg)
            if room.door_count is None:
                msg = f"Door count for '{r_id}' is not specified (will use default or remain unset)."
                if msg not in warnings:
                    warnings.append(msg)

        # 3. Units warning if missing
        units_entry = self.checklist.get("units")
        if units_entry and units_entry.status == FieldStatus.MISSING:
            msg = "units not specified; a default will be applied."
            if msg not in warnings:
                warnings.append(msg)

        self.warnings = warnings
        return warnings

    def normalize_rooms(self) -> None:
        """Normalize all rooms into canonical individual rooms and normalize relationship endpoints."""
        self.rooms = canonicalize_rooms(self.rooms)
        valid_ids = {r.room_id: r.room_id for r in self.rooms if r.room_id}
        type_counts: Dict[str, int] = {}
        for r in self.rooms:
            type_counts[r.room_type] = type_counts.get(r.room_type, 0) + 1
        unique_type_to_id = {
            r.room_type: r.room_id for r in self.rooms if type_counts.get(r.room_type) == 1
        }

        normalized_rels = []
        existing = set()
        for rel in self.relationships:
            a_norm = normalize_room_id(rel.room_a)
            b_norm = normalize_room_id(rel.room_b)
            norm_a = a_norm if a_norm in valid_ids else unique_type_to_id.get(rel.room_a.lower(), a_norm)
            norm_b = b_norm if b_norm in valid_ids else unique_type_to_id.get(rel.room_b.lower(), b_norm)
            resolved = RoomRelationship(
                room_a=norm_a,
                room_b=norm_b,
                relationship=rel.relationship,
                notes=rel.notes,
            )
            key = (resolved.room_a, resolved.room_b, resolved.relationship)
            if key not in existing:
                normalized_rels.append(resolved)
                existing.add(key)
        self.relationships = normalized_rels
        self.recalculate_warnings()

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def confirm(self) -> None:
        """Mark the session as confirmed by the user."""
        self.confirmed_by_user = True
        for entry in self.checklist.values():
            if entry.status in (FieldStatus.PROVIDED, FieldStatus.DEFAULTED):
                entry.user_confirmed = True

    def increment_round(self) -> None:
        """Increment the clarification round counter."""
        self.clarification_round += 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the state."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "clarification_round": self.clarification_round,
            "confirmed_by_user": self.confirmed_by_user,
            "created_at": self.created_at,
            "checklist": {
                k: v.model_dump() for k, v in self.checklist.items()
            },
            "rooms": [r.model_dump() for r in self.rooms],
            "relationships": [r.model_dump() for r in self.relationships],
            "constraints": self.constraints,
            "preferences": self.preferences,
            "pending_questions": self.pending_questions,
            "ambiguities": self.ambiguities,
            "warnings": self.warnings,
            "conflicts": self.conflicts,
        }

    def to_json(self) -> str:
        """Return a JSON string of the state."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementState":
        """Reconstruct a ``RequirementState`` from a serialised dict."""
        state = cls(session_id=data["session_id"])
        state.clarification_round = data["clarification_round"]
        state.confirmed_by_user = data.get("confirmed_by_user", False)
        state.created_at = data.get("created_at", state.created_at)
        if "status" in data:
            try:
                state.status = SessionStatus(data["status"])
            except ValueError:
                pass

        for key, entry_data in data.get("checklist", {}).items():
            state.checklist[key] = FieldEntry(**entry_data)

        state.rooms = [RoomSpec(**r) for r in data.get("rooms", [])]
        state.normalize_rooms()
        state.relationships = [
            RoomRelationship(**r) for r in data.get("relationships", [])
        ]
        state.constraints = data.get("constraints", [])
        state.preferences = data.get("preferences", [])
        state.pending_questions = data.get("pending_questions", [])
        state.ambiguities = data.get("ambiguities", [])
        state.warnings = data.get("warnings", [])
        state.conflicts = data.get("conflicts", [])
        state.recalculate_warnings()
        return state

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_scalar(self, field: str) -> Optional[Any]:
        """Return the current value of a scalar checklist field."""
        return self.checklist[field].value

    def has_rooms(self) -> bool:
        return len(self.rooms) > 0

    def room_count_total(self) -> int:
        return sum(r.count for r in self.rooms)

    def __repr__(self) -> str:
        return (
            f"RequirementState(session_id={self.session_id!r}, "
            f"round={self.clarification_round}, "
            f"rooms={[r.room_id or r.room_type for r in self.rooms]}, "
            f"conflicts={self.conflicts})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_room_name(name: str) -> tuple[str, Optional[int]]:
    """
    Parse a room name or identifier like 'room 1', 'room_2', 'bathroom', 'hall 1'.
    Returns (base_room_type, index_or_None).
    """
    if not name:
        return ("", None)
    clean = name.strip()
    m = re.match(r"^([a-zA-Z\s_-]+?)(?:[\s_#-]+(\d+))?$", clean)
    if m:
        base = m.group(1).strip().lower().replace("-", "_").replace(" ", "_")
        idx = int(m.group(2)) if m.group(2) else None
        return (base, idx)
    return (clean.lower().replace("-", "_").replace(" ", "_"), None)


def normalize_room_id(name: str) -> str:
    """
    Return a canonical room ID string like 'room_1', 'bathroom_1', 'hall_1'.
    """
    base, idx = parse_room_name(name)
    if idx is not None:
        return f"{base}_{idx}"
    return base


def canonicalize_rooms(rooms: List[RoomSpec]) -> List[RoomSpec]:
    """
    Normalize and deduplicate room entities into one canonical object per physical room.

    - Grouped rooms with count > 1 are expanded into individual rooms with stable IDs
      (e.g. room_1, room_2) and count=1.
    - If individual rooms already exist for a room type (e.g. room 1, room 2), any grouped
      room entry for that type is merged into them (preserving properties) and dropped to avoid
      duplicating physical spaces.
    - Numbered rooms (e.g. bathroom 1, hall 1) map to canonical IDs (bathroom_1, hall_1).
    - Preserves all known properties: window_count (including 0), door_count, dimensions, notes.
    - Ensures every room has count=1, a valid stable room_id, and room_type.
    """
    if not rooms:
        return []

    by_type_individuals: Dict[str, Dict[int, RoomSpec]] = {}
    by_type_grouped: Dict[str, List[RoomSpec]] = {}
    by_type_unindexed: Dict[str, List[RoomSpec]] = {}

    order: List[str] = []

    for r in rooms:
        raw_name = r.room_id or r.room_type
        base, idx = parse_room_name(raw_name)
        base = base or "room"
        if base not in order:
            order.append(base)

        if idx is not None:
            if base not in by_type_individuals:
                by_type_individuals[base] = {}
            if idx in by_type_individuals[base]:
                existing = by_type_individuals[base][idx]
                by_type_individuals[base][idx] = _merge_room_specs(existing, r)
            else:
                r_copy = r.model_copy()
                r_copy.room_id = f"{base}_{idx}"
                r_copy.room_type = base
                r_copy.count = 1
                by_type_individuals[base][idx] = r_copy
        elif r.count > 1:
            if base not in by_type_grouped:
                by_type_grouped[base] = []
            by_type_grouped[base].append(r)
        else:
            if base not in by_type_unindexed:
                by_type_unindexed[base] = []
            by_type_unindexed[base].append(r)

    canonical: List[RoomSpec] = []

    for base in order:
        individuals = by_type_individuals.get(base, {})
        grouped_list = by_type_grouped.get(base, [])
        unindexed_list = by_type_unindexed.get(base, [])

        # Transfer grouped properties to individual rooms if present, or expand grouped room
        for grp in grouped_list:
            if individuals:
                for idx, ind in individuals.items():
                    if ind.window_count is None and grp.window_count is not None:
                        ind.window_count = grp.window_count
                    if ind.door_count is None and grp.door_count is not None:
                        ind.door_count = grp.door_count
                    if ind.width_m is None and grp.width_m is not None:
                        ind.width_m = grp.width_m
                    if ind.length_m is None and grp.length_m is not None:
                        ind.length_m = grp.length_m
                    if not ind.notes and grp.notes:
                        ind.notes = grp.notes
            else:
                for i in range(1, grp.count + 1):
                    individuals[i] = RoomSpec(
                        room_id=f"{base}_{i}",
                        room_type=base,
                        count=1,
                        window_count=grp.window_count,
                        door_count=grp.door_count,
                        width_m=grp.width_m,
                        length_m=grp.length_m,
                        notes=grp.notes,
                    )

        # Merge or add unindexed single rooms
        for unind in unindexed_list:
            if 1 in individuals and len(individuals) == 1:
                ind = individuals[1]
                if unind.window_count is not None:
                    ind.window_count = unind.window_count
                if unind.door_count is not None:
                    ind.door_count = unind.door_count
                if unind.width_m is not None:
                    ind.width_m = unind.width_m
                if unind.length_m is not None:
                    ind.length_m = unind.length_m
                if unind.notes:
                    ind.notes = unind.notes
            elif not individuals:
                individuals[1] = RoomSpec(
                    room_id=f"{base}_1",
                    room_type=base,
                    count=1,
                    window_count=unind.window_count,
                    door_count=unind.door_count,
                    width_m=unind.width_m,
                    length_m=unind.length_m,
                    notes=unind.notes,
                )
            else:
                for ind in individuals.values():
                    if unind.window_count is not None:
                        ind.window_count = unind.window_count
                    if unind.door_count is not None:
                        ind.door_count = unind.door_count
                    if unind.width_m is not None:
                        ind.width_m = unind.width_m
                    if unind.length_m is not None:
                        ind.length_m = unind.length_m
                    if unind.notes:
                        ind.notes = unind.notes

        for idx in sorted(individuals.keys()):
            canonical.append(individuals[idx])

    return canonical


def _merge_room_specs(existing: RoomSpec, new: RoomSpec) -> RoomSpec:
    """
    Merge two RoomSpec objects for the same room.

    New explicit values win; None in the new spec preserves existing value.
    Zero counts (0) are valid explicit values and are never treated as missing.
    """
    return RoomSpec(
        room_id=new.room_id or existing.room_id,
        room_type=existing.room_type,
        count=1,
        window_count=new.window_count if new.window_count is not None else existing.window_count,
        door_count=new.door_count if new.door_count is not None else existing.door_count,
        width_m=new.width_m if new.width_m is not None else existing.width_m,
        length_m=new.length_m if new.length_m is not None else existing.length_m,
        notes=new.notes or existing.notes,
    )
