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
    RelationshipType,
    RoomRelationship,
    RoomSpec,
    SessionStatus,
)
from requirements.unit_normalizer import canonicalize_unit_name, normalize_to_metres


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

        # Scalar checklist fields (canonical units is always 'm')
        self.checklist: Dict[str, FieldEntry] = {
            "building_type": FieldEntry(blocking=True),
            "floor_count": FieldEntry(blocking=True),
            "units": FieldEntry(blocking=False),  # can always be defaulted to 'm'
        }

        # Defaults preference state (None = not yet asked/selected)
        # Options: "use_defaults", "specify_details", "hybrid"
        self.defaults_preference: Optional[str] = None
        self.hybrid_default_categories: List[str] = []
        self.user_supplied_units: Optional[str] = None

        # Overall building footprint dimensions (explicit from user or None pending layout)
        self.overall_dimensions: Optional[Dimensions] = None
        self.has_user_explicit_footprint: bool = False

        # Structured sub-collections
        self.rooms: List[RoomSpec] = []
        self.relationships: List[RoomRelationship] = []
        self._unresolved_relationships: List[Dict[str, Any]] = []
        self.constraints: List[str] = []
        self.preferences: List[str] = []

        # Provenance tracking for individual room fields: {room_id: {field: FieldSource}}
        self.room_field_sources: Dict[str, Dict[str, FieldSource]] = {}

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

        # --- Units and canonical unit handling ---
        if candidate.units is not None:
            raw_u = candidate.units.strip()
            canon_u = canonicalize_unit_name(raw_u)
            if canon_u:
                self.user_supplied_units = canon_u
                self._update_scalar("units", "m", source)
                self.checklist["units"].notes = f"Canonical unit: metres (user specified: {raw_u})"
            else:
                amb_msg = f"Unrecognized unit system '{raw_u}'. Supported units: m, cm, mm, ft, in."
                if amb_msg not in self.ambiguities:
                    self.ambiguities.append(amb_msg)

        # --- Overall dimensions ---
        # Never infer overall footprint (width_m, length_m) from room dimensions or LLM candidate
        # unless explicitly supplied by the user. Keep footprint unset (None) pending layout.
        if candidate.overall_dimensions:
            if self.overall_dimensions is None:
                self.overall_dimensions = Dimensions(
                    width_m=candidate.overall_dimensions.width_m if self.has_user_explicit_footprint else None,
                    length_m=candidate.overall_dimensions.length_m if self.has_user_explicit_footprint else None,
                    floor_height_m=candidate.overall_dimensions.floor_height_m or 3.0,
                    wall_thickness_m=candidate.overall_dimensions.wall_thickness_m or 0.2,
                )
            else:
                if self.has_user_explicit_footprint:
                    if candidate.overall_dimensions.width_m is not None:
                        self.overall_dimensions.width_m = candidate.overall_dimensions.width_m
                    if candidate.overall_dimensions.length_m is not None:
                        self.overall_dimensions.length_m = candidate.overall_dimensions.length_m
                if candidate.overall_dimensions.floor_height_m is not None:
                    self.overall_dimensions.floor_height_m = candidate.overall_dimensions.floor_height_m
                if candidate.overall_dimensions.wall_thickness_m is not None:
                    self.overall_dimensions.wall_thickness_m = candidate.overall_dimensions.wall_thickness_m

        # --- Rooms ---
        if candidate.rooms:
            self._merge_rooms(candidate.rooms, source)

        # --- Relationships ---
        if candidate.relationships:
            self._merge_relationships(candidate.relationships)

        # --- Constraints ---
        if candidate.constraints:
            for c in candidate.constraints:
                # Extract structured relationships if described in constraint text
                extracted_rels = _extract_relationships_from_text(c, self)
                if extracted_rels:
                    self._merge_relationships(extracted_rels)
                if c not in self.constraints:
                    self.constraints.append(c)

        # --- Preferences ---
        if candidate.preferences:
            known_modes = {"use_defaults", "specify_details", "hybrid"}
            for p in candidate.preferences:
                p_norm = p.strip().lower().replace(" ", "_")
                if p_norm in known_modes:
                    if self.defaults_preference is None:
                        self.set_defaults_preference(p_norm)
                    elif p_norm not in self.preferences:
                        self.preferences.append(p_norm)
                elif p not in self.preferences:
                    self.preferences.append(p)

        # --- LLM ambiguity notes ---
        if candidate.ambiguities:
            for note in candidate.ambiguities:
                if note not in self.ambiguities:
                    self.ambiguities.append(note)

        self.recalculate_warnings()

    def _extract_relationships_from_text(self, text: str) -> List[RoomRelationship]:
        """Extract structured RoomRelationship objects from natural language text using this state."""
        return _extract_relationships_from_text(text, self)

    def resolve_room_reference(self, name: str) -> List[str]:
        """Resolve a free-form or structured room reference to canonical room ID(s) in this state."""
        return resolve_room_reference(name, self)

    def allows_default(self, category: str) -> bool:
        """
        Return True if system defaults are permitted for the given category.

        Categories:
          - 'dimensions' (room dimensions, wall thickness, floor height)
          - 'openings' (doors, windows)
          - 'relationships' (connectivity defaults)
          - 'units' (measurement unit)
        """
        if self.defaults_preference == "use_defaults":
            # Standard defaults cover dimensions, openings, and units.
            # Choosing default dimensions or opening counts must NOT implicitly authorize default room connections!
            if category.strip().lower() in ("relationship", "relationships", "connection", "connections", "room_connections"):
                return False
            return True
        if self.defaults_preference == "specify_details":
            return False
        if self.defaults_preference == "hybrid":
            cat_norm = category.strip().lower()
            dim_synonyms = {"dimension", "dimensions", "room_dimensions", "size", "sizes"}
            rel_synonyms = {"relationship", "relationships", "connection", "connections", "room_connections"}
            op_synonyms = {"opening", "openings", "doors", "windows", "doors_and_windows"}
            unit_synonyms = {"unit", "units"}

            for allowed in self.hybrid_default_categories:
                a_clean = allowed.strip().lower()
                if cat_norm in dim_synonyms and a_clean in dim_synonyms:
                    return True
                if cat_norm in rel_synonyms and a_clean in rel_synonyms:
                    return True
                if cat_norm in op_synonyms and a_clean in op_synonyms:
                    return True
                if cat_norm in unit_synonyms and a_clean in unit_synonyms:
                    return True
                if cat_norm == a_clean:
                    return True
            return False
        return False

    def set_defaults_preference(
        self,
        preference: str,
        categories: Optional[List[str]] = None,
    ) -> None:
        """Set and persist the user's defaults preference throughout the session."""
        self.defaults_preference = preference
        if categories is not None:
            self.hybrid_default_categories = list(categories)

        # Retain the selected mode consistently in preferences without redundant entries
        known_modes = {"use_defaults", "specify_details", "hybrid"}
        self.preferences = [p for p in self.preferences if p not in known_modes]
        if preference in known_modes:
            self.preferences.append(preference)

    def set_room_field_source(
        self, room_id: str, field_name: str, source: FieldSource
    ) -> None:
        """Record provenance for a room-level field."""
        if room_id not in self.room_field_sources:
            self.room_field_sources[room_id] = {}
        self.room_field_sources[room_id][field_name] = source

    def get_room_field_source(
        self, room_id: str, field_name: str
    ) -> Optional[FieldSource]:
        """Return provenance for a room-level field."""
        return self.room_field_sources.get(room_id, {}).get(field_name)

    def _normalize_dimension(self, val: Optional[float]) -> Optional[float]:
        """Convert a dimension value to canonical metres if user supplied non-metre units."""
        if val is None:
            return None
        if self.user_supplied_units and self.user_supplied_units != "m":
            try:
                return normalize_to_metres(val, self.user_supplied_units)
            except ValueError:
                pass
        return round(val, 4)

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

        # Both old and new are USER_EXPLICIT — update to new value and clear prior conflicts.
        # This allows user corrections/updates during clarification or review without getting trapped in CONFLICT.
        self.conflicts = [
            c for c in self.conflicts if not (f"'{field}'" in c or f"{field}=" in c)
        ]
        entry.value = new_value
        entry.status = FieldStatus.PROVIDED
        entry.source = source
        entry.notes = None

    def _merge_rooms(
        self, new_rooms: List[RoomSpec], source: FieldSource
    ) -> None:
        """
        Merge new room specs into existing canonical state without appending duplicates.
        Mutates existing RoomSpec instances in-place to preserve references.
        """
        if not self.rooms:
            self.rooms = canonicalize_rooms(new_rooms)
            for r in self.rooms:
                r_id = r.room_id or r.room_type
                if r.width_m is not None:
                    r.width_m = self._normalize_dimension(r.width_m)
                    self.set_room_field_source(r_id, "width_m", source)
                if r.length_m is not None:
                    r.length_m = self._normalize_dimension(r.length_m)
                    self.set_room_field_source(r_id, "length_m", source)
                if r.window_count is not None:
                    self.set_room_field_source(r_id, "window_count", source)
                if r.door_count is not None:
                    self.set_room_field_source(r_id, "door_count", source)
            return

        unindexed_matched_counts: Dict[str, int] = {}
        newly_added: List[RoomSpec] = []

        for new_room in new_rooms:
            raw_name = new_room.room_id or new_room.room_type
            base, idx = parse_room_name(raw_name)
            base = base or "room"

            matched_rooms: List[RoomSpec] = []
            if new_room.room_id:
                matched_rooms = [r for r in self.rooms if r.room_id and r.room_id.lower() == new_room.room_id.lower()]

            if not matched_rooms and idx is not None:
                target_id = f"{base}_{idx}"
                matched_rooms = [r for r in self.rooms if r.room_id and r.room_id.lower() == target_id.lower()]

            if not matched_rooms:
                existing_of_type = [r for r in self.rooms if r.room_type == base or parse_room_name(r.room_id or "")[0] == base]
                if len(existing_of_type) == 1:
                    matched_rooms = existing_of_type
                elif len(existing_of_type) > 1:
                    is_collective = (
                        raw_name.lower().endswith("s")
                        or new_room.count == len(existing_of_type)
                        or (new_room.notes and "both" in new_room.notes.lower())
                    )
                    if is_collective:
                        matched_rooms = existing_of_type
                    else:
                        start_idx = unindexed_matched_counts.get(base, 0)
                        if start_idx < len(existing_of_type):
                            matched_rooms = [existing_of_type[start_idx]]
                            unindexed_matched_counts[base] = start_idx + 1
                        else:
                            matched_rooms = existing_of_type

            if matched_rooms:
                for target in matched_rooms:
                    t_id = target.room_id or target.room_type
                    if new_room.window_count is not None:
                        target.window_count = new_room.window_count
                        self.set_room_field_source(t_id, "window_count", source)
                    if new_room.door_count is not None:
                        target.door_count = new_room.door_count
                        self.set_room_field_source(t_id, "door_count", source)
                    if new_room.width_m is not None:
                        target.width_m = self._normalize_dimension(new_room.width_m)
                        self.set_room_field_source(t_id, "width_m", source)
                    if new_room.length_m is not None:
                        target.length_m = self._normalize_dimension(new_room.length_m)
                        self.set_room_field_source(t_id, "length_m", source)
                    if new_room.notes:
                        target.notes = f"{target.notes}; {new_room.notes}" if target.notes and target.notes != new_room.notes else new_room.notes
            else:
                w_norm = self._normalize_dimension(new_room.width_m)
                l_norm = self._normalize_dimension(new_room.length_m)

                # Determine next available index for this room type
                existing_type_indices = []
                for r in self.rooms:
                    b, i = parse_room_name(r.room_id or r.room_type)
                    if b == base and i is not None:
                        existing_type_indices.append(i)
                max_idx = max(existing_type_indices) if existing_type_indices else 0

                # When rooms are already established, do not invent new rooms beyond established inventory
                if self.rooms and existing_type_indices:
                    if idx is not None and idx > max_idx:
                        err_msg = (
                            f"Referenced room '{raw_name}' does not exist in the established inventory "
                            f"({len(existing_type_indices)} {base} established). Please clarify."
                        )
                        if err_msg not in self.ambiguities:
                            self.ambiguities.append(err_msg)
                        continue
                    if idx is None:
                        target = next((r for r in self.rooms if (r.room_type == base or parse_room_name(r.room_id or "")[0] == base)), None)
                        if target:
                            t_id = target.room_id or target.room_type
                            if new_room.window_count is not None:
                                target.window_count = new_room.window_count
                                self.set_room_field_source(t_id, "window_count", source)
                            if new_room.door_count is not None:
                                target.door_count = new_room.door_count
                                self.set_room_field_source(t_id, "door_count", source)
                            if new_room.width_m is not None:
                                target.width_m = w_norm
                                self.set_room_field_source(t_id, "width_m", source)
                            if new_room.length_m is not None:
                                target.length_m = l_norm
                                self.set_room_field_source(t_id, "length_m", source)
                            continue

                if new_room.count > 1 and idx is None:
                    for i in range(1, new_room.count + 1):
                        r_id = f"{base}_{max_idx + i}"
                        added_r = RoomSpec(
                            room_id=r_id,
                            room_type=base,
                            count=1,
                            window_count=new_room.window_count,
                            door_count=new_room.door_count,
                            width_m=w_norm,
                            length_m=l_norm,
                            notes=new_room.notes,
                        )
                        self.rooms.append(added_r)
                        newly_added.append(added_r)
                        if new_room.window_count is not None:
                            self.set_room_field_source(r_id, "window_count", source)
                        if new_room.door_count is not None:
                            self.set_room_field_source(r_id, "door_count", source)
                        if w_norm is not None:
                            self.set_room_field_source(r_id, "width_m", source)
                        if l_norm is not None:
                            self.set_room_field_source(r_id, "length_m", source)
                else:
                    allocated_idx = idx if (idx is not None and idx > max_idx) else (max_idx + 1)
                    r_id = f"{base}_{allocated_idx}"
                    added_r = RoomSpec(
                        room_id=r_id,
                        room_type=base,
                        count=1,
                        window_count=new_room.window_count,
                        door_count=new_room.door_count,
                        width_m=w_norm,
                        length_m=l_norm,
                        notes=new_room.notes,
                    )
                    self.rooms.append(added_r)
                    newly_added.append(added_r)
                    if new_room.window_count is not None:
                        self.set_room_field_source(r_id, "window_count", source)
                    if new_room.door_count is not None:
                        self.set_room_field_source(r_id, "door_count", source)
                    if w_norm is not None:
                        self.set_room_field_source(r_id, "width_m", source)
                    if l_norm is not None:
                        self.set_room_field_source(r_id, "length_m", source)

        self.rooms = canonicalize_rooms(self.rooms)

    def _merge_relationships(
        self,
        new_rels: List[RoomRelationship],
        source: Optional[FieldSource] = None,
    ) -> None:
        """Merge relationships and normalize room endpoints to canonical room IDs."""
        self.normalize_rooms()
        valid_ids = {r.room_id.lower(): r.room_id for r in self.rooms if r.room_id}

        type_counts: Dict[str, int] = {}
        for r in self.rooms:
            type_counts[r.room_type] = type_counts.get(r.room_type, 0) + 1

        existing = set()
        for r in self.relationships:
            a_low = r.room_a.lower()
            b_low = r.room_b.lower()
            is_neg = getattr(r, "negative", False)
            pair_key = (min(a_low, b_low), max(a_low, b_low), r.relationship, is_neg)
            existing.add(pair_key)

        for rel in new_rels:
            a_raw = rel.room_a.strip()
            b_raw = rel.room_b.strip()
            is_neg = getattr(rel, "negative", False)
            rel_src = getattr(rel, "source", None) or source or FieldSource.USER_EXPLICIT

            base_a, idx_a = parse_room_name(a_raw)
            base_b, idx_b = parse_room_name(b_raw)

            # Case 1: Symmetric collective relationship between identical room types (e.g. "bedrooms are adjacent", "bedroom to bedroom")
            if base_a == base_b and idx_a is None and idx_b is None:
                rooms_of_type = [r.room_id for r in self.rooms if r.room_id and (r.room_type == base_a or parse_room_name(r.room_id or "")[0] == base_a)]
                if len(rooms_of_type) == 2:
                    resolved_rel = RoomRelationship(
                        room_a=rooms_of_type[0],
                        room_b=rooms_of_type[1],
                        relationship=rel.relationship,
                        notes=rel.notes,
                        negative=is_neg,
                        source=rel_src,
                    )
                    key = (min(rooms_of_type[0].lower(), rooms_of_type[1].lower()), max(rooms_of_type[0].lower(), rooms_of_type[1].lower()), resolved_rel.relationship, is_neg)
                    opp_key = (min(rooms_of_type[0].lower(), rooms_of_type[1].lower()), max(rooms_of_type[0].lower(), rooms_of_type[1].lower()), resolved_rel.relationship, not is_neg)
                    if opp_key in existing:
                        opp_rel = next((r for r in self.relationships if (min(r.room_a.lower(), r.room_b.lower()), max(r.room_a.lower(), r.room_b.lower()), r.relationship, getattr(r, "negative", False)) == opp_key), None)
                        if opp_rel is not None and (rel_src == FieldSource.USER_EXPLICIT or getattr(opp_rel, "source", None) == FieldSource.SYSTEM_DEFAULT):
                            self.relationships = [r for r in self.relationships if r is not opp_rel]
                            existing.discard(opp_key)
                    if key not in existing:
                        self.relationships.append(resolved_rel)
                        existing.add(key)
                    continue

            # Case 2: Resolve endpoints using resolve_room_reference
            res_a = resolve_room_reference(a_raw, self)
            res_b = resolve_room_reference(b_raw, self)

            # Check if one or both endpoints are ambiguous
            if not res_a:
                if type_counts.get(base_a, 0) > 1 and idx_a is None:
                    amb_msg = (
                        f"Ambiguous room reference '{a_raw}' in relationship: "
                        f"there are {type_counts[base_a]} {base_a}s. Please clarify which room."
                    )
                    if amb_msg not in self.ambiguities:
                        self.ambiguities.append(amb_msg)
                    self._unresolved_relationships.append({
                        "rel": rel,
                        "ambiguous_endpoint": "a",
                        "ambiguous_raw": a_raw,
                        "base": base_a,
                        "other_endpoint_raw": b_raw,
                    })
                    continue
                else:
                    amb_msg = f"Relationship references room '{a_raw}' which does not exist in the room list."
                    if amb_msg not in self.ambiguities:
                        self.ambiguities.append(amb_msg)
                    continue

            if not res_b:
                if type_counts.get(base_b, 0) > 1 and idx_b is None:
                    amb_msg = (
                        f"Ambiguous room reference '{b_raw}' in relationship: "
                        f"there are {type_counts[base_b]} {base_b}s. Please clarify which room."
                    )
                    if amb_msg not in self.ambiguities:
                        self.ambiguities.append(amb_msg)
                    self._unresolved_relationships.append({
                        "rel": rel,
                        "ambiguous_endpoint": "b",
                        "ambiguous_raw": b_raw,
                        "base": base_b,
                        "other_endpoint_raw": a_raw,
                    })
                    continue
                else:
                    amb_msg = f"Relationship references room '{b_raw}' which does not exist in the room list."
                    if amb_msg not in self.ambiguities:
                        self.ambiguities.append(amb_msg)
                    continue

            # Both endpoints resolved (could be collective with multiple IDs, e.g. hall connects to both bedrooms)
            for norm_a in res_a:
                for norm_b in res_b:
                    if norm_a.lower() == norm_b.lower():
                        continue  # avoid self-reference
                    resolved_rel = RoomRelationship(
                        room_a=norm_a,
                        room_b=norm_b,
                        relationship=rel.relationship,
                        notes=rel.notes,
                        negative=is_neg,
                        source=rel_src,
                    )
                    key = (min(norm_a.lower(), norm_b.lower()), max(norm_a.lower(), norm_b.lower()), resolved_rel.relationship, is_neg)
                    opp_key = (min(norm_a.lower(), norm_b.lower()), max(norm_a.lower(), norm_b.lower()), resolved_rel.relationship, not is_neg)
                    if opp_key in existing:
                        opp_rel = next((r for r in self.relationships if (min(r.room_a.lower(), r.room_b.lower()), max(r.room_a.lower(), r.room_b.lower()), r.relationship, getattr(r, "negative", False)) == opp_key), None)
                        if opp_rel is not None and (rel_src == FieldSource.USER_EXPLICIT or getattr(opp_rel, "source", None) == FieldSource.SYSTEM_DEFAULT):
                            self.relationships = [r for r in self.relationships if r is not opp_rel]
                            existing.discard(opp_key)
                    if key not in existing:
                        self.relationships.append(resolved_rel)
                        existing.add(key)
                    else:
                        # Upgrade source to USER_EXPLICIT if existing was SYSTEM_DEFAULT
                        for ex_rel in self.relationships:
                            ex_a = ex_rel.room_a.lower()
                            ex_b = ex_rel.room_b.lower()
                            ex_neg = getattr(ex_rel, "negative", False)
                            ex_key = (min(ex_a, ex_b), max(ex_a, ex_b), ex_rel.relationship, ex_neg)
                            if ex_key == key:
                                if getattr(ex_rel, "source", None) == FieldSource.SYSTEM_DEFAULT and rel_src == FieldSource.USER_EXPLICIT:
                                    ex_rel.source = FieldSource.USER_EXPLICIT

        self.synchronize_constraints_with_relationships()

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
        target_base, target_idx = parse_room_name(clean_target)
        for room in self.rooms:
            r_id = (room.room_id or room.room_type or "").strip().lower()
            r_base, r_idx = parse_room_name(r_id)
            if target_idx is not None:
                matches = (r_id == clean_target) or (r_base == target_base and r_idx == target_idx)
            else:
                matches = (
                    (r_id == clean_target)
                    or (room.room_type.strip().lower() == clean_target)
                    or (r_base == target_base)
                )
            if matches:
                r_id = room.room_id or room.room_type
                if room.window_count is None and window_count is not None:
                    if self.get_room_field_source(r_id, "window_count") != FieldSource.USER_EXPLICIT:
                        object.__setattr__(room, "window_count", window_count)
                        self.set_room_field_source(r_id, "window_count", FieldSource.SYSTEM_DEFAULT)
                if room.door_count is None and door_count is not None:
                    if self.get_room_field_source(r_id, "door_count") != FieldSource.USER_EXPLICIT:
                        object.__setattr__(room, "door_count", door_count)
                        self.set_room_field_source(r_id, "door_count", FieldSource.SYSTEM_DEFAULT)

    def apply_room_dimension_defaults(
        self,
        room_type: str,
        width_m: Optional[float],
        length_m: Optional[float],
        notes: Optional[str] = None,
    ) -> List[str]:
        """
        Apply dimension defaults to all rooms matching room_type that lack them.

        Returns list of applied default field names.
        Preserves explicit user dimensions and only applies if allowed by preference.
        """
        applied: List[str] = []
        clean_target = room_type.strip().lower().replace("-", "_").replace(" ", "_")
        target_base, target_idx = parse_room_name(clean_target)
        for room in self.rooms:
            r_id = (room.room_id or room.room_type or "").strip().lower()
            r_base, r_idx = parse_room_name(r_id)
            if target_idx is not None:
                matches = (r_id == clean_target) or (r_base == target_base and r_idx == target_idx)
            else:
                matches = (
                    (r_id == clean_target)
                    or (room.room_type.strip().lower() == clean_target)
                    or (r_base == target_base)
                )
            if matches:
                r_id = room.room_id or room.room_type
                if room.width_m is None and width_m is not None:
                    if self.get_room_field_source(r_id, "width_m") != FieldSource.USER_EXPLICIT:
                        room.width_m = width_m
                        applied.append(f"{r_id}_width_m")
                        self.set_room_field_source(r_id, "width_m", FieldSource.SYSTEM_DEFAULT)
                if room.length_m is None and length_m is not None:
                    if self.get_room_field_source(r_id, "length_m") != FieldSource.USER_EXPLICIT:
                        room.length_m = length_m
                        applied.append(f"{r_id}_length_m")
                        self.set_room_field_source(r_id, "length_m", FieldSource.SYSTEM_DEFAULT)
        return applied

    def apply_relationship_defaults(
        self, new_rels: List[RoomRelationship]
    ) -> List[str]:
        """
        Apply defaulted relationships to state.

        Returns list of applied relationship identifiers.
        Preserves explicit positive and negative constraints.
        """
        applied: List[str] = []
        existing = {
            (min(r.room_a.lower(), r.room_b.lower()), max(r.room_a.lower(), r.room_b.lower()), r.relationship, getattr(r, "negative", False))
            for r in self.relationships
        }
        for rel in new_rels:
            is_neg = getattr(rel, "negative", False)
            key = (min(rel.room_a.lower(), rel.room_b.lower()), max(rel.room_a.lower(), rel.room_b.lower()), rel.relationship, is_neg)
            opp_key = (min(rel.room_a.lower(), rel.room_b.lower()), max(rel.room_a.lower(), rel.room_b.lower()), rel.relationship, not is_neg)
            if key not in existing and opp_key not in existing:
                self.relationships.append(rel)
                existing.add(key)
                applied.append(f"rel_{rel.room_a}_{rel.relationship.value}_{rel.room_b}")
        return applied

    def check_user_reply_resolutions(self, reply: str) -> None:
        """
        Check user reply for explicit resolutions of ambiguities and specifications
        for room references, opening counts, and door details.
        """
        clean = reply.strip().lower()

        # 1. Hall / Room doors and entrance specification
        # Look for explicit door specifications, e.g. "one main entrance door", "1 door", "4 doors"
        hall_door_match = re.search(r"\b(?:hall(?:_1)?|entrance)\b.*?\b(\d+|one|two|three|four)\s+(?:main\s+)?(?:entrance\s+)?doors?\b", clean)
        if not hall_door_match:
            hall_door_match = re.search(r"\b(\d+|one|two|three|four)\s+(?:main\s+)?(?:entrance\s+)?doors?\b.*?\b(?:for\s+)?(?:the\s+)?hall(?:_1)?\b", clean)
        if not hall_door_match and "main entrance door" in clean:
            hall_door_match = re.search(r"\b(\d+|one|two|three|four)\s+main\s+entrance\s+door\b", clean)

        if hall_door_match:
            word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4}
            d_str = hall_door_match.group(1).lower()
            count = word_to_num.get(d_str, int(d_str) if d_str.isdigit() else 1)
            for r in self.rooms:
                if r.room_type == "hall":
                    r.door_count = count
                    if "main entrance" in clean or "entrance" in clean:
                        r.notes = "main entrance door" if not r.notes else f"{r.notes}; main entrance door"
                    self.set_room_field_source(r.room_id or r.room_type, "door_count", FieldSource.USER_EXPLICIT)
            self.resolve_ambiguity(r"hall.*door")

        # Hall windows specification, e.g. "no hall windows", "no windows in the hall", "0 windows"
        if "hall" in clean and re.search(r"\b(no|0|zero)\s+windows?\b", clean):
            for r in self.rooms:
                if r.room_type == "hall":
                    r.window_count = 0
                    self.set_room_field_source(r.room_id or r.room_type, "window_count", FieldSource.USER_EXPLICIT)
            self.resolve_ambiguity(r"hall.*window")

        # 2. Generic room ambiguity resolution
        if "generic" in clean or ("room" in clean and any(w in clean for w in ("bedroom", "living", "function", "type"))):
            self.resolve_ambiguity(r"generic\s+rooms?.*function|room.*type")

        # 3. Room reference ambiguity resolution
        # Handles user clarifications such as:
        # "bedroom meant bedroom_2", "the bedroom is bedroom 2", "second bedroom", "bedroom 2"
        for amb in list(self.ambiguities):
            m_amb = re.search(r"Ambiguous room reference '([^']+)' in relationship: there are \d+ (\w+)s", amb)
            if m_amb:
                raw_ref = m_amb.group(1)
                base_type = m_amb.group(2)
                resolved_ids = resolve_room_reference(clean, self)
                if not resolved_ids:
                    for r in self.rooms:
                        if r.room_id and r.room_id.lower() in clean:
                            resolved_ids = [r.room_id]
                            break
                        base, idx = parse_room_name(r.room_id or "")
                        if idx is not None:
                            patterns = [
                                rf"\b{base}\s*{idx}\b",
                                rf"\b{base}_{idx}\b",
                                rf"\b{base}\s*#\s*{idx}\b",
                            ]
                            for word, num in _ORDINALS.items():
                                if num == idx:
                                    patterns.append(rf"\b{word}\s+{base}\b")
                                    patterns.append(rf"\b{base}\s+{word}\b")
                            if any(re.search(pat, clean) for pat in patterns):
                                resolved_ids = [r.room_id]
                                break

                clarif_match = re.search(rf"\b{base_type}\s+(?:meant|means|is|was|refers?\s+to)\s+([a-zA-Z0-9_\s]+)\b", clean)
                if clarif_match:
                    target_str = clarif_match.group(1).strip()
                    res_target = resolve_room_reference(target_str, self)
                    if res_target:
                        resolved_ids = res_target

                if resolved_ids:
                    canonical_id = resolved_ids[0]
                    self.ambiguities = [a for a in self.ambiguities if a != amb]
                    self.warnings = [w for w in self.warnings if w != amb]
                    remaining_unresolved = []
                    for unres in self._unresolved_relationships:
                        if unres.get("base") == base_type:
                            rel = unres["rel"]
                            other_raw = unres["other_endpoint_raw"]
                            other_res = resolve_room_reference(other_raw, self)
                            other_id = other_res[0] if other_res else other_raw
                            if unres["ambiguous_endpoint"] == "a":
                                new_rel = RoomRelationship(
                                    room_a=canonical_id,
                                    room_b=other_id,
                                    relationship=rel.relationship,
                                    notes=rel.notes,
                                    negative=getattr(rel, "negative", False),
                                    source=FieldSource.USER_EXPLICIT,
                                )
                            else:
                                new_rel = RoomRelationship(
                                    room_a=other_id,
                                    room_b=canonical_id,
                                    relationship=rel.relationship,
                                    notes=rel.notes,
                                    negative=getattr(rel, "negative", False),
                                    source=FieldSource.USER_EXPLICIT,
                                )
                            self.relationships.append(new_rel)
                        else:
                            remaining_unresolved.append(unres)
                    self._unresolved_relationships = remaining_unresolved

    def resolve_ambiguity(self, pattern: str) -> None:
        """Resolve any ambiguity matching the regex pattern and remove from warnings."""
        regex = re.compile(pattern, re.IGNORECASE)
        self.ambiguities = [a for a in self.ambiguities if not regex.search(a)]
        self.warnings = [w for w in self.warnings if not regex.search(w)]

    def synchronize_constraints_with_relationships(self) -> None:
        """
        Ensure natural-language constraints in self.constraints do not contradict
        or misrepresent the structured relationships in self.relationships.

        Preserves all useful constraints unless they assert a relationship polarity
        opposite to what is recorded in self.relationships.
        """
        if not self.constraints:
            return

        valid_constraints: List[str] = []
        for c in self.constraints:
            rels_in_c = _extract_relationships_from_text(c, self)
            contradiction_found = False
            for rc in rels_in_c:
                pair_c = (min(rc.room_a.lower(), rc.room_b.lower()), max(rc.room_a.lower(), rc.room_b.lower()))
                is_neg_c = getattr(rc, "negative", False)
                for r_state in self.relationships:
                    pair_s = (min(r_state.room_a.lower(), r_state.room_b.lower()), max(r_state.room_a.lower(), r_state.room_b.lower()))
                    if pair_c == pair_s and rc.relationship == r_state.relationship:
                        is_neg_s = getattr(r_state, "negative", False)
                        if is_neg_c != is_neg_s:
                            contradiction_found = True
                            break
                if contradiction_found:
                    break

            if not contradiction_found and c not in valid_constraints:
                valid_constraints.append(c)

        self.constraints = valid_constraints

    def recalculate_warnings(self) -> List[str]:
        """
        Recalculate warnings reflecting the current state only.
        Removes obsolete warnings and retains only active ambiguities.
        """
        self.synchronize_constraints_with_relationships()
        warnings: List[str] = []

        # 1. Active ambiguities (excluding any whose subject is already resolved)
        active_ambs: List[str] = []
        for amb in self.ambiguities:
            if re.search(r"\bunit[s]?\b|\bmeasurement\b", amb, re.IGNORECASE):
                if self.units is not None:
                    continue  # Units resolved!
            if re.search(r"hall.*door", amb, re.IGNORECASE):
                hall = next((r for r in self.rooms if r.room_type == "hall"), None)
                if hall and hall.door_count is not None:
                    continue  # Resolved!
            if re.search(r"generic\s+rooms?.*function|room.*type", amb, re.IGNORECASE):
                if self.clarification_round > 0 or all(r.room_type != "room" for r in self.rooms):
                    continue  # Resolved!
            if re.search(r"\bwindow[s]?\b", amb, re.IGNORECASE):
                if all(r.window_count is not None for r in self.rooms):
                    continue  # Resolved!
            if re.search(r"\bdoor[s]?\b", amb, re.IGNORECASE):
                if all(r.door_count is not None for r in self.rooms):
                    continue  # Resolved!
            active_ambs.append(amb)
            if amb not in warnings:
                warnings.append(amb)

        self.ambiguities = active_ambs

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

            if norm_a not in valid_ids:
                res_a = resolve_room_reference(rel.room_a, self)
                if len(res_a) == 1 and res_a[0] in valid_ids:
                    norm_a = res_a[0]
            if norm_b not in valid_ids:
                res_b = resolve_room_reference(rel.room_b, self)
                if len(res_b) == 1 and res_b[0] in valid_ids:
                    norm_b = res_b[0]

            # Prune dangling relationships whose endpoints cannot be resolved to canonical rooms
            if norm_a not in valid_ids or norm_b not in valid_ids or norm_a.lower() == norm_b.lower():
                continue

            is_neg = getattr(rel, "negative", False)
            rel_src = getattr(rel, "source", None) or FieldSource.USER_EXPLICIT
            resolved = RoomRelationship(
                room_a=norm_a,
                room_b=norm_b,
                relationship=rel.relationship,
                notes=rel.notes,
                negative=is_neg,
                source=rel_src,
            )
            key = (min(resolved.room_a.lower(), resolved.room_b.lower()), max(resolved.room_a.lower(), resolved.room_b.lower()), resolved.relationship, is_neg)
            if key not in existing:
                normalized_rels.append(resolved)
                existing.add(key)
        self.relationships = normalized_rels

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
            "defaults_preference": self.defaults_preference,
            "hybrid_default_categories": self.hybrid_default_categories,
            "user_supplied_units": self.user_supplied_units,
            "overall_dimensions": self.overall_dimensions.model_dump() if self.overall_dimensions else None,
            "room_field_sources": {
                r_id: {k: v.value for k, v in fields.items()}
                for r_id, fields in self.room_field_sources.items()
            },
            "checklist": {
                k: v.model_dump() for k, v in self.checklist.items()
            },
            "rooms": [r.model_dump() for r in self.rooms],
            "relationships": [r.model_dump() for r in self.relationships],
            "constraints": list(self.constraints),
            "preferences": list(self.preferences),
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
        state.defaults_preference = data.get("defaults_preference")
        state.hybrid_default_categories = data.get("hybrid_default_categories", [])
        state.user_supplied_units = data.get("user_supplied_units")
        if "overall_dimensions" in data and data["overall_dimensions"] is not None:
            state.overall_dimensions = Dimensions(**data["overall_dimensions"])
        raw_sources = data.get("room_field_sources", {})
        for r_id, fields in raw_sources.items():
            state.room_field_sources[r_id] = {
                k: FieldSource(v) for k, v in fields.items()
            }

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
        known_modes = {"use_defaults", "specify_details", "hybrid"}
        if state.defaults_preference in known_modes and state.defaults_preference not in state.preferences:
            state.preferences.append(state.defaults_preference)
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


_ROOM_SYNONYMS: Dict[str, str] = {
    "bed": "bedroom",
    "br": "bedroom",
    "bdr": "bedroom",
    "bath": "bathroom",
    "ba": "bathroom",
    "btr": "bathroom",
    "toilet": "bathroom",
    "wc": "bathroom",
    "washroom": "bathroom",
    "restroom": "bathroom",
    "powder_room": "bathroom",
    "powder": "bathroom",
    "corridor": "hall",
    "hallway": "hall",
    "living": "living_room",
    "livingroom": "living_room",
    "lounge": "living_room",
    "dining": "dining_room",
    "diningroom": "dining_room",
    "kit": "kitchen",
}

_ORDINALS: Dict[str, int] = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}


def parse_room_name(name: str) -> tuple[str, Optional[int]]:
    """
    Parse a room name or identifier like 'room 1', 'room_2', 'bathroom', 'hall 1',
    'the second bedroom', 'second bedroom', 'bedroom #2', 'bed 1', 'bath'.
    Returns (canonical_base_room_type, index_or_None).
    """
    if not name:
        return ("", None)
    clean = name.strip().lower()
    # Remove leading determiners: 'the', 'a', 'an', 'both', 'all', 'each'
    clean = re.sub(r"^(?:the|a|an|both|all|each)\s+", "", clean).strip()

    # Prefix ordinals: "second bedroom", "2nd bedroom"
    for word, num in _ORDINALS.items():
        m = re.match(rf"^{word}\s+([a-zA-Z\s_-]+)$", clean)
        if m:
            base_str = m.group(1).strip().replace("-", "_").replace(" ", "_")
            if base_str.endswith("s") and not base_str.endswith("ss") and len(base_str) > 3:
                base_str = base_str[:-1]
            base_str = _ROOM_SYNONYMS.get(base_str, base_str)
            return (base_str, num)

    # Suffix ordinals: "bedroom second"
    for word, num in _ORDINALS.items():
        m = re.match(rf"^([a-zA-Z\s_-]+)\s+{word}$", clean)
        if m:
            base_str = m.group(1).strip().replace("-", "_").replace(" ", "_")
            if base_str.endswith("s") and not base_str.endswith("ss") and len(base_str) > 3:
                base_str = base_str[:-1]
            base_str = _ROOM_SYNONYMS.get(base_str, base_str)
            return (base_str, num)

    # Standard numeric pattern: 'bedroom 2', 'bedroom_2', 'bedroom #2', 'bedroom-2', 'bedroom1', 'bed1'
    m = re.match(r"^([a-zA-Z\s_-]+?)(?:[\s_#-]*(\d+))?$", clean)
    if m:
        base = m.group(1).strip().replace("-", "_").replace(" ", "_")
        idx = int(m.group(2)) if m.group(2) else None
        if idx is None and base.endswith("s") and not base.endswith("ss") and len(base) > 3:
            base = base[:-1]
        base = _ROOM_SYNONYMS.get(base, base)
        return (base, idx)

    clean_norm = clean.replace("-", "_").replace(" ", "_")
    if clean_norm.endswith("s") and not clean_norm.endswith("ss") and len(clean_norm) > 3:
        clean_norm = clean_norm[:-1]
    clean_norm = _ROOM_SYNONYMS.get(clean_norm, clean_norm)
    return (clean_norm, None)


def is_collective_room_reference(name: str) -> bool:
    """Return True if name indicates a collective room reference (e.g. 'both bedrooms', 'all bedrooms', 'the bedrooms', 'bedrooms')."""
    if not name:
        return False
    clean = name.strip().lower()
    if any(re.search(rf"\b{w}\b", clean) for w in ("both", "all", "each", "every")):
        return True
    words = clean.split()
    return any(w.endswith("s") and len(w) > 3 and not w.endswith("ss") for w in words)


def resolve_room_reference(name: str, state: Any) -> List[str]:
    """
    Resolve a free-form or structured room reference to canonical room ID(s)
    present in the state or rooms list (e.g. 'bedroom_1', 'bedroom_2').

    Handles:
    - Canonical room IDs directly ('bedroom_1')
    - Numbered / ordinal references ('bedroom 2', 'the second bedroom' -> ['bedroom_2'])
    - Synonyms / abbreviations ('bed 1' -> ['bedroom_1'], 'bath' -> ['bathroom_1'])
    - Collective references ('both bedrooms', 'the bedrooms' -> ['bedroom_1', 'bedroom_2'])
    - Unique room types ('hall' -> ['hall_1'] when 1 hall exists)
    - Returns [] when genuinely ambiguous or when target does not exist.
    """
    if not name:
        return []
    rooms = getattr(state, "rooms", state)
    if not rooms or not isinstance(rooms, (list, tuple)):
        return []
    clean = name.strip().lower()

    # Direct match on room_id
    for r in rooms:
        if getattr(r, "room_id", None) and r.room_id.lower() == clean:
            return [r.room_id]

    base, idx = parse_room_name(clean)
    base = _ROOM_SYNONYMS.get(base, base)

    # Explicit index
    if idx is not None:
        target_id = f"{base}_{idx}"
        for r in rooms:
            if getattr(r, "room_id", None) and r.room_id.lower() == target_id.lower():
                return [r.room_id]
        return []  # Target room does not exist; never return phantom room IDs

    # Collective reference
    if is_collective_room_reference(name):
        matching = [r.room_id for r in rooms if getattr(r, "room_id", None) and (r.room_type == base or parse_room_name(r.room_id)[0] == base)]
        if matching:
            return matching

    # No index - check if room type is unique
    matching = [r.room_id for r in rooms if getattr(r, "room_id", None) and (r.room_type == base or parse_room_name(r.room_id)[0] == base)]
    if len(matching) == 1:
        return matching

    return []


def _extract_relationships_from_text(text: str, state: Any) -> List[RoomRelationship]:
    """
    Extract structured RoomRelationship objects from natural language text
    (e.g. in prompt, user replies, or candidate constraints).
    Preserves negative status and uses FieldSource.USER_EXPLICIT.
    """
    if not text:
        return []
    rooms = getattr(state, "rooms", state)
    if not rooms or not isinstance(rooms, (list, tuple)):
        return []

    extracted: List[RoomRelationship] = []
    seen_pairs = set()

    type_counts: Dict[str, int] = {}
    for r in rooms:
        type_counts[r.room_type] = type_counts.get(r.room_type, 0) + 1

    sentences = re.split(r"[.;\n]+", text)

    for sentence in sentences:
        s_clean = sentence.strip()
        if not s_clean:
            continue

        # Split compound clauses separated by conjunctions followed by relational verbs or predicates,
        # e.g. "Bathroom must share 1 wall with bedroom1 and share another wall with bedroom2"
        raw_clauses = re.split(
            r"(?<=[a-zA-Z0-9])\s+(?:and|,)\s+(?=(?:(?:it|they)\s+)?(?:must\s+|should\s+|will\s+|also\s+)?(?:share[s]?|sharing|connect[s]?|connecting|is\s+adjacent|are\s+adjacent|adjacent))",
            s_clean,
            flags=re.IGNORECASE,
        )

        last_subject: Optional[str] = None

        for clause in raw_clauses:
            c_clean = clause.strip()
            if not c_clean:
                continue
            c_low = c_clean.lower()

            is_neg = bool(re.search(r"\b(must\s+not|not\s+connect|no\s+connection|never\s+adjacent|cannot\s+connect|should\s+not|not\s+share)\b", c_low))

            rel_type = None
            split_pattern = None

            if re.search(r"\b(?:must\s+|should\s+|will\s+|to\s+|has\s+to\s+|have\s+to\s+)?(?:share[s]?|sharing)\s+(?:(?:a|an|1|one|another|(?:a\s+)?common|\d+)\s+)?wall[s]?\s+with\b", c_low):
                rel_type = RelationshipType.SHARES_WALL_WITH
                split_pattern = r"\b(?:must\s+|should\s+|will\s+|to\s+|has\s+to\s+|have\s+to\s+)?(?:share[s]?|sharing)\s+(?:(?:a|an|1|one|another|(?:a\s+)?common|\d+)\s+)?wall[s]?\s+with\b"
            elif re.search(r"\b(?:must\s+be\s+|should\s+be\s+|will\s+be\s+|is\s+|are\s+|to\s+be\s+)?(?:adjacent\s+to|adjacent|next\s+to|beside)\b", c_low):
                rel_type = RelationshipType.ADJACENT_TO
                split_pattern = r"\b(?:must\s+be\s+|should\s+be\s+|will\s+be\s+|is\s+|are\s+|to\s+be\s+)?(?:adjacent\s+to|adjacent|next\s+to|beside)\b"
            elif re.search(r"\b(?:must\s+|should\s+|will\s+|to\s+|has\s+to\s+)?(?:connect[s]?|connecting)\s+(?:directly\s+)?to\b", c_low):
                rel_type = RelationshipType.CONNECTS_TO
                split_pattern = r"\b(?:must\s+|should\s+|will\s+|to\s+|has\s+to\s+)?(?:connect[s]?|connecting)\s+(?:directly\s+)?to\b"

            if not rel_type or not split_pattern:
                continue

            parts = re.split(split_pattern, c_clean, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 2:
                left_str = parts[0].strip()
                right_str = parts[1].strip()

                # If left is empty or a conjunction/pronoun, inherit from last_subject
                if (not left_str or left_str.lower() in ("and", "also", "it", "they", "that", "which")) and last_subject:
                    left_str = last_subject
                elif left_str:
                    last_subject = left_str

                # Case A: "the bedrooms must be adjacent" (right_str is empty or minimal)
                if not right_str or right_str.lower() in ("each other", "one another", "together", "."):
                    base, idx = parse_room_name(left_str)
                    if idx is None and type_counts.get(base, 0) == 2:
                        rooms_of_type = [r.room_id for r in rooms if r.room_id and (r.room_type == base or parse_room_name(r.room_id)[0] == base)]
                        if len(rooms_of_type) == 2:
                            pair_key = (min(rooms_of_type[0].lower(), rooms_of_type[1].lower()), max(rooms_of_type[0].lower(), rooms_of_type[1].lower()), rel_type, is_neg)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                extracted.append(RoomRelationship(
                                    room_a=rooms_of_type[0],
                                    room_b=rooms_of_type[1],
                                    relationship=rel_type,
                                    negative=is_neg,
                                    source=FieldSource.USER_EXPLICIT,
                                ))
                        continue

                # Case B: Standard two-sided relationship e.g. "bathroom ... shares a wall with ... bedroom 2"
                left_clean = re.sub(r"^(?:and\s+|that\s+|with\s+|the\s+|also\s+)+", "", left_str, flags=re.IGNORECASE).strip()
                res_left = resolve_room_reference(left_clean, rooms)
                if not res_left:
                    for r in rooms:
                        if r.room_id and r.room_id.lower() in left_clean.lower():
                            res_left = [r.room_id]
                            break
                        base, idx = parse_room_name(r.room_id or "")
                        if idx is not None and (f"{base} {idx}" in left_clean.lower() or f"{base}_{idx}" in left_clean.lower() or f"{base}{idx}" in left_clean.lower()):
                            res_left = [r.room_id]
                            break
                        elif idx is None and base in left_clean.lower() and type_counts.get(base, 0) == 1:
                            res_left = [r.room_id]
                            break

                sub_targets = re.split(r"\b(?:and|,)\b", right_str, flags=re.IGNORECASE)
                res_rights = []
                for sub in sub_targets:
                    sub_clean = sub.strip()
                    if not sub_clean or sub_clean.lower() in ("each other", "with", "to"):
                        continue
                    sub_res = resolve_room_reference(sub_clean, rooms)
                    if not sub_res:
                        for r in rooms:
                            if r.room_id and r.room_id.lower() in sub_clean.lower():
                                sub_res = [r.room_id]
                                break
                            base, idx = parse_room_name(r.room_id or "")
                            if idx is not None and (f"{base} {idx}" in sub_clean.lower() or f"{base}_{idx}" in sub_clean.lower() or f"{base}{idx}" in sub_clean.lower()):
                                sub_res = [r.room_id]
                                break
                            elif idx is None and base in sub_clean.lower() and type_counts.get(base, 0) == 1:
                                sub_res = [r.room_id]
                                break
                    res_rights.extend(sub_res)

                if res_left and res_rights:
                    for a in res_left:
                        for b in res_rights:
                            if a.lower() != b.lower():
                                pair_key = (min(a.lower(), b.lower()), max(a.lower(), b.lower()), rel_type, is_neg)
                                if pair_key not in seen_pairs:
                                    seen_pairs.add(pair_key)
                                    extracted.append(RoomRelationship(
                                        room_a=a,
                                        room_b=b,
                                        relationship=rel_type,
                                        negative=is_neg,
                                        source=FieldSource.USER_EXPLICIT,
                                    ))

    return extracted


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
