"""
IR_builder.py
=============
Building Intermediate Representation (Building IR) v1 Builder.

Deterministic, LLM-independent component that transforms validated
CandidateRequirements or ExtractionResult data into a canonical Building IR v1
graph document according to the Building IR v1 Specification.

Responsibilities
----------------
* Input normalization and structural validation.
* Construction of authoritative entity records (building, rooms, openings).
* Conversion and deduplication of spatial/functional requirement edges.
* Preservation of constraints (structured and unparsed natural language).
* Representation of graph indexing (node_refs, edge_refs by ID).
* Preservation of provenance and assumption records.
* Creation of initial placeholder layers for downstream pipeline stages
  (layout, validation).

Out of scope
------------
* LLM calling or interactive clarification.
* Layout generation, wall placement, or coordinate solving.
* Applying defaults (defaults applied upstream are preserved).
* File I/O (persistence is owned by the orchestrator/application).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from schemas import (
    CandidateRequirements,
    ExtractionResult,
    RelationshipType,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IRBuilderError(ValueError):
    """Base exception for all Building IR Builder errors."""


class IRInputError(IRBuilderError):
    """Raised when input requirements are malformed, missing required fields, or of invalid type."""


class IRReferenceError(IRBuilderError):
    """Raised when an edge or constraint references a non-existent entity."""


class IRValidationError(IRBuilderError):
    """Raised when structural validation of the generated Building IR fails."""


# Backward compatibility alias for earlier stub imports
BuildingIRNotImplementedError = IRBuilderError


# ---------------------------------------------------------------------------
# Constants and Specification Mappings
# ---------------------------------------------------------------------------

IR_VERSION: str = "1.0"
CANONICAL_UNITS: str = "m"

SYMMETRIC_EDGE_TYPES: Set[str] = {
    RelationshipType.ADJACENT_TO.value,
    RelationshipType.SHARES_WALL_WITH.value,
    RelationshipType.CONNECTS_TO.value,
    RelationshipType.SHARED_FACILITY.value,
    RelationshipType.OPPOSITE_TO.value,
}

DIRECTED_EDGE_TYPES: Set[str] = {
    "opens_into",
    "belongs_to",
}

VALID_EDGE_TYPES: Set[str] = SYMMETRIC_EDGE_TYPES | DIRECTED_EDGE_TYPES

VALID_CONSTRAINT_STATUSES: Set[str] = {
    "not_evaluated",
    "satisfied",
    "violated",
    "unsupported",
    "unresolved",
}

VALID_PROVENANCE_SOURCES: Set[str] = {
    "user_explicit",
    "system_default",
    "derived",
    "unresolved",
    "inferred",
    "user_confirmed_default",
}


# ---------------------------------------------------------------------------
# Internal Mapping and Validation Helpers
# ---------------------------------------------------------------------------


def _extract_raw_data(
    candidate_requirements: Union[ExtractionResult, CandidateRequirements, Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize input into a plain Python dictionary."""
    if isinstance(candidate_requirements, (ExtractionResult, CandidateRequirements)):
        return candidate_requirements.model_dump()
    if hasattr(candidate_requirements, "model_dump") and callable(candidate_requirements.model_dump):
        return candidate_requirements.model_dump()
    if isinstance(candidate_requirements, dict):
        return candidate_requirements
    raise IRInputError(
        f"Expected ExtractionResult, CandidateRequirements, or dict, "
        f"got {type(candidate_requirements).__name__}."
    )


def _build_building_record(data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Construct the authoritative building record.

    Validates building type, floor count (v1 single-floor prototype constraint),
    and footprint/overall dimensions without fabricating coordinates or missing values.
    """
    building_id = str(data.get("building_id") or "bldg_001")

    raw_btype = data.get("building_type")
    if not raw_btype:
        raise IRInputError("Missing required field 'building_type'.")
    building_type = raw_btype.value if hasattr(raw_btype, "value") else str(raw_btype)

    raw_fcount = data.get("floor_count")
    if raw_fcount is None:
        raise IRInputError("Missing required field 'floor_count'.")
    if not isinstance(raw_fcount, int) or isinstance(raw_fcount, bool) or raw_fcount < 1:
        raise IRValidationError(f"'floor_count' must be a positive integer, got {raw_fcount}.")
    if raw_fcount != 1:
        raise IRValidationError(
            f"Building IR v1 prototype supports single-floor buildings only, got floor_count={raw_fcount}."
        )

    raw_dims = data.get("overall_dimensions") or {}
    if hasattr(raw_dims, "model_dump"):
        raw_dims = raw_dims.model_dump()

    w_val = raw_dims.get("width_m")
    l_val = raw_dims.get("length_m")
    fh_val = raw_dims.get("floor_height_m")
    wt_val = raw_dims.get("wall_thickness_m")

    # Backward compatibility with wall_thickness_mm if wall_thickness_m is absent
    if wt_val is None and raw_dims.get("wall_thickness_mm") is not None:
        raw_mm = raw_dims["wall_thickness_mm"]
        if isinstance(raw_mm, (int, float)) and not isinstance(raw_mm, bool) and raw_mm > 0:
            wt_val = round(raw_mm / 1000.0, 4)

    # Validate numeric properties when present
    for name, val in [
        ("width_m", w_val),
        ("length_m", l_val),
        ("floor_height_m", fh_val),
        ("wall_thickness_m", wt_val),
    ]:
        if val is not None:
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
                raise IRValidationError(f"overall_dimensions.{name} must be a positive number, got {val}.")

    building_record: Dict[str, Any] = {
        "id": building_id,
        "building_type": building_type,
        "floor_count": raw_fcount,
        "overall_dimensions": {
            "width_m": float(w_val) if w_val is not None else None,
            "length_m": float(l_val) if l_val is not None else None,
            "floor_height_m": float(fh_val) if fh_val is not None else None,
            "wall_thickness_m": float(wt_val) if wt_val is not None else None,
        },
    }

    return building_record, building_id


def _build_room_records(
    data: Dict[str, Any],
    defaults_applied: List[str],
    provenance_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Construct authoritative room records under requirements.rooms.

    Ensures stable IDs, validates dimensions and opening counts, computes
    derived area_from_dimensions_sqm without inventing target areas or placements,
    and returns a mapping from room_id to room_type.
    """
    raw_rooms = data.get("rooms")
    if not raw_rooms or not isinstance(raw_rooms, list):
        raise IRInputError("Input requirements must contain a non-empty list of 'rooms'.")

    rooms: List[Dict[str, Any]] = []
    room_id_to_type: Dict[str, str] = {}
    seen_room_ids: Set[str] = set()

    for idx, r_entry in enumerate(raw_rooms, start=1):
        r = r_entry.model_dump() if hasattr(r_entry, "model_dump") else dict(r_entry)

        r_type = str(r.get("room_type") or "room").strip().lower()
        count = r.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise IRValidationError(f"Room count must be a positive integer, got {count}.")

        # In v1, individual rooms are authoritative entities.
        # If count > 1, expand into canonical individual rooms (e.g. bedroom_1, bedroom_2).
        expansion_count = count
        base_id = (r.get("room_id") or f"{r_type}_{idx}").strip()

        for c_idx in range(1, expansion_count + 1):
            if expansion_count > 1:
                r_id = f"{r_type}_{len(rooms) + 1}"
                r_label = f"{r_type}_{len(rooms) + 1}"
            else:
                r_id = base_id
                r_label = str(r.get("label") or r.get("room_id") or base_id).strip()

            if r_id in seen_room_ids:
                raise IRValidationError(f"Duplicate room ID detected: '{r_id}'.")
            seen_room_ids.add(r_id)
            room_id_to_type[r_id] = r_type

            # Dimension validation
            w_m = r.get("width_m")
            l_m = r.get("length_m")
            if w_m is not None:
                if not isinstance(w_m, (int, float)) or isinstance(w_m, bool) or w_m <= 0:
                    raise IRValidationError(f"Room '{r_id}' width_m must be a positive number, got {w_m}.")
                w_m = float(w_m)
            if l_m is not None:
                if not isinstance(l_m, (int, float)) or isinstance(l_m, bool) or l_m <= 0:
                    raise IRValidationError(f"Room '{r_id}' length_m must be a positive number, got {l_m}.")
                l_m = float(l_m)

            # Area representation (§8)
            target_area = r.get("target_area_sqm")
            if target_area is not None:
                if not isinstance(target_area, (int, float)) or isinstance(target_area, bool) or target_area <= 0:
                    raise IRValidationError(f"Room '{r_id}' target_area_sqm must be positive, got {target_area}.")
                target_area = float(target_area)

            area_from_dims = round(w_m * l_m, 4) if (w_m is not None and l_m is not None) else None

            # Openings requirement collection (§5, §8)
            openings: List[Dict[str, Any]] = []

            win_count = r.get("window_count")
            if win_count is not None:
                if not isinstance(win_count, int) or isinstance(win_count, bool) or win_count < 0:
                    raise IRValidationError(f"Room '{r_id}' window_count must be a non-negative integer, got {win_count}.")
                win_src = (
                    provenance_map.get(f"{r_id}.window_count")
                    or provenance_map.get(f"{r_id}_window_count")
                    or ("system_default" if f"{r_id}_window_count" in defaults_applied else "user_explicit")
                )
                openings.append({
                    "id": f"{r_id}_win",
                    "kind": "window",
                    "count": win_count,
                    "source": win_src,
                })

            door_count = r.get("door_count")
            if door_count is not None:
                if not isinstance(door_count, int) or isinstance(door_count, bool) or door_count < 0:
                    raise IRValidationError(f"Room '{r_id}' door_count must be a non-negative integer, got {door_count}.")
                door_src = (
                    provenance_map.get(f"{r_id}.door_count")
                    or provenance_map.get(f"{r_id}_door_count")
                    or ("system_default" if f"{r_id}_door_count" in defaults_applied else "user_explicit")
                )
                openings.append({
                    "id": f"{r_id}_door",
                    "kind": "door",
                    "count": door_count,
                    "source": door_src,
                })

            notes_val = r.get("notes")
            notes = str(notes_val).strip() if notes_val is not None and str(notes_val).strip() else None

            room_record = {
                "id": r_id,
                "type": r_type,
                "label": r_label,
                "dimensions": {
                    "width_m": w_m,
                    "length_m": l_m,
                },
                "area": {
                    "target_area_sqm": target_area,
                    "area_from_dimensions_sqm": area_from_dims,
                    "computed_area_sqm": None,
                },
                "openings": openings,
                "notes": notes,
            }
            rooms.append(room_record)

    return rooms, room_id_to_type


def _resolve_endpoint(
    endpoint: str,
    valid_entities: Set[str],
    room_type_to_ids: Dict[str, List[str]],
) -> str:
    """
    Resolve an endpoint identifier to a valid canonical entity ID.

    If given a room type (e.g. 'bedroom') and exactly one matching room exists,
    resolves to that room's canonical ID (e.g. 'bedroom_1'). Otherwise validates
    that the endpoint exists in valid_entities.
    """
    clean_ep = endpoint.strip()
    if clean_ep in valid_entities:
        return clean_ep

    # Check if endpoint matches a single room by room_type
    low_ep = clean_ep.lower()
    if low_ep in room_type_to_ids:
        matched_ids = room_type_to_ids[low_ep]
        if len(matched_ids) == 1:
            return matched_ids[0]
        raise IRReferenceError(
            f"Relationship endpoint '{clean_ep}' is ambiguous; matches multiple rooms: {matched_ids}."
        )

    raise IRReferenceError(
        f"Relationship endpoint '{clean_ep}' does not reference any known entity or valid room ID."
    )


def _build_edge_records(
    data: Dict[str, Any],
    valid_entities: Set[str],
    room_type_to_ids: Dict[str, List[str]],
    provenance_map: Dict[str, str],
    defaults_applied: List[str],
) -> List[Dict[str, Any]]:
    """
    Construct authoritative relationship edges under requirements.edges.

    Validates endpoints, canonicalizes symmetric pairs for duplicate detection,
    preserves distinct edge types between the same endpoints, and initializes
    evaluation status to 'not_evaluated'.
    """
    raw_rels = data.get("relationships") or []
    if not isinstance(raw_rels, list):
        raise IRInputError("'relationships' must be a list.")

    edges: List[Dict[str, Any]] = []
    seen_edge_keys: Set[Tuple[str, Tuple[str, ...], bool]] = set()

    for r_entry in raw_rels:
        rel = r_entry.model_dump() if hasattr(r_entry, "model_dump") else dict(r_entry)

        raw_from = rel.get("room_a") or rel.get("from")
        raw_to = rel.get("room_b") or rel.get("to")
        if not raw_from or not raw_to:
            raise IRInputError(f"Relationship record missing required endpoints: {rel}.")

        from_node = _resolve_endpoint(str(raw_from), valid_entities, room_type_to_ids)
        to_node = _resolve_endpoint(str(raw_to), valid_entities, room_type_to_ids)

        if from_node == to_node:
            raise IRValidationError(
                f"Self-referential relationship between '{from_node}' and '{to_node}' is not permitted."
            )

        raw_type = rel.get("relationship") or rel.get("type")
        if not raw_type:
            raise IRInputError(f"Relationship missing 'relationship' type: {rel}.")
        rel_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type).strip().lower()

        negative = bool(rel.get("negative", False))

        # Determine source provenance
        source_val = rel.get("source")
        if source_val is not None:
            source_str = source_val.value if hasattr(source_val, "value") else str(source_val)
        else:
            rel_prov_key1 = f"rel_{from_node}_{rel_type}_{to_node}"
            rel_prov_key2 = f"rel_{to_node}_{rel_type}_{from_node}"
            if rel_prov_key1 in provenance_map:
                source_str = provenance_map[rel_prov_key1]
            elif rel_prov_key2 in provenance_map:
                source_str = provenance_map[rel_prov_key2]
            elif rel_prov_key1 in defaults_applied or rel_prov_key2 in defaults_applied:
                source_str = "system_default"
            else:
                source_str = "user_explicit"

        # Canonicalize symmetric edges for deduplication
        if rel_type in SYMMETRIC_EDGE_TYPES:
            canon_endpoints = tuple(sorted([from_node, to_node]))
        else:
            canon_endpoints = (from_node, to_node)

        dedup_key = (rel_type, canon_endpoints, negative)
        if dedup_key in seen_edge_keys:
            # Skip exact duplicate
            continue
        seen_edge_keys.add(dedup_key)

        notes_val = rel.get("notes")
        notes = str(notes_val).strip() if notes_val is not None and str(notes_val).strip() else None

        edge_record = {
            "id": f"re_{len(edges) + 1:03d}",
            "from": from_node,
            "to": to_node,
            "type": rel_type,
            "negative": negative,
            "source": source_str,
            "status": "not_evaluated",
            "notes": notes,
        }
        edges.append(edge_record)

    return edges


def _build_constraint_records(
    data: Dict[str, Any],
    valid_entities: Set[str],
) -> List[Dict[str, Any]]:
    """
    Construct structured constraint records under requirements.constraints.

    Preserves plain natural language strings without fabricating operators or values,
    and supports structured constraints with status initialized to 'not_evaluated'.
    """
    raw_constraints = data.get("constraints") or []
    if not isinstance(raw_constraints, list):
        raise IRInputError("'constraints' must be a list.")

    constraints: List[Dict[str, Any]] = []

    for c_entry in raw_constraints:
        c_id = f"c_{len(constraints) + 1:03d}"

        if isinstance(c_entry, str):
            c_str = c_entry.strip()
            if not c_str:
                continue
            constraints.append({
                "id": c_id,
                "scope": "global",
                "target": None,
                "type": "unparsed_text",
                "property": None,
                "operator": None,
                "value": None,
                "unit": None,
                "source": "user_explicit",
                "status": "not_evaluated",
                "raw_text": c_str,
                "message": None,
            })
        elif isinstance(c_entry, dict):
            target = c_entry.get("target")
            if target and target not in valid_entities:
                raise IRReferenceError(f"Constraint target '{target}' references unknown entity.")

            status = str(c_entry.get("status") or "not_evaluated")
            if status not in VALID_CONSTRAINT_STATUSES:
                status = "not_evaluated"

            constraints.append({
                "id": c_entry.get("id") or c_id,
                "scope": str(c_entry.get("scope") or "global"),
                "target": target,
                "type": str(c_entry.get("type") or "custom"),
                "property": c_entry.get("property"),
                "operator": c_entry.get("operator"),
                "value": c_entry.get("value"),
                "unit": c_entry.get("unit"),
                "source": str(c_entry.get("source") or "user_explicit"),
                "status": status,
                "raw_text": c_entry.get("raw_text"),
                "message": c_entry.get("message"),
            })
        else:
            # Preserve unknown constraint representation
            constraints.append({
                "id": c_id,
                "scope": "global",
                "target": None,
                "type": "unparsed_text",
                "property": None,
                "operator": None,
                "value": None,
                "unit": None,
                "source": "user_explicit",
                "status": "not_evaluated",
                "raw_text": str(c_entry),
                "message": None,
            })

    return constraints


def _build_preference_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct preference records under requirements.preferences."""
    raw_prefs = data.get("preferences") or []
    if not isinstance(raw_prefs, list):
        raise IRInputError("'preferences' must be a list.")

    preferences: List[Dict[str, Any]] = []
    for p_entry in raw_prefs:
        p_id = f"pref_{len(preferences) + 1:03d}"
        if isinstance(p_entry, str):
            p_str = p_entry.strip()
            if not p_str:
                continue
            preferences.append({
                "id": p_id,
                "description": p_str,
                "source": "user_explicit",
            })
        elif isinstance(p_entry, dict):
            preferences.append({
                "id": p_entry.get("id") or p_id,
                "description": str(p_entry.get("description") or ""),
                "source": str(p_entry.get("source") or "user_explicit"),
            })

    return preferences


def _build_unresolved_records(
    data: Dict[str, Any],
    rooms: List[Dict[str, Any]],
    inconsistencies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Construct unresolved records under requirements.unresolved.

    Preserves warnings, un-defaulted missing dimensions, explicit ambiguity notes,
    and reports provenance/defaults inconsistencies without guessing.
    """
    unresolved: List[Dict[str, Any]] = []
    seen_descriptions: Set[str] = set()

    # 1. Existing explicit unresolved items if passed
    for item in data.get("unresolved") or []:
        desc = item.get("description") if isinstance(item, dict) else str(item)
        if desc and desc not in seen_descriptions:
            seen_descriptions.add(desc)
            unresolved.append({
                "id": f"unres_{len(unresolved) + 1:03d}",
                "description": desc,
                "status": "unresolved",
            })

    # 2. Ambiguities from LLM / candidate extraction
    for amb in data.get("ambiguities") or []:
        desc = str(amb).strip()
        if desc and desc not in seen_descriptions:
            seen_descriptions.add(desc)
            unresolved.append({
                "id": f"unres_{len(unresolved) + 1:03d}",
                "description": desc,
                "status": "unresolved",
            })

    # 3. Warnings from validation or extraction
    for warn in data.get("warnings") or []:
        desc = str(warn).strip()
        if desc and desc not in seen_descriptions:
            seen_descriptions.add(desc)
            unresolved.append({
                "id": f"unres_{len(unresolved) + 1:03d}",
                "description": desc,
                "status": "unresolved",
            })

    # 4. Check for rooms with missing dimensions
    for r in rooms:
        dims = r.get("dimensions") or {}
        missing = []
        if dims.get("width_m") is None:
            missing.append("width")
        if dims.get("length_m") is None:
            missing.append("length")
        if missing:
            desc = f"Room '{r['id']}' missing explicit or default {', '.join(missing)} dimension(s)."
            if desc not in seen_descriptions:
                seen_descriptions.add(desc)
                unresolved.append({
                    "id": f"unres_{len(unresolved) + 1:03d}",
                    "description": desc,
                    "status": "unresolved",
                })

    # 5. Provenance conflicts and inconsistencies
    for inc in inconsistencies or []:
        if inc and inc not in seen_descriptions:
            seen_descriptions.add(inc)
            unresolved.append({
                "id": f"unres_{len(unresolved) + 1:03d}",
                "description": inc,
                "status": "unresolved",
            })

    return unresolved


def _build_provenance_records(
    data: Dict[str, Any],
    rooms: List[Dict[str, Any]],
    building_record: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Construct provenance records sorted deterministically by target path.

    Preserves original source provenance without reclassification, records 'derived'
    provenance for calculated area metrics, assigns default provenance for defaulted
    building-level dimensions, and reports any conflicting provenance entries.
    """
    raw_prov = data.get("provenance")
    prov_dict: Dict[str, str] = {}

    if isinstance(raw_prov, dict):
        for target, val in raw_prov.items():
            src_str = val.value if hasattr(val, "value") else str(val)
            prov_dict[str(target)] = src_str
    elif isinstance(raw_prov, list):
        for entry in raw_prov:
            if isinstance(entry, dict) and "target" in entry and "source" in entry:
                prov_dict[str(entry["target"])] = str(entry["source"])

    # 1. Mark derived room areas as 'derived' (§8, §9)
    for r in rooms:
        r_id = r["id"]
        area_info = r.get("area") or {}
        if area_info.get("area_from_dimensions_sqm") is not None:
            prov_dict[f"{r_id}.area_from_dimensions_sqm"] = "derived"

    # 2. Building dimensions default provenance if not already explicit
    overall_dims = building_record.get("overall_dimensions") or {}
    if overall_dims.get("floor_height_m") is not None:
        if "overall_dimensions.floor_height_m" not in prov_dict and "building.floor_height_m" not in prov_dict:
            prov_dict["overall_dimensions.floor_height_m"] = "system_default"
    if overall_dims.get("wall_thickness_m") is not None:
        if "overall_dimensions.wall_thickness_m" not in prov_dict and "building.wall_thickness_m" not in prov_dict:
            prov_dict["overall_dimensions.wall_thickness_m"] = "system_default"

    # 3. Detect inconsistencies between input provenance and defaults_applied
    defaults_applied = set(str(d) for d in (data.get("defaults_applied") or []))
    inconsistencies: List[str] = []

    for target, source in prov_dict.items():
        if source == "derived":
            continue
        norm_key = target.replace(".", "_")
        if norm_key in defaults_applied and source == "user_explicit":
            inconsistencies.append(
                f"Provenance conflict: '{target}' is marked 'user_explicit' in provenance, "
                f"but listed in defaults_applied."
            )

    provenance_list: List[Dict[str, Any]] = [
        {"target": target, "source": prov_dict[target]}
        for target in sorted(prov_dict.keys())
    ]

    return provenance_list, inconsistencies


def _build_assumption_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct human-readable assumptions from defaults applied."""
    assumptions: List[Dict[str, Any]] = []
    defaults_applied = data.get("defaults_applied") or []

    for item in sorted(defaults_applied):
        assumptions.append({
            "id": f"asm_{len(assumptions) + 1:03d}",
            "description": f"Applied default: {item}",
            "source": "system_default",
        })

    # Preserve any explicit input assumptions
    for raw_asm in data.get("assumptions") or []:
        desc = raw_asm.get("description") if isinstance(raw_asm, dict) else str(raw_asm)
        assumptions.append({
            "id": f"asm_{len(assumptions) + 1:03d}",
            "description": desc,
            "source": raw_asm.get("source", "system_default") if isinstance(raw_asm, dict) else "system_default",
        })

    return assumptions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_ir(
    candidate_requirements: Union[ExtractionResult, CandidateRequirements, Dict[str, Any]],
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct a canonical Building IR v1 document from candidate requirements.

    Parameters
    ----------
    candidate_requirements : Union[ExtractionResult, CandidateRequirements, Dict[str, Any]]
        The validated extraction result or candidate requirements dictionary.
    session_id : Optional[str], keyword-only
        Optional session identifier override. If not provided, the session_id
        from the input data is used or a fresh uuid is generated.

    Returns
    -------
    Dict[str, Any]
        A structured, JSON-serializable Building IR v1 document.

    Raises
    ------
    IRInputError
        If required input fields are missing or of an invalid type.
    IRReferenceError
        If relationships or constraints reference non-existent entity IDs.
    IRValidationError
        If structural requirements or constraints fail validation.
    """
    data = _extract_raw_data(candidate_requirements)

    # 1. Determine session ID and units
    sid = str(session_id or data.get("session_id") or uuid.uuid4())
    units = str(data.get("units") or CANONICAL_UNITS).strip().lower()
    if units != CANONICAL_UNITS:
        raise IRValidationError(
            f"Canonical units for Building IR v1 must be '{CANONICAL_UNITS}', got '{units}'."
        )

    # 2. Extract building record
    building_record, building_id = _build_building_record(data)

    # 3. Extract provenance map and defaults applied for opening/room annotations
    raw_prov = data.get("provenance") or {}
    prov_map: Dict[str, str] = {}
    if isinstance(raw_prov, dict):
        for k, v in raw_prov.items():
            prov_map[k] = v.value if hasattr(v, "value") else str(v)
    elif isinstance(raw_prov, list):
        for item in raw_prov:
            if isinstance(item, dict) and "target" in item and "source" in item:
                prov_map[str(item["target"])] = str(item["source"])

    defaults_applied: List[str] = [str(d) for d in (data.get("defaults_applied") or [])]

    # 4. Extract authoritative room records
    room_records, room_id_to_type = _build_room_records(data, defaults_applied, prov_map)

    # Assemble mapping from room_type to list of room IDs for endpoint resolution
    type_to_ids: Dict[str, List[str]] = {}
    for r_id, r_type in room_id_to_type.items():
        type_to_ids.setdefault(r_type, []).append(r_id)

    # Valid entity IDs that edges and constraints may reference
    valid_entities: Set[str] = {building_id} | set(room_id_to_type.keys()) | {"exterior"}

    # 5. Extract authoritative relationship edges
    edge_records = _build_edge_records(data, valid_entities, type_to_ids, prov_map, defaults_applied)

    # 6. Extract provenance records, derived area provenance, and detect conflicts
    provenance_records, inconsistencies = _build_provenance_records(data, room_records, building_record)

    # 7. Extract constraints, preferences, and unresolved requirements
    constraint_records = _build_constraint_records(data, valid_entities)
    preference_records = _build_preference_records(data)
    unresolved_records = _build_unresolved_records(data, room_records, inconsistencies)

    # 8. Construct graph index (references IDs only; does not duplicate objects)
    node_refs = [building_id] + [r["id"] for r in room_records]
    edge_refs = [e["id"] for e in edge_records]

    # 9. Construct initial layout layer (status 'not_started'; no coordinates or walls invented)
    layout_layer: Dict[str, Any] = {
        "status": "not_started",
        "room_placements": [],
        "walls": [],
        "openings": [],
        "footprint": None,
        "edges": [],
        "validation_results": [],
    }

    # 10. Construct initial validation layer (all statuses 'not_run')
    validation_layer: Dict[str, Any] = {
        "pre_layout": {"status": "not_run", "results": []},
        "layout": {"status": "not_run", "results": []},
        "geometry": {"status": "not_run", "results": []},
    }

    # 11. Extract assumption records
    assumption_records = _build_assumption_records(data)

    # 11. Assemble the canonical Building IR v1 document
    building_ir: Dict[str, Any] = {
        "ir_version": IR_VERSION,
        "session_id": sid,
        "units": CANONICAL_UNITS,
        "building": building_record,
        "requirements": {
            "rooms": room_records,
            "edges": edge_records,
            "constraints": constraint_records,
            "preferences": preference_records,
            "unresolved": unresolved_records,
        },
        "graph": {
            "node_refs": node_refs,
            "edge_refs": edge_refs,
        },
        "layout": layout_layer,
        "validation": validation_layer,
        "provenance": provenance_records,
        "assumptions": assumption_records,
    }

    # 12. Strict structural assertion: output must be JSON serializable
    try:
        json.dumps(building_ir)
    except (TypeError, ValueError) as exc:
        raise IRValidationError(f"Constructed Building IR failed JSON serialization: {exc}") from exc

    return building_ir


if __name__ == "__main__":
    import argparse

    cli_parser = argparse.ArgumentParser(
        description="Building IR v1 Builder — Build Building IR from requirements JSON",
    )
    cli_parser.add_argument(
        "requirements_file",
        help="Path to requirements JSON file (e.g. outputs/<session_id>.json)",
    )
    cli_parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT_FILE",
        help="Optional output path for Building IR JSON (defaults to outputs/<session_id>_ir.json)",
    )

    cli_args = cli_parser.parse_args()

    req_path = cli_args.requirements_file
    if not os.path.isfile(req_path):
        print(f"❌ File not found: '{req_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        with open(req_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"❌ JSON parse error in '{req_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw_json, dict):
        print(f"❌ Expected JSON object in '{req_path}', got {type(raw_json).__name__}", file=sys.stderr)
        sys.exit(1)

    try:
        ir = build_ir(raw_json)
    except IRBuilderError as exc:
        print(f"❌ Building IR validation error: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = cli_args.output
    if not out_path:
        sid = ir.get("session_id") or os.path.splitext(os.path.basename(req_path))[0]
        req_dir = os.path.dirname(req_path) or "outputs"
        out_path = os.path.join(req_dir, f"{sid}_ir.json")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2)

    print("✅ Building IR v1 constructed and saved successfully:")
    print(f"   Output: {out_path}")

