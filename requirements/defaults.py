"""
requirements/defaults.py
========================
Central registry of documented system defaults.

Design principles
-----------------
* Defaults are prototype values, NOT architectural standards.
* Every default carries a rationale so users understand why it was chosen.
* Explicit user values always take precedence over defaults.
* Defaults marked ``requires_confirmation=True`` should be highlighted in the
  final summary and may need explicit user sign-off in stricter workflows.
* The registry is a plain dict for easy inspection, testing, and overriding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DefaultEntry:
    """A single default definition."""

    field: str
    value: Any
    unit: Optional[str] = None
    rationale: str = ""
    requires_confirmation: bool = False
    # For room-level defaults, ``room_type`` identifies which room(s) they apply to
    room_type: Optional[str] = None
    sub_field: Optional[str] = None  # e.g. "window_count", "door_count"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, DefaultEntry] = {
    # ----- Units -----
    "units": DefaultEntry(
        field="units",
        value="m",
        rationale="Metres are the canonical storage unit for linear dimensions in this prototype.",
        requires_confirmation=False,
    ),
    # ----- Building type -----
    # building_type has NO safe default — user must specify it.

    # ----- Floor count -----
    "floor_count": DefaultEntry(
        field="floor_count",
        value=1,
        rationale="The current prototype supports single-floor residential buildings only.",
        requires_confirmation=False,
    ),

    # ----- Dimensions -----
    "wall_thickness_m": DefaultEntry(
        field="wall_thickness_m",
        value=0.2,
        unit="m",
        rationale="Typical internal wall thickness of 0.2 m for prototype layout generation.",
        requires_confirmation=False,
    ),
    # Backward compatibility alias
    "wall_thickness_mm": DefaultEntry(
        field="wall_thickness_mm",
        value=200,
        unit="mm",
        rationale="Typical internal wall thickness for prototype layout generation.",
        requires_confirmation=False,
    ),
    "floor_height_m": DefaultEntry(
        field="floor_height_m",
        value=3.0,
        unit="m",
        rationale="Standard residential floor-to-ceiling height of 3.0 m.",
        requires_confirmation=False,
    ),

    # ----- Room dimension defaults (prototype demonstration values) -----
    # Bedroom: 3.0 m width x 3.5 m length
    "bedroom_width_m": DefaultEntry(
        field="width_m",
        value=3.0,
        unit="m",
        room_type="bedroom",
        sub_field="width_m",
        rationale="Prototype demonstration bedroom width: 3.0 m.",
        requires_confirmation=False,
    ),
    "bedroom_length_m": DefaultEntry(
        field="length_m",
        value=3.5,
        unit="m",
        room_type="bedroom",
        sub_field="length_m",
        rationale="Prototype demonstration bedroom length: 3.5 m.",
        requires_confirmation=False,
    ),
    # Bathroom: 1.8 m width x 2.4 m length
    "bathroom_width_m": DefaultEntry(
        field="width_m",
        value=1.8,
        unit="m",
        room_type="bathroom",
        sub_field="width_m",
        rationale="Prototype demonstration bathroom width: 1.8 m.",
        requires_confirmation=False,
    ),
    "bathroom_length_m": DefaultEntry(
        field="length_m",
        value=2.4,
        unit="m",
        room_type="bathroom",
        sub_field="length_m",
        rationale="Prototype demonstration bathroom length: 2.4 m.",
        requires_confirmation=False,
    ),
    # Hall: 3.5 m width x 4.0 m length
    "hall_width_m": DefaultEntry(
        field="width_m",
        value=3.5,
        unit="m",
        room_type="hall",
        sub_field="width_m",
        rationale="Prototype demonstration hall width: 3.5 m.",
        requires_confirmation=False,
    ),
    "hall_length_m": DefaultEntry(
        field="length_m",
        value=4.0,
        unit="m",
        room_type="hall",
        sub_field="length_m",
        rationale="Prototype demonstration hall length: 4.0 m.",
        requires_confirmation=False,
    ),

    # ----- Room opening defaults (applied per-room when not specified) -----
    # Bedroom
    "bedroom_window_count": DefaultEntry(
        field="window_count",
        value=1,
        room_type="bedroom",
        sub_field="window_count",
        rationale="Each bedroom defaults to one window for natural light and ventilation.",
        requires_confirmation=False,
    ),
    "bedroom_door_count": DefaultEntry(
        field="door_count",
        value=1,
        room_type="bedroom",
        sub_field="door_count",
        rationale="Each bedroom defaults to one door.",
        requires_confirmation=False,
    ),
    # Bathroom
    "bathroom_window_count": DefaultEntry(
        field="window_count",
        value=1,
        room_type="bathroom",
        sub_field="window_count",
        rationale="Bathroom defaults to one window for ventilation.",
        requires_confirmation=False,
    ),
    "bathroom_door_count": DefaultEntry(
        field="door_count",
        value=1,
        room_type="bathroom",
        sub_field="door_count",
        rationale="Bathroom defaults to one door.",
        requires_confirmation=False,
    ),
    # Hall / corridor
    "hall_window_count": DefaultEntry(
        field="window_count",
        value=0,
        room_type="hall",
        sub_field="window_count",
        rationale="Halls typically have no external windows in compact layouts.",
        requires_confirmation=False,
    ),
    "hall_door_count": DefaultEntry(
        field="door_count",
        value=1,
        room_type="hall",
        sub_field="door_count",
        rationale="Hall defaults to one main entrance door.",
        requires_confirmation=False,
    ),
    # Kitchen
    "kitchen_window_count": DefaultEntry(
        field="window_count",
        value=1,
        room_type="kitchen",
        sub_field="window_count",
        rationale="Kitchens default to one window for ventilation.",
        requires_confirmation=False,
    ),
    "kitchen_door_count": DefaultEntry(
        field="door_count",
        value=1,
        room_type="kitchen",
        sub_field="door_count",
        rationale="Kitchen defaults to one door.",
        requires_confirmation=False,
    ),
    # Living room
    "living_room_window_count": DefaultEntry(
        field="window_count",
        value=2,
        room_type="living_room",
        sub_field="window_count",
        rationale="Living rooms default to two windows for natural light.",
        requires_confirmation=False,
    ),
    "living_room_door_count": DefaultEntry(
        field="door_count",
        value=1,
        room_type="living_room",
        sub_field="door_count",
        rationale="Living room defaults to one door.",
        requires_confirmation=False,
    ),
}

# Fields that have no safe default and MUST be provided by the user
NO_DEFAULT_FIELDS = {"building_type"}


# ---------------------------------------------------------------------------
# Accessor helpers
# ---------------------------------------------------------------------------


def get_scalar_default(field: str) -> Optional[DefaultEntry]:
    """Return the default entry for a top-level scalar field, or None."""
    return DEFAULTS.get(field)


def get_room_opening_defaults(room_type: str) -> Dict[str, Optional[int]]:
    """
    Return opening defaults (window_count, door_count) for a given room type.

    Returns a dict with keys ``window_count`` and ``door_count``; values are
    ``None`` when no default is configured for that room type.
    """
    result: Dict[str, Optional[int]] = {"window_count": None, "door_count": None}
    clean = room_type.strip().lower().replace("-", "_").replace(" ", "_")
    for entry in DEFAULTS.values():
        if entry.room_type == clean:
            if entry.sub_field in result:
                result[entry.sub_field] = entry.value
    return result


def get_room_dimension_defaults(room_type: str) -> Dict[str, Optional[float]]:
    """
    Return prototype dimension defaults (width_m, length_m) for a given room type.

    Returns a dict with keys ``width_m`` and ``length_m``.
    Values are ``None`` for unknown room types (arbitrary dimensions are never assigned).
    """
    result: Dict[str, Optional[float]] = {"width_m": None, "length_m": None}
    clean = room_type.strip().lower().replace("-", "_").replace(" ", "_")
    for entry in DEFAULTS.values():
        if entry.room_type == clean and entry.sub_field in result:
            result[entry.sub_field] = entry.value
    return result


def propose_residential_relationship_defaults(
    rooms: List[Any],
    existing_relationships: List[Any],
    constraints: Optional[List[str]] = None,
) -> List[Any]:
    """
    Propose prototype connections from the hall to bedrooms and bathrooms
    for eligible residential buildings.

    Rules
    -----
    - Applies only when a hall and at least one bedroom or bathroom exist.
    - Preserves explicit positive and negative relationship requirements.
    - Prohibits proposing connections that violate negative constraints.
    - Marks source as FieldSource.SYSTEM_DEFAULT.
    """
    from schemas import FieldSource, RelationshipType, RoomRelationship

    # Find hall(s)
    halls = [r for r in rooms if getattr(r, "room_type", "") == "hall"]
    if not halls:
        return []
    main_hall = halls[0]
    hall_id = getattr(main_hall, "room_id", None) or "hall_1"

    # Targets: bedrooms and bathrooms
    targets = [
        r for r in rooms
        if getattr(r, "room_type", "") in ("bedroom", "bathroom")
    ]
    if not targets:
        return []

    # Identify existing connected pairs and negative constraints
    existing_connected = set()
    negative_pairs = set()

    for rel in existing_relationships:
        a = rel.room_a.lower()
        b = rel.room_b.lower()
        pair = tuple(sorted([a, b]))
        if rel.relationship == RelationshipType.CONNECTS_TO:
            if getattr(rel, "negative", False):
                negative_pairs.add(pair)
            else:
                existing_connected.add(pair)
        elif getattr(rel, "negative", False):
            negative_pairs.add(pair)

    # Check textual negative constraints
    if constraints:
        for c in constraints:
            c_low = c.lower()
            if any(w in c_low for w in ("not connect", "cannot connect", "no connection", "must not connect")):
                for r in targets:
                    r_id = (getattr(r, "room_id", "") or getattr(r, "room_type", "")).lower()
                    h_id_low = hall_id.lower()
                    if (r_id in c_low or getattr(r, "room_type", "") in c_low) and (h_id_low in c_low or "hall" in c_low):
                        negative_pairs.add(tuple(sorted([h_id_low, r_id])))

    proposed: List[RoomRelationship] = []
    for r in targets:
        r_id = getattr(r, "room_id", None) or getattr(r, "room_type", "room")
        pair = tuple(sorted([hall_id.lower(), r_id.lower()]))
        if pair not in existing_connected and pair not in negative_pairs:
            proposed.append(
                RoomRelationship(
                    room_a=hall_id,
                    room_b=r_id,
                    relationship=RelationshipType.CONNECTS_TO,
                    notes="Prototype default: hall connects to bedrooms and bathroom.",
                    source=FieldSource.SYSTEM_DEFAULT,
                )
            )

    return proposed


def get_all_defaultable_scalar_fields() -> Dict[str, DefaultEntry]:
    """Return all scalar (non-room-level) default entries."""
    return {
        k: v for k, v in DEFAULTS.items() if v.room_type is None
    }


def format_defaults_summary(applied_fields: list[str]) -> str:
    """
    Build a human-readable summary of the defaults that were applied.

    Parameters
    ----------
    applied_fields : list[str]
        Names of the default keys from DEFAULTS or custom keys that were applied.
    """
    if not applied_fields:
        return "No system defaults were applied."
    lines = ["System defaults applied:"]
    for key in applied_fields:
        entry = DEFAULTS.get(key)
        if entry:
            unit_str = f" {entry.unit}" if entry.unit else ""
            lines.append(
                f"  • {entry.field}"
                + (f" ({entry.room_type})" if entry.room_type else "")
                + f": {entry.value}{unit_str} — {entry.rationale}"
            )
        else:
            lines.append(f"  • {key}")
    return "\n".join(lines)
