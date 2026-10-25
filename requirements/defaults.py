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
        value="mm",
        rationale="Millimetres are the standard unit for architectural drawings in this prototype.",
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
        rationale="Standard residential floor-to-ceiling height of 3 m.",
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
    for entry in DEFAULTS.values():
        if entry.room_type == room_type:
            if entry.sub_field in result:
                result[entry.sub_field] = entry.value
    return result


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
        Names of the default keys from DEFAULTS that were applied.
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
    return "\n".join(lines)
