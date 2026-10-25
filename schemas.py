"""
schemas.py
==========
Shared Pydantic models and enumerations for the Requirements Extraction Agent.

This file is the single source of truth for the data contract shared between:
  - The requirements extractor / checklist
  - The Gemini client response parser
  - The completeness checker and validators
  - The Building IR module (future)

Design notes
------------
* ``Optional[X]`` fields represent "not yet supplied".  A missing value is
  ``None``; it is never silently defaulted here.
* ``FieldStatus`` and ``FieldSource`` track provenance so every value can be
  traced back to its origin.
* ``RoomSpec`` stores per-room requirements rather than a flat room count so
  that window / door counts remain anchored to the correct room.
* ``ExtractionResult`` is the final handoff object to Building IR.  It must
  only be produced after the completion policy passes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FieldStatus(str, Enum):
    """Lifecycle status of a single requirement field."""

    MISSING = "missing"
    PROVIDED = "provided"
    DEFAULTED = "defaulted"
    CONFIRMED = "confirmed"
    CONFLICT = "conflict"


class FieldSource(str, Enum):
    """Provenance of a field value."""

    USER_EXPLICIT = "user_explicit"
    USER_CONFIRMED_DEFAULT = "user_confirmed_default"
    SYSTEM_DEFAULT = "system_default"
    INFERRED = "inferred"


class BuildingType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    MIXED = "mixed"
    OTHER = "other"


class RelationshipType(str, Enum):
    ADJACENT_TO = "adjacent_to"
    CONNECTS_TO = "connects_to"
    SHARED_FACILITY = "shared_facility"
    OPPOSITE_TO = "opposite_to"


class SessionStatus(str, Enum):
    """Overall status of an extraction session turn."""

    NEEDS_CLARIFICATION = "needs_clarification"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    FINALIZED = "finalized"
    ERROR = "error"
    CONFLICT = "conflict"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RoomSpec(BaseModel):
    """Specification for one room or space in the building."""

    room_id: Optional[str] = Field(None, description="Stable unique identifier for the physical room, e.g. 'room_1', 'bathroom_1'.")
    room_type: str = Field(..., description="E.g. 'bedroom', 'bathroom', 'hall', 'kitchen'.")
    count: int = Field(1, ge=1, description="Number of rooms of this type (1 for canonical individual rooms).")
    window_count: Optional[int] = Field(None, ge=0, description="Windows per room; None = not specified.")
    door_count: Optional[int] = Field(None, ge=0, description="Doors per room; None = not specified.")
    width_m: Optional[float] = Field(None, gt=0, description="Room width in metres; None = not specified.")
    length_m: Optional[float] = Field(None, gt=0, description="Room length in metres; None = not specified.")
    notes: Optional[str] = Field(None, description="Free-form notes from the user about this room.")

    model_config = {"extra": "forbid"}


class RoomRelationship(BaseModel):
    """Spatial or functional relationship between two named rooms."""

    room_a: str = Field(..., description="First room type (e.g. 'kitchen').")
    room_b: str = Field(..., description="Second room type (e.g. 'hall').")
    relationship: RelationshipType = Field(..., description="Type of relationship.")
    notes: Optional[str] = None

    model_config = {"extra": "forbid"}


class Dimensions(BaseModel):
    """Overall building footprint dimensions."""

    width_m: Optional[float] = Field(None, gt=0)
    length_m: Optional[float] = Field(None, gt=0)
    floor_height_m: Optional[float] = Field(None, gt=0)
    wall_thickness_mm: Optional[float] = Field(None, gt=0)

    model_config = {"extra": "forbid"}


class FieldEntry(BaseModel):
    """
    Checklist entry for a single top-level requirement field.

    Tracks value, status, provenance, unit, and whether the field must be
    resolved before finalization is allowed.
    """

    value: Optional[Any] = None
    unit: Optional[str] = None
    status: FieldStatus = FieldStatus.MISSING
    source: Optional[FieldSource] = None
    user_confirmed: bool = False
    blocking: bool = True  # if True, finalization is blocked when missing/conflict
    notes: Optional[str] = None

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Candidate requirements (extracted by Gemini — may be incomplete)
# ---------------------------------------------------------------------------


class CandidateRequirements(BaseModel):
    """
    Partially-populated requirements object returned by Gemini.

    All fields are Optional because Gemini may not extract them from the
    initial prompt.  ``None`` always means "not mentioned" — never a default.

    This model is used only for parsing the LLM response; it is then merged
    into ``RequirementState`` by the agent orchestrator.
    """

    building_type: Optional[BuildingType] = None
    floor_count: Optional[int] = Field(None, ge=1, le=50)
    rooms: Optional[List[RoomSpec]] = None
    overall_dimensions: Optional[Dimensions] = None
    relationships: Optional[List[RoomRelationship]] = None
    units: Optional[str] = Field(None, description="Measurement unit system, e.g. 'mm', 'm', 'ft'.")
    constraints: Optional[List[str]] = Field(None, description="Explicit must/must-not constraints.")
    preferences: Optional[List[str]] = Field(None, description="Non-blocking preferences (roof style, materials).")
    ambiguities: Optional[List[str]] = Field(
        None,
        description="Notes from the model about values it could not determine confidently.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("floor_count")
    @classmethod
    def floor_count_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("floor_count must be at least 1.")
        return v


# ---------------------------------------------------------------------------
# Agent turn (returned to the caller / CLI on every interaction)
# ---------------------------------------------------------------------------


class AgentTurn(BaseModel):
    """
    The response returned by the agent after each interaction turn.

    The caller (CLI or UI) uses this to display messages to the user and
    decide the next action.
    """

    status: SessionStatus
    session_id: str
    clarification_round: int = Field(0, ge=0, le=3)
    message: str = Field(..., description="User-facing message: question, summary, or error.")
    pending_fields: List[str] = Field(default_factory=list, description="Field names still needing input.")
    requirements: Optional[Dict[str, Any]] = Field(
        None, description="Current candidate requirement values (for display)."
    )
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    is_finalized: bool = False

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Final extraction result (handoff to Building IR)
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """
    Validated, finalized requirements object handed off to the Building IR module.

    This is the only object the IR builder should accept.  It is produced only
    when the completion policy passes and the user has confirmed the summary.

    Schema versioning allows the IR module to detect incompatible changes.
    """

    schema_version: str = Field("1.0", description="Schema version for forward compatibility.")
    session_id: str
    building_type: BuildingType
    floor_count: int = Field(..., ge=1)
    units: str = Field(..., description="Canonical unit for all dimensions, e.g. 'mm'.")
    rooms: List[RoomSpec]
    overall_dimensions: Dimensions
    relationships: List[RoomRelationship] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    provenance: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps field name to FieldSource value for every field.",
    )
    defaults_applied: List[str] = Field(
        default_factory=list,
        description="Field names that received system defaults.",
    )
    warnings: List[str] = Field(default_factory=list)
    confirmed_by_user: bool = False

    model_config = {"extra": "forbid"}

    @field_validator("floor_count")
    @classmethod
    def single_floor_only(cls, v: int) -> int:
        """
        Prototype constraint: only single-floor buildings are supported.
        This guard ensures an invalid scope cannot reach the IR module.
        """
        if v != 1:
            raise ValueError(
                "The current prototype supports single-floor buildings only. "
                f"Received floor_count={v}."
            )
        return v

    @model_validator(mode="after")
    def rooms_not_empty(self) -> "ExtractionResult":
        if not self.rooms:
            raise ValueError("At least one room must be specified in the final requirements.")
        return self
