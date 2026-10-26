# Building IR v1 — IR Builder Software Development Specification

Status: Implementation specification | Scope: IR construction only

Purpose: define how the IR Builder transforms the existing Requirements Extraction Agent JSON into a versioned, JSON-serialized building graph. This document is intended to be given to Antigravity before implementation.


## 1. Objective and scope

Implement the IR Builder as a deterministic Python component that accepts the candidate requirements JSON produced by the existing application and returns a canonical Building IR v1 document. The document represents a graph using authoritative entity collections and ID-referenced edges, with separate requirements and layout layers.

- In scope: input mapping, canonical IDs/references, structural validation, relationship conversion, constraint preservation, provenance mapping, and JSON-serializable output.

- Out of scope: LLM prompting or extraction, interactive clarification, applying new defaults, resolving ambiguous requirements, layout search, CP-SAT/slicing, geometric feasibility, wall placement, CAD/mesh generation.

- Inspect and reuse the existing schemas.py, IR_builder.py stub, and current CandidateRequirements/output schema. Do not redesign the Requirements Extraction Agent.

- Do not silently drop unknown or unsupported constraints. Preserve them in a representable form and mark their evaluation state as not_evaluated or unsupported where applicable.


## 2. Architectural position

1. Requirements Extraction Agent: converts natural language into candidate requirements JSON.

1. IR Builder: maps candidate JSON into canonical Building IR and checks structural correctness.

1. Requirements Resolution / Pre-layout Validation (separate future component): applies approved defaults, computes deterministic derived values, checks requirement consistency, and classifies unresolved information.

1. Layout Engine (future): computes room placements, walls, actual openings, footprint, and spatial relationships.

1. Geometric Verifier (future): verifies the candidate layout against requirements and geometry rules.

1. Geometry Backend (future): produces 3D geometry from a verified layout.

The IR Builder must not claim that a layout is feasible. Structural validity of the IR is different from requirement consistency, geometric feasibility, and final geometry validity.


## 3. Input contract

Input is the JSON-compatible CandidateRequirements object produced by the current Requirements Extraction Agent/orchestrator. The exact source schema must be discovered from existing code and treated as the input contract; do not invent a replacement schema.

- Expected concepts may include building_type, floor_count, units, rooms, relationships, constraints, preferences, provenance, defaults_applied, warnings, and session/confirmation metadata where present.

- Room fields may include room_id, room_type, count, width_m, length_m, window_count, door_count, and notes.

- Relationship fields may include room_a, room_b, relationship, negative, notes, and source.

- Overall dimensions may be absent or null. Preserve unknown overall width/length as null.

- Handle optional fields and nulls according to the existing schema. Do not infer missing values from examples.

- Accept the input type expected by the current code. Avoid unnecessary JSON serialize/deserialize round-trips.


## 4. Output contract

Return a JSON-serializable mapping matching the IR v1 structure. Follow the existing calling convention if it already specifies a return type. Output must be deterministic for equivalent input and configuration.

{
  "ir_version": "1.0",
  "session_id": "...",
  "units": "m",
  "building": {},
  "requirements": {
    "rooms": [],
    "edges": [],
    "constraints": [],
    "preferences": [],
    "unresolved": []
  },
  "graph": {"node_refs": [], "edge_refs": []},
  "layout": {
    "status": "not_started",
    "room_placements": [],
    "walls": [],
    "openings": [],
    "footprint": null,
    "edges": [],
    "validation_results": []
  },
  "validation": {
    "pre_layout": {"status": "not_run", "results": []},
    "layout": {"status": "not_run", "results": []},
    "geometry": {"status": "not_run", "results": []}
  },
  "provenance": [],
  "assumptions": []
}

This is the required top-level shape. The precise fields inside building, room, edge, constraint, and validation records must follow the rules below. Do not duplicate full node or edge records under graph.


## 5. Entity and node model

The graph index contains IDs, not copies of entity objects. graph.node_refs is derived from the authoritative building and room records. Opening requirements are not graph nodes in v1 unless the schema is explicitly extended.


## 6. Edge model and relationship rules

- Requirements edges live in requirements.edges; realized spatial edges live in layout.edges.

- Each edge has a stable id, from, to, type, source, and optional notes/negative metadata where supported.

- Every endpoint must reference a valid entity or explicitly permitted special endpoint.

- Preserve different edge types between the same endpoints. Deduplicate only exact duplicates.

- For symmetric types, canonicalize endpoint order for duplicate detection.

- Never add connects_to because rooms are adjacent, share a wall, or have default door counts.

- Do not invent minimum shared-wall lengths. Preserve user-provided values; otherwise omit or null them according to schema.


## 7. Constraint model

Constraints are structured records. Preserve supported and unsupported types. The builder maps and structurally validates them; it does not solve them or claim they are satisfied.

- Common fields: id, scope, target, type, property (optional), operator, value/parameters, unit where applicable, source, status, and optional message.

- Evaluation statuses: not_evaluated, satisfied, violated, unsupported, unresolved. Builder normally initializes constraints to not_evaluated unless structural invalidity is identified.

- Never mark a geometric constraint satisfied merely because it was extracted or stored.

- If input constraints are plain text, preserve the text in a descriptive/unresolved representation rather than fabricating operators or numeric values.


## 8. Dimensions, areas, defaults, unresolved data

- Canonical linear units are metres. Convert only when source units are known and conversion is deterministic. Never guess units.

- Keep width_m and length_m separate; do not swap them based on display order.

- Keep target_area_sqm separate from area_from_dimensions_sqm and computed_area_sqm.

- The builder must not infer missing width/length or target area, and must not compute the overall footprint.

- Applying defaults and resolving derived values belong to a separate Requirements Resolution / Pre-layout Validation component. The builder preserves existing values and provenance, and represents absent values as null/unresolved according to schema.

- If the Requirements Agent already applied defaults, preserve those values and system_default provenance; do not apply defaults a second time.

- Do not replace unknown values with zero unless zero is explicit or valid for that field.

- Preserve unresolved items and warnings where present; do not silently discard ambiguity or contradictory requirements.


## 9. Provenance and assumptions

- user_explicit — directly stated or explicitly confirmed by the user.

- system_default — supplied by a documented default policy.

- derived — calculated deterministically from known inputs by a named method.

- unresolved — missing or ambiguous information not yet resolved.

Provenance records should identify a target path and source, with optional method, input paths and origin metadata. Map existing provenance where valid; do not blindly trust arbitrary LLM provenance labels. Preserve traceability to source requirements. Assumptions are human-readable supporting information, not authoritative machine rules. Do not convert an assumption into a hard constraint unless the input or a documented policy explicitly does so.


## 10. Provided and required interfaces


### Input interface

build_ir(candidate_requirements, *, session_id=None) -> IR mapping

Use the exact function name/signature already defined by IR_builder.py and its call sites if present. The builder receives candidate requirements and optional session metadata; it must not call the LLM.


### Output interface

- Return a JSON-serializable mapping matching the IR v1 structure.

- The orchestrator/application owns file writing unless the existing architecture explicitly assigns persistence to the builder.

- Preserve the existing caller contract or document a minimal compatible change.

- Return data to the caller; printing JSON must not be the only interface.

- Errors must distinguish malformed input, invalid references, and unsupported representations and be actionable.


### Provided by existing components

- CandidateRequirements JSON from the Requirements Extraction Agent.

- Existing schemas.py data models/enums and validation utilities.

- Session ID and available session metadata from the orchestrator.

- Existing defaults/provenance already applied upstream, if present.


### Downstream consumers

- Orchestrator/application: receives IR and serializes/saves it.

- Future resolution/pre-layout validator: applies approved defaults, computes deterministic values, checks consistency.

- Future layout engine: consumes requirements, edges and constraints, writes layout results.

- Future geometric verifier: checks candidate layout and writes validation results.


## 11. Validation and error handling

- Validate top-level structure and required fields against the agreed schema.

- Ensure IDs are unique within their namespace and generated IDs are deterministic where possible.

- Ensure edge and constraint references resolve to valid entities/edges.

- Validate enum values and field types without dropping unknown constraints.

- Dimensions must be numeric, finite, and positive when present; counts must be integers and non-negative when present.

- Preserve null for unknown dimensions and footprint values.

- Do not repair contradictions silently. Return a clear validation error or represent the issue as unresolved, following existing application conventions.

- Do not mark layout or geometry validation as passed; initialize their statuses to not_run.

- Output must be deterministic for equivalent input and configuration.


## 12. Non-functional requirements

- Provider-independent: no Gemini- or LM Studio-specific logic.

- Deterministic: no LLM calls, random assumptions, network access or hidden inference.

- Maintainable: separate mapping and validation functions for entities, edges, constraints and provenance.

- Compatible: reuse existing models and minimize unrelated refactoring.

- Traceable: preserve session ID, provenance and source references.

- Extensible: new types should be addable without changing existing meanings; schema changes must be versioned explicitly.

- JSON-safe: no non-serializable objects in the returned mapping.


## 13. Acceptance criteria for code review

1. Builder accepts the actual CandidateRequirements structure used by the application.

1. Output has the agreed top-level IR structure and is JSON serializable.

1. Building and room records are stored once; graph indexes and edges reference IDs.

1. Explicit dimensions and relationships are preserved with correct meanings and provenance.

1. No automatic connects_to edges, coordinates, walls, placed openings or footprint are invented.

1. Different edge types between the same rooms are retained; exact duplicates are handled deterministically.

1. Unknown and unsupported constraints are preserved and not marked satisfied.

1. Defaults are not applied twice; existing system_default provenance is retained.

1. Builder does not implement resolution, layout solving, geometric verification or CAD generation.

1. Existing application interface remains compatible, or necessary interface changes are documented.


## 14. Instructions for implementation

- Inspect IR_builder.py, schemas.py, orchestrator.py, the actual requirements output schema and current saved output JSON before editing.

- Treat existing code as authoritative where this document leaves a detail unspecified. Report conflicts instead of silently inventing a new contract.

- Implement only the IR Builder scope. Do not implement the separate resolver, layout engine, CP-SAT, geometric verifier or geometry backend.

- Do not create or run tests, launch the application, or claim tests have passed. The project owner will perform manual verification.

- After changes, summarize files changed, public interface, assumptions made and unresolved design conflicts.


## Appendix A — Example requirements edge

{
  "id": "re_001",
  "from": "r_hall1",
  "to": "r_bath1",
  "type": "shares_wall_with",
  "source": "user_explicit",
  "status": "not_evaluated"
}

This edge records a requirement; it does not prove a shared wall exists. A future layout edge should represent the realized relationship and link to the requirement edge where possible.



| Entity / node type | v1 handling | Required behaviour |

| --- | --- | --- |

| building | Authoritative building record | Stable id; preserve building type, floor count and known building-level properties. |

| room | Authoritative records under requirements.rooms | Stable unique id; type, label, dimensions, area fields, opening requirements and notes where available. |

| opening requirement | Nested under its room initially | Stable ID, kind, requested count and source. Not a placed opening. |

| wall | Layout layer only | Empty before layout; populated only by a future layout engine. |

| floor | Deferred | Current scope is single-floor. Do not create floor nodes or a floors array in v1. |

| exterior | Special endpoint only if needed | Do not create a fake room entity to represent exterior; define permitted special endpoints explicitly. |





| Type | Meaning | Direction / rule |

| --- | --- | --- |

| adjacent_to | Required adjacency between rooms | Symmetric; does not imply a door or physical passage. |

| shares_wall_with | Required shared wall between rooms | Symmetric; does not imply a door or passage. |

| connects_to | Required physical access connection | Symmetric room-access relation in v1; never add automatically. |

| opens_into | Opening leads into a room/space | Directed; future use for opening objects. |

| belongs_to | Entity membership | Directed; use only when needed. |

| realized_adjacency | Adjacency in a candidate layout | Layout layer only; reference the requirement edge where possible. |

| realized_shared_wall | Actual shared wall between placed rooms | Layout layer only; link to wall geometry and requirement edge where possible. |





| Constraint type | Examples | Builder responsibility |

| --- | --- | --- |

| dimension | Width equals 3 m; length at least 4 m | Preserve target/property/operator/value/unit and validate field shape. |

| area | Target area; permitted area range | Keep target area separate from derived/calculated area. |

| topology | Must/must-not be adjacent | Preserve relationship semantics and references. |

| wall | Minimum shared-wall length | Preserve value and unit; no geometry verification here. |

| opening | Required number/kind of openings | Preserve requested requirements; do not create placed openings. |

| exterior | Must/must-not touch exterior | Preserve as constraint; no spatial claim. |

| clearance | Minimum clearance | Preserve even if not yet supported. |

| footprint | Maximum building width/length | Preserve; do not calculate footprint. |

