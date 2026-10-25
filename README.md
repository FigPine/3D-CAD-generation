# Requirements Extraction Agent

> **Module 1** of the AI-Assisted Parametric 3D Building Model Generation project.

## What it does

Converts a free-form natural-language building description into a **structured, validated requirements object** that the future Building IR module can consume.

```
User text  →  Requirements Extraction Agent  →  ExtractionResult (JSON)
              (Gemini extracts NL meaning;               ↓
               Python owns state + validation)     Building IR (future)
```

### Key features

| Feature | Detail |
|---|---|
| NL extraction | Gemini parses the user's description into structured fields |
| Stateful checklist | Python tracks value, status, and provenance for every field |
| Interactive clarification | Up to **3 rounds** of focused Q&A to fill missing fields |
| Default registry | Documented fallback values applied after round 3 or on "use defaults" |
| Conflict detection | Contradictory explicit requirements are flagged, never silently resolved |
| Final confirmation | User reviews a readable summary before the result is produced |
| JSON output | Validated `ExtractionResult` saved to `outputs/<session-id>.json` |
| No real API key for tests | All tests mock the Gemini client |

---

## Project structure

```
3D-CAD-generation/
├── app.py                          # CLI entry point
├── schemas.py                      # Shared Pydantic models and enums (contract)
├── IR_builder.py                   # Placeholder for future Building IR module
├── agents/
│   └── req_ext_agent.py            # Orchestrator (session API)
├── llm/
│   ├── gemini_client.py            # Gemini SDK adapter (isolated here)
│   └── prompts.py                  # Versioned prompt templates
├── requirements/
│   ├── requirement_state.py        # Mutable session state + merge logic
│   ├── completeness_checker.py     # Deterministic completeness rules
│   ├── defaults.py                 # Central defaults registry
│   └── dialogue_manager.py        # Round counting, question generation, summaries
├── validators/
│   └── requirements_validator.py  # Domain-level validation gate
├── tests/
│   ├── conftest.py                 # Shared fixtures (mock Gemini)
│   ├── test_extraction_schema.py   # Pydantic schema tests
│   ├── test_completeness.py        # Checklist + state merge tests
│   ├── test_defaults.py            # Defaults registry + application tests
│   ├── test_dialogue_round_limit.py# Round limit + question generation tests
│   └── test_req_ext_agent.py       # Integration tests for the orchestrator
├── examples/
│   └── sample_prompts.json         # Labelled test prompts
├── outputs/                        # Finalized requirement JSON files (gitignored)
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Setup

### 1. Python version

Python **3.11** or later is required (uses `from __future__ import annotations` and `match`/`case`-compatible type hints).

### 2. Create a virtual environment

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

Or set the environment variable directly:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"
```

```bash
# macOS / Linux
export GEMINI_API_KEY="your_key_here"
```

> **Never** commit `.env` or hardcode your key. The `.gitignore` already excludes `.env`.

---

## Running the CLI

```bash
python app.py
```

**Example session:**

```
============================================================
  AI-Assisted 3D Building Model Generator
  Requirements Extraction Agent — v1.0
============================================================

Describe your building: Build a single-floor house with two bedrooms, one bathroom, and a hall.
                        Each bedroom should have two windows.

⏳  Extracting requirements (calling Gemini)…

------------------------------------------------------------
============================================================
  REQUIREMENTS SUMMARY
============================================================

User-supplied requirements:
  Building type: residential
  Floor count: 1
  2× bedroom (2 window(s), 1 door(s))
  1× bathroom (1 window(s), 1 door(s))
  1× hall (0 window(s), 1 door(s))

System defaults applied:
  Units: mm

============================================================
Please review the above and reply 'confirm' to finalize,
or describe any changes you'd like to make.
============================================================

Your response (type 'confirm' to finalize): confirm

✅  Session complete.
    Session ID : 3f8a1b2c-...
    Output     : outputs/3f8a1b2c-....json
```

---

## Running the tests

```bash
pytest tests/ -v
```

Tests do **not** require a real Gemini API key. The Gemini client is mocked in all test fixtures.

Expected output: all tests pass (≈ 40+ assertions across 5 test files).

---

## Output schema

The finalized `ExtractionResult` is a Pydantic model serialized to JSON:

```json
{
  "schema_version": "1.0",
  "session_id": "3f8a1b2c-...",
  "building_type": "residential",
  "floor_count": 1,
  "units": "mm",
  "rooms": [
    {"room_type": "bedroom", "count": 2, "window_count": 2, "door_count": 1, ...},
    {"room_type": "bathroom", "count": 1, "window_count": 1, "door_count": 1, ...},
    {"room_type": "hall", "count": 1, "window_count": 0, "door_count": 1, ...}
  ],
  "overall_dimensions": {"wall_thickness_mm": 200, "floor_height_m": 3.0, ...},
  "relationships": [],
  "constraints": [],
  "preferences": [],
  "provenance": {
    "building_type": "user_explicit",
    "floor_count": "user_explicit",
    "units": "system_default"
  },
  "defaults_applied": ["units"],
  "warnings": [],
  "confirmed_by_user": true
}
```

---

## Defaults policy

Defaults are prototype values — **not architectural standards**.

| Field | Default | Rationale |
|---|---|---|
| `units` | `mm` | Standard unit for architectural drawings |
| `floor_count` | `1` | Prototype supports single-floor only |
| `wall_thickness_mm` | `200 mm` | Typical internal wall |
| `floor_height_m` | `3.0 m` | Standard residential ceiling height |
| `bedroom` windows | `1` | Natural light / ventilation |
| `bathroom` windows | `1` | Ventilation |
| `hall` windows | `0` | Compact layout |
| `kitchen` windows | `1` | Ventilation |

- Defaults are applied **after round 3** to eligible missing fields.
- Saying **"use defaults"** applies them immediately.
- Defaults **never** override explicit user values.
- Defaults **never** resolve conflicting explicit constraints.
- Every applied default is recorded in `provenance` and `defaults_applied`.

---

## Clarification round policy

| Stage | Action |
|---|---|
| After initial extraction | If complete → summary. If not → start clarification. |
| Rounds 1–2 | Ask about missing blocking fields. Accept values or "use defaults". |
| Round 3 | Final clarification questions for remaining important fields. |
| After round 3 | Apply documented defaults to eligible missing fields; flag assumptions. |
| Final summary | Show user-supplied vs. defaulted; ask for confirmation. |

One round = one agent question + one user reply. The initial extraction does not count.

---

## Conflict resolution policy

If two **explicit user** values contradict each other for the same field:
1. The field is marked `CONFLICT` status.
2. A conflict message is added to `state.conflicts`.
3. The agent surfaces the conflict and asks the user to resolve it.
4. Defaults are **not** used to resolve conflicts.
5. Finalization is **blocked** until all conflicts are resolved.

---

## Architecture decisions

### LLM vs. Python separation

| Task | Owner |
|---|---|
| Parsing natural language | Gemini |
| Checklist state, round counting | Python |
| Completeness check | Python |
| Default application | Python |
| Conflict detection | Python |
| Final validation | Python |
| JSON packaging | Python |

Gemini is called at most twice per turn (initial extraction + optional follow-up interpretation). It is never called for checklist evaluation or simple structured replies.

### Session persistence

Sessions are stored in an in-memory dict for the first prototype. They survive for the lifetime of the process. JSON persistence (`outputs/<session-id>.json`) is written on finalization. A future version can replace the in-memory store with SQLite or Redis without changing the public API.

### Provider isolation

All Gemini SDK calls are in `llm/gemini_client.py`. Swapping providers requires only replacing that file and updating `requirements.txt`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LLMAuthError: GEMINI_API_KEY is not set` | Set `GEMINI_API_KEY` in `.env` or environment |
| `LLMRateLimitError` | Wait and retry; the client uses exponential backoff automatically |
| `ValidationError: floor_count=2` | Only single-floor buildings are supported in v1 |
| Tests fail with `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| `Session 'xyz' not found` | Sessions are in-memory; start a new session after restarting the process |

---

## What is out of scope (future modules)

- Building Intermediate Representation (Building IR) builder
- Geometric verification and repair
- CAD compiler (OpenSCAD / CadQuery / Blender)
- Floor-plan image interpretation
- Multi-floor buildings
- Web frontend
