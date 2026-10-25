**AI-Assisted Parametric 3D Model Generation**

**Implementation Plan — Requirements Extraction Agent**

**Module 1 of the project pipeline  
**Version 1.0 | LLM provider assumption: Gemini API

Purpose: define module scope, interfaces, workflow, checklist, feedback loop, file structure, and implementation tests.

# 1. Overall Project Idea

The project aims to build an AI-assisted framework that converts natural-language building requirements (and, in later stages, potentially plan or elevation images) into editable parametric 3D building models. Rather than sending a prompt directly to a CAD tool and relying on one-shot code generation, the system extracts explicit requirements, tracks assumptions and relationships in a structured Building Intermediate Representation (Building IR), and uses deterministic software components to validate and transform that representation into CAD output. Later modules are intended to support dependency-aware localized repair, deterministic geometric verification, and compilation to one or more selected backends such as OpenSCAD, CadQuery, build123d, or Blender. The first implementation milestone is deliberately narrower: build a reliable interactive Requirements Extraction Agent that produces a complete, validated requirements object for the Building IR module.

# 2. Module Summary

The Requirements Extraction Agent converts a user's free-form description into a structured set of building requirements. It must also manage a short clarification dialogue: identify missing or contradictory details, ask focused questions, accept user answers, offer documented defaults, and stop after at most three clarification rounds. A Python checklist maintains state between turns so the entire requirement set does not need to be resent to the LLM each time. Once the required fields are complete and conflicts are resolved—or eligible missing fields have been defaulted after the round limit—the module validates and presents a final summary for user confirmation. It then returns a typed result to the Building IR module.

- In scope: text prompt extraction, requirement schema, completeness checklist, clarification state, default management, validation, Gemini API adapter, finalization, and tests.
- Out of scope for this module: interpreting floor-plan images, assigning precise room coordinates, generating walls or meshes, CAD compilation, geometric verification, and repair of an existing CAD model.
- Core principle: the LLM interprets language; deterministic Python code owns state, checklist decisions, validation gates, round counting, and final output packaging.

# 3. Interfaces

## 3.1 Provided interface (what this module exposes)

| **Interface**                         | **Purpose**                                                   | **Expected contract**                                                                            |
| ------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| start_session(prompt)                 | Starts a new extraction session from the initial user prompt. | Returns a session state and either a clarification question or a candidate requirements summary. |
| process_user_reply(session_id, reply) | Merges a user's answer into the existing session.             | Returns the next question/summary or a finalized result; does not reset previous answers.        |
| get_session_state(session_id)         | Retrieves current checklist and round count for UI/testing.   | Returns serializable state without exposing secrets.                                             |
| finalize_requirements(session_id)     | Final validation and handoff to IR.                           | Returns ExtractionResult only when the completion policy passes.                                 |

Recommended primary Python API:

```
start_session(prompt: str) -> AgentTurn
process_user_reply(session_id: str, reply: str) -> AgentTurn
```

An AgentTurn should identify the current status, user-facing message, pending questions, checklist summary, and (only when finalized) the structured requirements. Use stable session IDs and do not rely on in-memory state alone if sessions must survive application restarts.

## 3.2 Required interfaces (what this module depends on)

| **Dependency**         | **Used for**                                                 | **Design note**                                                                  |
| ---------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| GeminiClient           | Sends extraction or interpretation requests to Gemini API.   | Keep provider-specific code isolated so the model/provider can be swapped later. |
| Requirement schema     | Defines valid fields and types.                              | Use Pydantic models; distinguish omitted values from explicit values.            |
| Completeness checker   | Determines missing required fields and unresolved conflicts. | Deterministic rules; no LLM call needed for routine checklist evaluation.        |
| Default registry       | Supplies documented fallback values.                         | Defaults must be configurable, unit-aware, and marked as system assumptions.     |
| Dialogue/session store | Preserves checklist, conversation round count, and answers.  | A Python object or JSON file is sufficient for the first prototype.              |
| Application/UI         | Displays questions and submits replies.                      | Can begin as a command-line interface.                                           |

# 4. Inputs and Outputs

## 4.1 Inputs

- Initial user prompt: free-form natural language such as "Build a floor plan with two bedrooms."
- Follow-up replies: short values or free-form corrections, such as "one bathroom," "use defaults," or "make the kitchen adjacent to the hall."
- Optional session context: current checklist, prior answers, pending questions, and clarification-round count.
- Configuration: supported building scope, required fields, default values, units, and maximum clarification rounds (fixed to 3 for this version).

## 4.2 Outputs

- AgentTurn: user-facing question or summary, status, pending fields, round number, and whether the session is awaiting a reply.
- ExtractionResult: finalized structured requirements, field provenance, warnings, schema version, and validation status.
- Structured errors: LLM/API failure, invalid response, unresolved contradiction, or invalid input. Errors must not be silently converted into a successful result.

```
{
  "status": "needs_clarification",
  "session_id": "session-...",
  "clarification_round": 1,
  "message": "How many bathrooms are required? I can use documented defaults for dimensions and openings.",
  "pending_fields": ["bathroom_count", "overall_dimensions", "opening_counts"],
  "requirements": { "...": "current candidate values" },
  "errors": [],
  "warnings": []
}
```

The example is illustrative. Final field names and enumerations should be defined once in schemas.py and reused by the extractor, checklist, validators, tests, and Building IR module.

# 5. Functional Requirements

1. Extract building type, floor count, room types/counts, per-room opening requirements, dimensions and units when stated, and semantic relationships when stated.
2. Preserve explicit user constraints and distinguish them from inferred values and system defaults.
3. Identify ambiguous, missing, malformed, or contradictory information.
4. Maintain a persistent-in-session checklist and merge new answers without losing previously captured values.
5. Ask no more than three clarification rounds. One round means one assistant question/message followed by the user's reply; the initial prompt is not a clarification round.
6. If required fields remain missing after the third round, fill only eligible fields using documented defaults, then report those assumptions.
7. Do not silently override contradictory explicit user constraints with defaults. Apply a documented conflict policy and expose unresolved conflicts.
8. Present a concise final requirements summary and request user confirmation before handoff.
9. Produce a validated, versioned JSON-compatible requirements object for the Building IR builder.
10. Handle API timeouts, invalid JSON/schema responses, empty user replies, and retryable failures safely.

# 6. Working / Processing Flow

11. Receive the initial prompt and create a new session with clarification_round = 0.
12. Call Gemini once to extract a candidate structured requirement object according to the schema. Request structured JSON output where supported, but never assume the model response is automatically valid.
13. Parse the response into Pydantic models. Reject or repair malformed output using a bounded retry policy; do not loop indefinitely.
14. Populate the checklist from the parsed candidate. Each field tracks value, status, source/provenance, and whether it blocks finalization.
15. Run deterministic validation and completeness checks. Detect missing blocking fields, invalid counts/units, unknown relationship references, and conflicts.
16. If complete and conflict-free, prepare a readable summary and ask for confirmation. If the user changes it, update only affected fields and revalidate.
17. If incomplete and fewer than three clarification rounds have been used, ask a focused set of questions for the highest-priority missing fields. Prefer deterministic question templates; call Gemini only when free-form answer interpretation is necessary.
18. When the user replies, increment the clarification round once, parse the answer, update the checklist, and rerun deterministic checks. Preserve all earlier valid answers.
19. After round three, apply configured defaults to remaining eligible missing fields. Record every default and its source. Do not use defaults to conceal contradictions.
20. Run final validation, show a summary that distinguishes user-provided values from defaults, and ask for confirmation. On confirmation, emit ExtractionResult for the IR builder.
21. If the user declines or corrects the summary, update the state and revalidate. The three-round limit applies to requirement-clarification rounds; confirmation/correction handling should be defined explicitly in the UI policy so it cannot become an unlimited hidden clarification loop.

# 7. Feedback Loop and Round-Limit Policy

A clarification round is one assistant message containing focused questions, followed by one user response. The initial extraction does not count as a round. The maximum is three rounds. Ask only questions needed to resolve blocking fields or important ambiguities; combine related questions into a compact message rather than asking one field at a time.

| **Point in flow**        | **Action**                                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| After initial extraction | Evaluate checklist. If complete, skip clarification and move to summary/confirmation.                            |
| Round 1 or 2             | Ask about missing blocking fields and key choices. Accept explicit values, "use defaults," or free-form answers. |
| Round 3                  | Ask final clarification questions for remaining important fields.                                                |
| After round 3            | Apply documented defaults to eligible missing fields; retain explicit user values; flag assumptions.             |
| Final summary            | Show requirements and mark which fields came from the user versus defaults; ask user to confirm.                 |

Recommended policy: a user explicitly saying "use defaults" can immediately authorize defaults for the fields covered by that choice, without waiting for round three. The three-round fallback is for fields that remain unspecified. Never invent values for fields that the current system cannot safely default; return a clear blocking error or ask for explicit confirmation.

# 8. Checklist Design

The checklist is application state, not a prompt that must be resent to Gemini on every turn. A Python object holds the current state; only a compact subset of that state is sent to Gemini when natural-language interpretation is actually needed.

| **Checklist field group** | **Example fields**                                                    | **Suggested handling**                                                                                                        |
| ------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Core building scope       | building_type, floors                                                 | Blocking if unsupported or absent; a single-floor-only prototype may default floors to 1 if this is explicitly part of scope. |
| Room programme            | room types, room counts                                               | Usually blocking unless the user authorizes a standard/default layout.                                                        |
| Openings                  | doors/windows per room                                                | Ask or use documented defaults; track per-room versus whole-building scope.                                                   |
| Dimensions and units      | overall width/length, room sizes, wall thickness, floor height, units | Ask or default if needed by the next stage; preserve units and source.                                                        |
| Relationships             | adjacent_to, connects_to, shared facility                             | Capture only stated relationships; distinguish adjacency from passage/connectivity.                                           |
| Constraints               | must/must_not conditions, explicit exclusions                         | Blocking conflicts require resolution; never silently override explicit constraints.                                          |
| Optional preferences      | roof style, materials, finish                                         | Non-blocking; use defaults or leave unspecified if not needed for the prototype.                                              |

## 8.1 Per-field state

```
{
  "value": 200,
  "unit": "mm",
  "status": "defaulted",
  "source": "system_default",
  "user_confirmed": false,
  "blocking": false
}
```

Suggested statuses: missing, provided, defaulted, confirmed, conflict. Suggested sources: user_explicit, user_confirmed_default, system_default, inferred. Use "inferred" sparingly and keep it distinct from explicit user input.

## 8.2 Deterministic completeness rules

- Required fields and blocking status are defined in configuration/schema, not improvised by the LLM.
- The checker returns complete, missing_fields, conflicts, and warnings.
- A field is not complete merely because a model guessed a value; it must have a valid value and an allowed provenance/status.
- After defaults are applied, rerun the same validators. Defaults do not bypass validation.
- Do not finalize while blocking conflicts remain unresolved.

# 9. Defaults and Provenance

Maintain a central defaults registry rather than scattering magic numbers across prompts or functions. Defaults are for prototype generation and must not be presented as code-compliant architectural recommendations.

| **Example field** | **Illustrative fallback**                                   | **Required provenance**   |
| ----------------- | ----------------------------------------------------------- | ------------------------- |
| units             | mm                                                          | system_default            |
| floors            | 1 only if single-floor scope is configured                  | system_default            |
| wall_thickness    | configured prototype value, e.g. 200 mm                     | system_default            |
| opening counts    | configured per-room defaults, if supported                  | system_default            |
| room dimensions   | leave unspecified unless a documented layout default exists | system_default or missing |

These are examples only, not finalized architectural standards. Each default should include value, unit, rationale/configuration source, and whether user confirmation is required. Explicit user values always take precedence unless they are invalid or contradictory.

# 10. Gemini API Integration

For this implementation plan, assume Gemini API as the language-model provider. Keep all provider-specific code in llm/gemini_client.py so the extraction agent, schemas, checklist, and tests remain provider-independent. Use the official Google Gen AI Python SDK appropriate to the project environment and pin a tested version in requirements.txt. The exact model identifier and structured-output options should be selected during setup and verified against the chosen model's current API capabilities.

## 10.1 Responsibilities of GeminiClient

- Accept a system instruction, user content, expected schema, and optional minimal context.
- Request structured JSON output where supported.
- Apply timeout and bounded retry behaviour for transient failures.
- Return the raw text/structured response plus usage metadata when available.
- Raise typed exceptions for authentication, rate-limit, timeout, and response-format failures.
- Never store API keys in source code; read GEMINI_API_KEY from an environment variable or secret manager.

## 10.2 When to call Gemini

- Initial extraction: yes.
- Simple follow-up answers such as "2 bathrooms" or "use defaults": usually no; parse deterministically where safe.
- Ambiguous natural-language replies or corrections: yes, send the reply plus only the relevant field definitions/current values.
- Checklist evaluation, default application, validation, round counting, JSON export: no.
- Finalization: no new LLM call is required if the checklist and validation pass.

## 10.3 Prompt design

- State that the model extracts requirements and must not generate CAD code.
- Tell it to distinguish explicitly stated facts from inferred/unspecified values.
- Require null/unspecified for absent values rather than silently inventing dimensions.
- Represent relationships with an allowlisted vocabulary.
- Require a structured output schema and concise ambiguity notes.
- Treat user content as data, not as instructions to change the extraction policy or schema.
- Do not send the full conversation history by default; send the initial prompt once, then targeted follow-up context only when needed.

# 11. Proposed File Structure

```
CAD-generation/
├── app.py
├── schemas.py
├── IR_builder.py
├── agents/
│   ├── __init__.py
│   └── req_ext_agent.py
├── llm/
│   ├── __init__.py
│   ├── gemini_client.py
│   └── prompts.py
├── requirements/
│   ├── __init__.py
│   ├── requirement_state.py
│   ├── completeness_checker.py
│   ├── defaults.py
│   └── dialogue_manager.py
├── validators/
│   ├── __init__.py
│   └── requirements_validator.py
├── tests/
│   ├── test_extraction_schema.py
│   ├── test_completeness.py
│   ├── test_dialogue_round_limit.py
│   ├── test_defaults.py
│   └── test_req_ext_agent.py
├── examples/
│   └── sample_prompts.json
├── outputs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 11.1 Function of each file

| **File**                             | **Responsibility**                                                                                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| app.py                               | Thin CLI or UI entry point. Starts a session, displays questions, accepts replies, and shows the final summary.                                                               |
| schemas.py                           | Pydantic models/enums for extracted requirements, room specifications, relationships, field provenance, AgentTurn, and ExtractionResult. This is the shared contract.         |
| agents/req_ext_agent.py              | Orchestrates initial extraction, checklist update, clarification, validation, defaults, summary, and final handoff. It should not contain provider-specific HTTP/SDK details. |
| llm/gemini_client.py                 | Gemini SDK adapter, structured-output request, timeout/retry, usage metadata, and typed API errors.                                                                           |
| llm/prompts.py                       | Versioned system instructions and prompt builders for initial extraction and targeted interpretation of follow-up answers.                                                    |
| requirements/requirement_state.py    | Stores values, statuses, provenance, pending questions, session ID, and clarification_round.                                                                                  |
| requirements/completeness_checker.py | Deterministic rules for blocking fields, missing fields, conflicts, and readiness to finalize.                                                                                |
| requirements/defaults.py             | Central registry of documented defaults, units, rationale, and confirmation policy.                                                                                           |
| requirements/dialogue_manager.py     | Chooses focused questions, increments rounds, tracks pending questions, and decides whether to clarify or apply defaults.                                                     |
| validators/requirements_validator.py | Checks types, positive counts/dimensions, unit consistency, relationship references, allowed relationship types, and contradiction rules.                                     |
| IR_builder.py                        | Downstream consumer. It should accept only finalized, validated requirements; implementation may follow after this module.                                                    |
| tests/                               | Unit and integration tests for extraction, validation, state retention, round cap, defaults, and Gemini adapter behaviour.                                                    |
| examples/sample_prompts.json         | Small labelled set of input prompts and expected extracted fields for repeatable testing.                                                                                     |
| outputs/                             | Local development outputs such as final requirement JSON; avoid committing personal data or secrets.                                                                          |
| .env.example / .gitignore            | Document required environment variables without real secrets and exclude .env, caches, and sensitive outputs.                                                                 |
| requirements.txt                     | Pinned runtime and test dependencies, including google-genai, pydantic, and pytest.                                                                                           |
| README.md                            | Setup, environment configuration, run instructions, schema contract, default policy, and tests.                                                                               |

# 12. Functions and Tools Needed

| **Function**                               | **Responsibility**                                                                                         |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| extract_initial_requirements(prompt)       | Calls Gemini and parses the initial candidate requirement object.                                          |
| interpret_followup(reply, relevant_fields) | Optional LLM call for ambiguous/free-form answers; returns field updates, not an entire replacement state. |
| parse_and_validate_candidate(data)         | Converts response into Pydantic types and runs schema checks.                                              |
| update_requirement_state(state, updates)   | Merges valid updates while preserving prior values and provenance.                                         |
| check_completeness(state)                  | Returns missing blocking fields, conflicts, warnings, and completion status.                               |
| generate_questions(check_result)           | Builds a compact question set from templates and field metadata.                                           |
| apply_defaults(state, fields)              | Fills eligible missing fields and marks their provenance as system_default.                                |
| build_confirmation_summary(state)          | Creates a readable summary grouped into user-specified values and defaults.                                |
| finalize_result(state)                     | Runs final validation and returns ExtractionResult only if completion policy passes.                       |
| save/load_session(state)                   | Optional first-version persistence to JSON for session continuity.                                         |

These functions are application tools, not necessarily LLM-callable tools. For the first version, use a deterministic Python orchestrator. Tool/function calling can be added later if it solves a concrete need.

# 13. Error Handling and Safety

- Invalid or incomplete model JSON: parse and validate; use a bounded retry or return invalid_output.
- Gemini unavailable or rate-limited: return llm_error with a retryable message; do not fabricate extracted requirements.
- User reply cannot be mapped to a field: ask a narrower question or use the next clarification round.
- Conflicting explicit requirements: mark conflict and request resolution; do not silently overwrite one.
- Unsupported features: state that the current prototype cannot represent them; don't pretend they were accepted.
- Defaults: show every default in the final summary and retain provenance in the JSON.
- Privacy: send only the prompt and relevant context needed for extraction; do not log API keys or unnecessary user data.

# 14. Testing and Acceptance Criteria

| **Test**                              | **Expected result**                                                                                             |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Minimal prompt: "Build a floor plan." | Agent detects missing blocking details and asks focused questions; does not fabricate them as user-provided.    |
| Complete prompt with rooms/counts     | Checklist is populated and no unnecessary clarification round is used.                                          |
| User replies "two bathrooms"          | Bathroom count updates without losing existing room requirements.                                               |
| User says "use defaults"              | Eligible missing fields are defaulted immediately and clearly marked.                                           |
| Three incomplete clarification rounds | Round count never exceeds three; eligible missing fields receive documented defaults after round three.         |
| Contradictory constraints             | Conflict is surfaced; defaults do not silently resolve it.                                                      |
| Invalid count or unit                 | Validator rejects or flags it according to the schema.                                                          |
| Gemini timeout / malformed output     | Typed failure returned; no invalid IR handoff.                                                                  |
| Finalization                          | IR builder receives only a validated, confirmed requirements object.                                            |
| Provenance                            | Every value can be traced to user input, user-confirmed default, system default, or a clearly marked inference. |

Suggested metrics: field-level extraction accuracy, room-count accuracy, relationship accuracy, invalid-schema rate, clarification rounds per session, percentage of defaults used, number of Gemini calls, input/output token usage when available, and end-to-end latency. Evaluate on a small manually labelled prompt set before adding CAD generation.

# 15. Implementation Sequence

22. Define Pydantic schemas and the exact list of required, optional, and defaultable fields.
23. Implement RequirementState and a checklist using hand-written structured examples; test completeness rules without Gemini.
24. Create the defaults registry and provenance tracking.
25. Implement the dialogue manager and enforce the three-round maximum.
26. Implement the Gemini client behind a small provider interface; add environment-variable configuration and timeout handling.
27. Implement initial extraction and targeted follow-up interpretation.
28. Connect the agent orchestrator to the checklist, validators, dialogue manager, and final summary.
29. Build a CLI in app.py and test complete, incomplete, ambiguous, and contradictory prompts.
30. Define and test the final JSON contract expected by IR_builder.py.
31. Measure extraction accuracy, LLM calls, and latency; refine prompts and schemas based on observed failures.

# 16. Key Design Decisions to Freeze Before Coding

- Exact prototype scope: e.g. single-floor residential buildings first, or multiple building types/floors.
- Required versus optional fields and which missing fields are safe to default.
- The definition of a clarification round and the treatment of user confirmation (recommended: initial prompt is not a round; maximum three assistant-question/user-reply cycles).
- Whether an explicit "use defaults" applies to all unspecified fields or only fields named in that response.
- The default values, units, rationales, and whether any require explicit user confirmation.
- Supported relationship vocabulary and conflict policy.
- Final schema contract shared with Building IR.
- Gemini model ID, SDK version, structured-output support, and local environment setup.
- Session persistence strategy: in-memory for a first demo or JSON/SQLite if sessions must survive restarts.

# 17. Summary

The Requirements Extraction Agent should be implemented as a stateful, checklist-driven workflow rather than an unrestricted autonomous agent. Gemini extracts the initial meaning and interprets genuinely ambiguous replies; Python maintains the checklist, asks targeted questions, enforces a maximum of three clarification rounds, applies documented defaults, validates the result, and packages a final requirements object. This architecture limits unnecessary LLM calls, makes the interaction auditable and testable, preserves user intent and default provenance, and establishes a stable interface for the Building IR module.