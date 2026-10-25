"""
IR_builder.py
=============
Placeholder for the Building Intermediate Representation (Building IR) builder.

This module is OUT OF SCOPE for the Requirements Extraction Agent (Module 1).
It is defined here only to:
  1. Document the interface that the agent's ``ExtractionResult`` must satisfy.
  2. Provide a stub that downstream code can import without errors.

The full implementation follows in a later project milestone.

Interface contract
------------------
The IR builder accepts only a fully validated ``ExtractionResult`` produced by
``agents.req_ext_agent.finalize_requirements()``.  It must never accept a
partially-filled state object or raw LLM output.

Expected future signature::

    def build_ir(result: ExtractionResult) -> BuildingIR:
        ...

Where ``BuildingIR`` will be a structured representation of rooms, walls,
openings, and spatial relationships ready for the CAD compiler.
"""

from __future__ import annotations

from schemas import ExtractionResult


class BuildingIRNotImplementedError(NotImplementedError):
    """Raised when the IR builder is called before it is implemented."""


def build_ir(result: ExtractionResult) -> None:
    """
    Stub for the future Building IR builder.

    Parameters
    ----------
    result : ExtractionResult
        The validated, finalized requirements from the extraction agent.

    Raises
    ------
    BuildingIRNotImplementedError
        Always — implementation is out of scope for Module 1.
    """
    if not isinstance(result, ExtractionResult):
        raise TypeError(
            "build_ir requires an ExtractionResult instance. "
            f"Got {type(result).__name__}."
        )
    if not result.confirmed_by_user:
        raise ValueError(
            "ExtractionResult must be confirmed by the user before "
            "being passed to the IR builder."
        )
    raise BuildingIRNotImplementedError(
        "The Building IR builder is not yet implemented. "
        "It will be the next module in the pipeline."
    )
