"""
Solver glue — orchestrates LLM interpret → validate → optimize → build response.

This module is the single entry point for `app.main`. Keeping it isolated
makes it easy to test and to swap the optimizer or LLM backend without
touching the HTTP layer.
"""
from __future__ import annotations

import logging

from .llm import interpret_notes
from .optimizer import InfeasibleSchedule, optimize
from .response_builder import build_response
from .schemas import (
    DirectiveInterpretation,
    OptimizeRequest,
    OptimizeResponse,
)
from .validator import validate_directives

log = logging.getLogger("gridwise.solver")


def solve(request: OptimizeRequest) -> OptimizeResponse:
    """End-to-end: LLM-interpret all notes, validate, optimize, build response."""
    # 1. LLM interpretation (sole interpreter — no heuristic fallback).
    directives: list[DirectiveInterpretation] = interpret_notes(
        request.operator_notes, request.battery
    )

    # 2. Schema-level validation.
    validate_directives(directives, request.battery, len(request.operator_notes))

    # 3. Optimize.
    result = optimize(request.hours, request.battery, directives)
    if result.status != "optimal":
        raise InfeasibleSchedule(f"Optimizer status: {result.status}")

    # 4. Assemble the public response (centralized rounding + invariants).
    return build_response(
        scenario_id=request.scenario_id,
        battery=request.battery,
        hours=request.hours,
        directives=directives,
        result=result,
    )
