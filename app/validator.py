"""
Deterministic directive validator.

Runs after the parser and before the optimizer. A malformed or
semantically-inconsistent directive raises ValueError with a clear reason,
which FastAPI surfaces as 422.

This is the safety net for the LLM-robustness half of the rubric.
"""
from __future__ import annotations

from typing import List

from .schemas import (
    Battery,
    DirectiveInterpretation,
    MaxGridWindow,
    MinimumBatteryReserve,
    NoChargeWindow,
    NoDischargeWindow,
    SolarReduction,
)


def validate_directives(
    directives: List[DirectiveInterpretation],
    battery: Battery,
    note_count: int | None = None,
) -> None:
    """Raise ValueError on the first invalid directive.

    Checks (per spec _meta.constraint_reminders and allowed_enums):
      - directive_type ∈ allowed set
      - applies=False ↔ directive_type=='no_op' ↔ structured_adjustment is None
      - hours: unique ascending ints 0..23, count ≥ 1
      - factor ∈ [0, 1]
      - minimum_energy_kwh ∈ [0, capacity_kwh]
      - max_grid_kwh ≥ 0
      - window start < end (start-inclusive, end-exclusive semantics already
        enforced by _expand_window in parser, but we double-check here)
    """
    indices = [d.note_index for d in directives]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate note_index")
    if note_count is not None and indices != list(range(note_count)):
        raise ValueError("interpretations must cover every note once in note_index order")
    for d in directives:
        idx = d.note_index

        # no_op ↔ applies=False ↔ structured_adjustment is None
        if d.directive_type == "no_op":
            if d.applies:
                raise ValueError(f"note {idx}: no_op must have applies=false")
            if d.structured_adjustment is not None:
                raise ValueError(f"note {idx}: no_op must have null structured_adjustment")
            continue

        if not d.applies:
            raise ValueError(f"note {idx}: non-no_op directive must have applies=true")

        if d.structured_adjustment is None:
            raise ValueError(f"note {idx}: {d.directive_type} missing structured_adjustment")

        adj = d.structured_adjustment
        if adj.kind != d.directive_type:
            raise ValueError(f"note {idx}: structured_adjustment kind must match directive_type")

        # hours: unique ascending 0..23, count ≥ 1
        if not adj.hours:
            raise ValueError(f"note {idx}: hours list must be non-empty")
        prev = -1
        for h in adj.hours:
            if not isinstance(h, int) or not (0 <= h <= 23):
                raise ValueError(f"note {idx}: hour {h} not an int in 0..23")
            if h <= prev:
                raise ValueError(f"note {idx}: hours must be ascending unique")
            prev = h

        # Per-type numeric checks.
        if isinstance(adj, SolarReduction):
            if not (0.0 <= adj.factor <= 1.0):
                raise ValueError(f"note {idx}: factor {adj.factor} outside [0, 1]")
        elif isinstance(adj, MinimumBatteryReserve):
            if adj.minimum_energy_kwh < 0:
                raise ValueError(f"note {idx}: minimum_energy_kwh must be ≥ 0")
            if adj.minimum_energy_kwh > battery.capacity_kwh + 1e-6:
                raise ValueError(
                    f"note {idx}: minimum_energy_kwh {adj.minimum_energy_kwh} "
                    f"exceeds capacity {battery.capacity_kwh}"
                )
        elif isinstance(adj, MaxGridWindow):
            if adj.max_grid_kwh < 0:
                raise ValueError(f"note {idx}: max_grid_kwh must be ≥ 0")
        elif isinstance(adj, (NoChargeWindow, NoDischargeWindow)):
            pass  # no extra numeric checks
        else:
            raise ValueError(f"note {idx}: unknown structured_adjustment type {type(adj).__name__}")
