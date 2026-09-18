"""
Tests for 11 paraphrase and distractor examples.

Each test constructs a minimal but realistic scenario containing the target
note, POSTs to /optimize-energy, and asserts that the LLM interpreter
returned the expected directive_type and structured_adjustment.

Run with --run-llm-tests. Unavailable models fail the explicitly requested live gate.
"""
from __future__ import annotations

import pytest


def _scenario_for(note: str) -> dict:
    """Build a minimal 24-hour scenario for one operator note.

    Battery is generous so the optimizer always finds a feasible schedule
    regardless of what directive the LLM extracts.
    """
    return {
        "scenario_id": "COUNTEREX",
        "operator_notes": [note],
        "hours": [
            {"hour": h, "demand_kwh": 100.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 10.0}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 200.0,
            "initial_energy_kwh": 100.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 50.0,
            "max_discharge_kwh_per_hour": 50.0,
        },
    }


# Each entry: (note, expected_type, expected_hours_or_None, expected_factor_or_None,
#              expected_min_kwh_or_None, expected_max_grid_or_None)
COUNTEREXAMPLES = [
    (
        "Keep at least 120 kWh in reserve from 6 PM until 9 PM.",
        "minimum_battery_reserve", [18, 19, 20], None, 120.0, None,
    ),
    (
        "PV production will drop to about 20% between 13:00 and 15:00.",
        "solar_reduction", [13, 14], 0.2, None, None,
    ),
    (
        "Panel washing from one until three will leave roughly one-fifth of normal solar output.",
        "solar_reduction", [13, 14], 0.2, None, None,
    ),
    (
        "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.",
        "solar_reduction", [13, 14], 0.2, None, None,
    ),
    (
        "Solar output will be reduced by 80% from 1 PM to 3 PM.",
        "solar_reduction", [13, 14], 0.2, None, None,
    ),
    (
        "Disable battery discharge from 6 PM to 8 PM.",
        "no_discharge_window", [18, 19], None, None, None,
    ),
    (
        "Hallway cleaning is scheduled from 1 PM to 3 PM.",
        "no_op", None, None, None, None,
    ),
    (
        "The sports office says do not charge the battery from 2 PM to 4 PM.",
        "no_charge_window", [14, 15], None, None, None,
    ),
    (
        "The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance.",
        "no_charge_window", [2, 3, 4], None, None, None,
    ),
    (
        "Cloud cover during panel inspection will leave about half of the forecast solar output from 10 AM until noon.",
        "solar_reduction", [10, 11], 0.5, None, None,
    ),
    (
        "The sports office moved next month's registration deadline.",
        "no_op", None, None, None, None,
    ),
]


@pytest.mark.live_llm
@pytest.mark.parametrize(
    "note,expected_type,expected_hours,expected_factor,expected_min_kwh,expected_max_grid",
    COUNTEREXAMPLES,
    ids=[f"note_{i+1}" for i in range(len(COUNTEREXAMPLES))],
)
def test_counterexample(
    client, ollama_available,
    note, expected_type, expected_hours, expected_factor, expected_min_kwh, expected_max_grid,
):

    payload = _scenario_for(note)
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    body = resp.json()
    directives = body["directive_interpretation"]
    assert len(directives) == 1, f"expected 1 directive, got {len(directives)}"
    d = directives[0]

    assert d["directive_type"] == expected_type, (
        f"note: {note!r}\n  expected type: {expected_type}\n  got: {d['directive_type']}"
    )

    adj = d["structured_adjustment"]
    if expected_type == "no_op":
        assert adj is None, f"no_op should have null structured_adjustment, got {adj}"
    else:
        assert adj is not None, f"{expected_type} must have structured_adjustment"
        if expected_hours is not None:
            assert adj.get("hours") == expected_hours, (
                f"hours mismatch: expected {expected_hours}, got {adj.get('hours')}"
            )
        if expected_factor is not None:
            assert adj.get("factor") is not None
            assert abs(adj["factor"] - expected_factor) < 1e-3, (
                f"factor: expected {expected_factor}, got {adj['factor']}"
            )
        if expected_min_kwh is not None:
            assert adj.get("minimum_energy_kwh") is not None
            assert abs(adj["minimum_energy_kwh"] - expected_min_kwh) < 1e-3, (
                f"min_kwh: expected {expected_min_kwh}, got {adj['minimum_energy_kwh']}"
            )
        if expected_max_grid is not None:
            assert adj.get("max_grid_kwh") is not None
            assert abs(adj["max_grid_kwh"] - expected_max_grid) < 1e-3, (
                f"max_grid: expected {expected_max_grid}, got {adj['max_grid_kwh']}"
            )
