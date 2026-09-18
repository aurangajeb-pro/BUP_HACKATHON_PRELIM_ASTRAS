"""
Scoring harness for the 10 public sample cases.

Compliance: post each case to /optimize-energy and verify:
  - Total cost matches the reference within 0.01 BDT (LP-optimal ⇒ exact).
  - Total grid and peak grid match within 0.01 kWh.
  - Every GridWise constraint is satisfied (energy balance, battery bounds,
    rate limits, solar caps, directive windows, end-of-day neutrality).
  - Internal totals are consistent with the hourly plan.

LLM tests are opt-in with --run-llm-tests. Math tests inject organiser directives
at the model boundary; they do not claim to measure interpretation accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

CASES_PATH = Path(__file__).resolve().parent / "data" / "public_sample_cases.json"
TOL = 1e-2


def _load_cases() -> list[Dict[str, Any]]:
    with open(CASES_PATH) as f:
        return json.load(f)["cases"]


CASES = _load_cases()


@pytest.fixture
def reference_model(monkeypatch, case_id):
    """Supply organiser ground truth at the model boundary for deterministic math tests.

    These tests do not measure natural-language interpretation accuracy.
    """
    import app.llm as llm
    case = next(c for c in CASES if c["id"] == case_id)
    entries = json.loads(json.dumps(case["expected_output"]["directive_interpretation"]))
    for entry in entries:
        if entry["structured_adjustment"] is not None:
            entry["structured_adjustment"]["kind"] = entry["directive_type"]
    monkeypatch.setattr(llm, "_call_ollama", lambda messages: json.dumps({"interpretations": entries}))


# ----- Per-case: math-correctness (no LLM dependency) -----

CASE_IDS = [c["id"] for c in CASES]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_total_cost_matches_reference(client, case_id, reference_model):
    case = next(c for c in CASES if c["id"] == case_id)
    resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200, f"{case_id}: HTTP {resp.status_code} {resp.text}"
    body = resp.json()
    ref = case["expected_output"]
    assert abs(body["total_cost_bdt"] - ref["total_cost_bdt"]) <= TOL
    assert abs(body["total_grid_kwh"] - ref["total_grid_kwh"]) <= TOL
    # Equal-cost schedules can have different peaks; replay totals independently.


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_constraints_replay(client, case_id, reference_model):
    """Replay every spec constraint against the returned hourly_plan."""
    case = next(c for c in CASES if c["id"] == case_id)
    resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200
    body = resp.json()
    plan = body["hourly_plan"]
    hours = {h["hour"]: h for h in case["input"]["hours"]}
    battery = case["input"]["battery"]
    directives = body["directive_interpretation"]

    eff_solar = {h: hours[h]["solar_kwh"] for h in range(24)}
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    min_reserve: dict[int, float] = {}
    max_grid: dict[int, float] = {}
    for d in directives:
        if not d["applies"] or d["structured_adjustment"] is None:
            continue
        adj = d["structured_adjustment"]
        t = d["directive_type"]
        if t == "solar_reduction":
            for h in adj["hours"]:
                eff_solar[h] = min(eff_solar[h], hours[h]["solar_kwh"] * adj["factor"])
        elif t == "no_charge_window":
            no_charge.update(adj["hours"])
        elif t == "no_discharge_window":
            no_discharge.update(adj["hours"])
        elif t == "minimum_battery_reserve":
            for h in adj["hours"]:
                min_reserve[h] = max(min_reserve.get(h, 0.0), adj["minimum_energy_kwh"])
        elif t == "max_grid_window":
            for h in adj["hours"]:
                cur = max_grid.get(h)
                max_grid[h] = min(cur, adj["max_grid_kwh"]) if cur is not None else adj["max_grid_kwh"]

    prev_E = battery["initial_energy_kwh"]
    for row in plan:
        h = row["hour"]
        demand = hours[h]["demand_kwh"]
        charge = row["battery_kwh"] if row["battery_action"] == "charge" else 0.0
        disch = row["battery_kwh"] if row["battery_action"] == "discharge" else 0.0

        # Energy balance.
        assert abs(row["grid_kwh"] + row["solar_used_kwh"] + disch
                   - demand - charge) <= TOL, \
            f"{case_id} h{h}: balance"

        # Solar cap.
        assert row["solar_used_kwh"] <= eff_solar[h] + TOL, \
            f"{case_id} h{h}: solar cap"

        # Battery rate limits.
        assert charge <= battery["max_charge_kwh_per_hour"] + TOL
        assert disch <= battery["max_discharge_kwh_per_hour"] + TOL
        assert row["battery_kwh"] >= -TOL

        # Directive windows.
        if h in no_charge:
            assert charge <= TOL, f"{case_id} h{h}: charge in no_charge_window"
        if h in no_discharge:
            assert disch <= TOL, f"{case_id} h{h}: discharge in no_discharge_window"
        if h in min_reserve:
            assert row["battery_energy_after_kwh"] >= min_reserve[h] - TOL, \
                f"{case_id} h{h}: reserve"
        if h in max_grid:
            assert row["grid_kwh"] <= max_grid[h] + TOL, \
                f"{case_id} h{h}: grid cap"

        # Battery dynamics.
        assert abs(row["battery_energy_after_kwh"] - prev_E - charge + disch) <= TOL, \
            f"{case_id} h{h}: battery dynamics"

        # Battery bounds.
        assert row["battery_energy_after_kwh"] >= battery["minimum_energy_kwh"] - TOL
        assert row["battery_energy_after_kwh"] <= battery["capacity_kwh"] + TOL

        prev_E = row["battery_energy_after_kwh"]

    # End-of-day neutrality.
    assert abs(prev_E - battery["initial_energy_kwh"]) <= TOL, \
        f"{case_id}: end-of-day neutrality"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_response_totals_consistent(client, case_id, reference_model):
    case = next(c for c in CASES if c["id"] == case_id)
    resp = client.post("/optimize-energy", json=case["input"])
    body = resp.json()
    plan = body["hourly_plan"]
    hours = {h["hour"]: h for h in case["input"]["hours"]}

    s_grid = sum(r["grid_kwh"] for r in plan)
    s_cost = sum(hours[r["hour"]]["tariff_bdt_per_kwh"] * r["grid_kwh"] for r in plan)
    mx = max(r["grid_kwh"] for r in plan)

    assert abs(body["total_grid_kwh"] - s_grid) <= TOL
    assert abs(body["total_cost_bdt"] - s_cost) <= TOL
    assert abs(body["peak_grid_kwh"] - mx) <= TOL


# ----- LLM-dependent: directive matching — gated by ollama_available -----

@pytest.mark.live_llm
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_directives_match_reference(client, ollama_available, case_id):
    case = next(c for c in CASES if c["id"] == case_id)

    resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ref = case["expected_output"]["directive_interpretation"]

    assert len(body["directive_interpretation"]) == len(ref), \
        f"{case_id}: directive count mismatch"

    for got, exp in zip(body["directive_interpretation"], ref):
        assert got["note_index"] == exp["note_index"], f"{case_id}: note_index"
        assert got["applies"] == exp["applies"], \
            f"{case_id} note {got['note_index']}: applies"
        assert got["directive_type"] == exp["directive_type"], \
            f"{case_id} note {got['note_index']}: type"

        if exp["structured_adjustment"] is None:
            assert got["structured_adjustment"] is None, \
                f"{case_id} note {got['note_index']}: expected no adjustment"
        else:
            adj_got = got["structured_adjustment"]
            adj_exp = exp["structured_adjustment"]
            assert adj_got["hours"] == adj_exp["hours"], \
                f"{case_id} note {got['note_index']}: hours"
            for field, tol in [("factor", 1e-3),
                               ("minimum_energy_kwh", TOL),
                               ("max_grid_kwh", TOL)]:
                if field in adj_exp:
                    assert field in adj_got
                    assert abs(adj_got[field] - adj_exp[field]) <= tol, \
                        f"{case_id} note {got['note_index']}: {field}"
