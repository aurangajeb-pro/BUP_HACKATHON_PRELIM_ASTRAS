"""
F5 — rounding and consistency tests.

Reproduces the two issues called out in the compliance review:
  A. Fractional inputs (0.014 kWh demand) — totals computed from rounded plan
     must equal reported totals within 0.005.
  B. Battery-state drift — replayed E[h] must equal reported E[h] for every hour.

These tests are LLM-independent (no operator notes) — they exercise the
optimizer + response_builder directly. Pass with Ollama online or offline.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _client():
    from app.main import app
    return TestClient(app)


def _scenario(hours, battery, notes=None):
    return {
        "scenario_id": "ROUNDING",
        # Pydantic schema requires at least 1 operator note; the LLM will
        # resolve a generic note as no_op, so this doesn't affect the schedule.
        "operator_notes": notes if notes else ["No additional operator notes today."],
        "hours": hours,
        "battery": battery,
    }


# ---- Reproduction A: fractional demand, no notes ----

def test_fractional_inputs_match_totals(no_directives):
    """Each hour: demand 0.014 kWh, no solar, tariff 10 BDT/kWh.
    Reported total_cost_bdt must equal sum(tariff * plan.grid_kwh)."""
    hours = [
        {"hour": h, "demand_kwh": 0.014, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 10.0}
        for h in range(24)
    ]
    battery = {
        "capacity_kwh": 1.0,
        "initial_energy_kwh": 0.5,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 0.5,
        "max_discharge_kwh_per_hour": 0.5,
    }
    with _client() as c:
        r = c.post("/optimize-energy", json=_scenario(hours, battery))
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["total_cost_bdt"] == pytest.approx(3.36, abs=0.005)
    assert body["total_grid_kwh"] == pytest.approx(0.336, abs=1e-6)

    # Recompute totals from the returned plan.
    plan = body["hourly_plan"]
    recomputed_grid = sum(row["grid_kwh"] for row in plan)
    tariff_by_hour = {h["hour"]: h["tariff_bdt_per_kwh"] for h in hours}
    recomputed_cost = sum(tariff_by_hour[row["hour"]] * row["grid_kwh"] for row in plan)

    assert abs(body["total_grid_kwh"] - recomputed_grid) <= 0.005, (
        f"total_grid {body['total_grid_kwh']} != sum(plan.grid) {recomputed_grid}"
    )
    assert abs(body["total_cost_bdt"] - recomputed_cost) <= 0.005, (
        f"total_cost {body['total_cost_bdt']} != sum(tariff*plan.grid) {recomputed_cost}"
    )


# ---- Reproduction B: battery-state drift ----

def test_battery_drift_zero(no_directives):
    """Hourly charge/discharge capped at 1.014 kWh. Replay from rounded E[0]
    and rounded actions; replayed E[h] must equal reported E[h] for all h."""
    hours = [
        # Alternate cheap-tariff (charge) and expensive-tariff (discharge) hours
        # so the optimizer exercises both directions.
        {"hour": h, "demand_kwh": 5.0, "solar_kwh": 0.0,
         "tariff_bdt_per_kwh": 5.0 if h < 12 else 50.0}
        for h in range(24)
    ]
    battery = {
        "capacity_kwh": 100.0,
        "initial_energy_kwh": 50.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 1.014,
        "max_discharge_kwh_per_hour": 1.014,
    }
    with _client() as c:
        r = c.post("/optimize-energy", json=_scenario(hours, battery))
    assert r.status_code == 200, r.text
    body = r.json()
    plan = body["hourly_plan"]

    # Replay from initial_energy_kwh using rounded charge/discharge values.
    E_replay = battery["initial_energy_kwh"]
    for row in plan:
        action = row["battery_action"]
        bat_kwh = row["battery_kwh"]
        if action == "charge":
            E_replay += bat_kwh
        elif action == "discharge":
            E_replay -= bat_kwh
        # idle: E unchanged
        # Compare to reported E with small tolerance for floating-point noise.
        assert abs(E_replay - row["battery_energy_after_kwh"]) <= 0.005, (
            f"h{row['hour']}: replayed E={E_replay} != reported {row['battery_energy_after_kwh']}"
        )


# ---- Sanity: total_cost equals sum(tariff * grid) for every public case ----

def test_totals_consistent_for_simple_no_directive_scenario(no_directives):
    """With no operator notes, the reported totals must equal sums over the plan."""
    hours = [
        {"hour": h, "demand_kwh": 50.0 + (h % 5) * 10, "solar_kwh": 0.0,
         "tariff_bdt_per_kwh": 10.0 + (h % 3) * 2}
        for h in range(24)
    ]
    battery = {
        "capacity_kwh": 100.0,
        "initial_energy_kwh": 50.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 20.0,
        "max_discharge_kwh_per_hour": 20.0,
    }
    with _client() as c:
        r = c.post("/optimize-energy", json=_scenario(hours, battery))
    assert r.status_code == 200, r.text
    body = r.json()
    plan = body["hourly_plan"]
    tariff_by_hour = {h["hour"]: h["tariff_bdt_per_kwh"] for h in hours}

    s_grid = sum(row["grid_kwh"] for row in plan)
    s_cost = sum(tariff_by_hour[row["hour"]] * row["grid_kwh"] for row in plan)
    mx_grid = max(row["grid_kwh"] for row in plan)

    assert abs(body["total_grid_kwh"] - s_grid) <= 0.005
    assert abs(body["total_cost_bdt"] - s_cost) <= 0.005
    assert abs(body["peak_grid_kwh"] - mx_grid) <= 0.005
