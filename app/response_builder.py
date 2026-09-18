"""Build and replay-check the actual serialized schedule before returning it.

Keep eight decimal places for energy instead of rounding every action to cents.
Only the billed total is rounded to two decimals; all totals come from the
returned rows. The organiser permits numeric tolerance, not invented energy.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from math import fsum, isfinite
from typing import List

from .optimizer import OptimizerResult, _apply_directives
from .schemas import Battery, DirectiveInterpretation, HourEntry, HourlyPlanRow, OptimizeResponse


class InvalidSchedule(RuntimeError):
    """A solved schedule failed the independent output replay."""


def _r2(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def _maybe_int(value: float) -> float:
    return float(round(value)) if abs(value - round(value)) < 1e-9 else value


def _energy(value: float) -> float:
    if not isfinite(value) or value < -1e-7:
        raise InvalidSchedule("Solver produced an invalid energy value.")
    return round(max(0.0, value), 8)


def _validate_plan(rows, hours, battery, directives) -> None:
    """Replay the public rows against demand, physical limits and directives."""
    tol = 0.001  # Stricter than the organiser's 0.01 kWh tolerance.
    caps = _apply_directives(hours, directives)
    energy = battery.initial_energy_kwh
    for row, entry in zip(rows, hours):
        h = row.hour
        charge = row.battery_kwh if row.battery_action == "charge" else 0.0
        discharge = row.battery_kwh if row.battery_action == "discharge" else 0.0
        energy += charge - discharge
        checks = [
            abs(row.grid_kwh + row.solar_used_kwh + discharge - entry.demand_kwh - charge) <= tol,
            row.solar_used_kwh <= caps["effective_solar"][h] + tol,
            charge <= battery.max_charge_kwh_per_hour + tol,
            discharge <= battery.max_discharge_kwh_per_hour + tol,
            abs(energy - row.battery_energy_after_kwh) <= tol,
            row.battery_energy_after_kwh >= max(battery.minimum_energy_kwh, caps["min_reserve"].get(h, 0)) - tol,
            row.battery_energy_after_kwh <= battery.capacity_kwh + tol,
            h not in caps["no_charge"] or charge <= tol,
            h not in caps["no_discharge"] or discharge <= tol,
            row.grid_kwh <= caps["max_grid"].get(h, float("inf")) + tol,
        ]
        if not all(checks):
            raise InvalidSchedule(f"Schedule failed energy or directive validation at hour {h}.")
    if abs(energy - battery.initial_energy_kwh) > tol:
        raise InvalidSchedule("Schedule did not restore the initial battery energy.")


def build_response(
    scenario_id: str,
    battery: Battery,
    hours: List[HourEntry],
    directives: List[DirectiveInterpretation],
    result: OptimizerResult,
) -> OptimizeResponse:
    hours = sorted(hours, key=lambda entry: entry.hour)
    raw_rows = sorted(result.hourly, key=lambda row: row["hour"])
    if [row["hour"] for row in raw_rows] != list(range(24)):
        raise InvalidSchedule("Solver did not return all 24 hours.")
    rows = []
    for raw in raw_rows:
        charge = _energy(raw["charge_kwh"])
        discharge = _energy(raw["disch_kwh"])
        if charge > 0 and discharge > 0:
            raise InvalidSchedule("Solver charged and discharged in the same hour.")
        action = "charge" if charge > 0 else "discharge" if discharge > 0 else "idle"
        rows.append(HourlyPlanRow(
            hour=int(raw["hour"]),
            grid_kwh=_energy(raw["grid_kwh"]),
            solar_used_kwh=_energy(raw["solar_used_kwh"]),
            battery_action=action,
            battery_kwh=charge or discharge,
            battery_energy_after_kwh=_energy(raw["battery_energy_after_kwh"]),
        ))
    _validate_plan(rows, hours, battery, directives)
    total_grid = round(fsum(row.grid_kwh for row in rows), 8)
    cost = sum((Decimal(str(h.tariff_bdt_per_kwh)) * Decimal(str(row.grid_kwh))
                for h, row in zip(hours, rows)), Decimal(0))
    total_cost = float(cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
    return OptimizeResponse(
        scenario_id=scenario_id,
        directive_interpretation=directives,  # Preserve exact factors and limits.
        hourly_plan=rows,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=max(row.grid_kwh for row in rows),
        plan_summary=_build_plan_summary(directives, total_cost),
    )


def _build_plan_summary(
    directives: List[DirectiveInterpretation],
    total_cost: float,
) -> str:
    """One-line summary of what the optimizer did."""
    active = [d for d in directives if d.applies and d.directive_type != "no_op"]
    bits: List[str] = []
    for d in active:
        adj = d.structured_adjustment
        if adj is None:
            continue
        from .schemas import (
            MaxGridWindow,
            MinimumBatteryReserve,
            NoChargeWindow,
            NoDischargeWindow,
            SolarReduction,
        )
        if isinstance(adj, SolarReduction):
            bits.append(f"solar reduced to {adj.factor*100:.0f}% in hours {adj.hours}")
        elif isinstance(adj, MinimumBatteryReserve):
            mn = _maybe_int(adj.minimum_energy_kwh)
            bits.append(f"battery reserve ≥ {mn:g} kWh in hours {adj.hours}")
        elif isinstance(adj, NoChargeWindow):
            bits.append(f"charging disabled in hours {adj.hours}")
        elif isinstance(adj, NoDischargeWindow):
            bits.append(f"discharging disabled in hours {adj.hours}")
        elif isinstance(adj, MaxGridWindow):
            cap = _maybe_int(adj.max_grid_kwh)
            bits.append(f"grid import capped at {cap:g} kWh/h in hours {adj.hours}")

    if bits:
        prefix = "Honors " + "; ".join(bits) + "."
    else:
        prefix = "No operator directives applied."
    return (
        f"{prefix} Shifts battery energy to higher-tariff hours, restores the "
        f"initial battery level, and finishes at total cost {total_cost:g} BDT."
    )
