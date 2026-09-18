"""
GridWise optimizer — mixed-integer linear programming via PuLP/CBC.

Minimizes total grid cost over 24 hours subject to:
  - Energy balance per hour
  - Battery bounds (min ≤ E ≤ capacity) and end-of-day neutrality
  - Per-hour charge/discharge rate limits
  - Effective solar caps (after solar_reduction)
  - Directive constraints: no_charge_window, no_discharge_window,
    minimum_battery_reserve, max_grid_window

Returns the solved hourly schedule as a list of dicts plus aggregate metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pulp

from .schemas import (
    Battery,
    DirectiveInterpretation,
    HourEntry,
    MaxGridWindow,
    MinimumBatteryReserve,
    NoChargeWindow,
    NoDischargeWindow,
    SolarReduction,
)


@dataclass
class OptimizerResult:
    """Raw (unrounded) solver output.

    `hourly` is a list of 24 dicts, each with full-precision floats:
        {hour, grid_kwh, solar_used_kwh, charge_kwh, disch_kwh, battery_energy_after_kwh}
    `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` are also full-precision.
    The response builder is responsible for rounding and for asserting the
    invariants that keep the rounded plan consistent with these raw vectors.
    """
    hourly: List[Dict[str, float]]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    status: str                       # 'optimal' | 'infeasible' | 'unbounded' | 'error'


class InfeasibleSchedule(Exception):
    """Raised when the directive set is incompatible with the scenario."""


class SolverFailure(Exception):
    """The solver could not prove an optimum or could not execute."""


def _apply_directives(
    hours: List[HourEntry],
    directives: List[DirectiveInterpretation],
) -> Dict[str, object]:
    """Pre-compute per-hour caps from the parsed directives."""
    effective_solar = [h.solar_kwh for h in hours]
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    min_reserve: Dict[int, float] = {}
    max_grid: Dict[int, float] = {}

    for d in directives:
        if not d.applies or d.structured_adjustment is None:
            continue
        adj = d.structured_adjustment
        if isinstance(adj, SolarReduction):
            for h in adj.hours:
                if 0 <= h < 24:
                    effective_solar[h] = min(effective_solar[h], hours[h].solar_kwh * adj.factor)
        elif isinstance(adj, NoChargeWindow):
            no_charge.update(adj.hours)
        elif isinstance(adj, NoDischargeWindow):
            no_discharge.update(adj.hours)
        elif isinstance(adj, MinimumBatteryReserve):
            for h in adj.hours:
                if 0 <= h < 24:
                    min_reserve[h] = max(min_reserve.get(h, 0.0), adj.minimum_energy_kwh)
        elif isinstance(adj, MaxGridWindow):
            for h in adj.hours:
                if 0 <= h < 24:
                    # If multiple max_grid windows overlap, take the strictest (min).
                    cur = max_grid.get(h)
                    max_grid[h] = min(cur, adj.max_grid_kwh) if cur is not None else adj.max_grid_kwh

    return {
        "effective_solar": effective_solar,
        "no_charge": no_charge,
        "no_discharge": no_discharge,
        "min_reserve": min_reserve,
        "max_grid": max_grid,
    }


def optimize(
    hours: List[HourEntry],
    battery: Battery,
    directives: List[DirectiveInterpretation],
    time_limit_seconds: int = 5,
) -> OptimizerResult:
    """Solve the 24-hour schedule as a linear program."""
    hours = sorted(hours, key=lambda entry: entry.hour)
    if [entry.hour for entry in hours] != list(range(24)):
        raise ValueError("hours must contain exactly one entry for each hour 0..23")

    ctx = _apply_directives(hours, directives)
    eff_solar: List[float] = ctx["effective_solar"]  # type: ignore[assignment]
    no_charge: set = ctx["no_charge"]  # type: ignore[assignment]
    no_discharge: set = ctx["no_discharge"]  # type: ignore[assignment]
    min_reserve: Dict[int, float] = ctx["min_reserve"]  # type: ignore[assignment]
    max_grid: Dict[int, float] = ctx["max_grid"]  # type: ignore[assignment]

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)

    # Decision variables (per hour)
    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(24)]
    solar = [pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=eff_solar[h]) for h in range(24)]
    charge = [pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=battery.max_charge_kwh_per_hour)
              for h in range(24)]
    discharge = [pulp.LpVariable(f"disch_{h}", lowBound=0, upBound=battery.max_discharge_kwh_per_hour)
                 for h in range(24)]
    E = [pulp.LpVariable(f"E_{h}", lowBound=battery.minimum_energy_kwh,
                         upBound=battery.capacity_kwh) for h in range(24)]

    # Indicator binaries so charge and discharge cannot both be nonzero in the same hour.
    y_ch = [pulp.LpVariable(f"y_ch_{h}", cat="Binary") for h in range(24)]
    y_dis = [pulp.LpVariable(f"y_dis_{h}", cat="Binary") for h in range(24)]

    M_CHARGE = battery.max_charge_kwh_per_hour
    M_DISCH = battery.max_discharge_kwh_per_hour

    # Objective: minimize total cost (grid is the only billed source).
    prob += pulp.lpSum(hours[h].tariff_bdt_per_kwh * grid[h] for h in range(24))

    # Energy balance: grid + solar + discharge = demand + charge.
    for h in range(24):
        prob += (
            grid[h] + solar[h] + discharge[h]
            == hours[h].demand_kwh + charge[h]
        ), f"balance_{h}"

    # Battery dynamics.
    prob += E[0] == battery.initial_energy_kwh + charge[0] - discharge[0], "E0"
    for h in range(1, 24):
        prob += E[h] == E[h - 1] + charge[h] - discharge[h], f"E_{h}"

    # End-of-day neutrality.
    prob += E[23] == battery.initial_energy_kwh, "E_end_neutrality"

    # Minimum reserve overrides where applicable.
    for h, mn in min_reserve.items():
        prob += E[h] >= mn, f"E_min_{h}"

    # max_grid caps.
    for h, cap in max_grid.items():
        prob += grid[h] <= cap, f"grid_max_{h}"

    # no_charge_window.
    for h in no_charge:
        prob += charge[h] == 0, f"no_charge_{h}"

    # no_discharge_window.
    for h in no_discharge:
        prob += discharge[h] == 0, f"no_discharge_{h}"

    # Mutual exclusion of charge/discharge via binaries.
    for h in range(24):
        prob += charge[h] <= M_CHARGE * y_ch[h], f"bigM_ch_{h}"
        prob += discharge[h] <= M_DISCH * y_dis[h], f"bigM_dis_{h}"
        prob += y_ch[h] + y_dis[h] <= 1, f"excl_{h}"

    # Solve.
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    try:
        prob.solve(solver)
    except pulp.PulpSolverError as exc:
        raise SolverFailure("Optimization solver could not execute.") from exc

    status = pulp.LpStatus[prob.status].lower()
    if status == "infeasible":
        raise InfeasibleSchedule(
            f"Solver returned status '{status}' — directives may be infeasible "
            f"(e.g. min reserve too high, conflicting no_charge/no_discharge windows)."
        )
    if status != "optimal" or prob.sol_status != pulp.LpSolutionOptimal:
        raise SolverFailure("Optimization did not finish with a proven optimal schedule.")

    # Extract raw solution (full precision — rounding happens in response_builder).
    grid_vals = [pulp.value(grid[h]) or 0.0 for h in range(24)]
    solar_vals = [pulp.value(solar[h]) or 0.0 for h in range(24)]
    charge_vals = [pulp.value(charge[h]) or 0.0 for h in range(24)]
    disch_vals = [pulp.value(discharge[h]) or 0.0 for h in range(24)]
    E_vals = [pulp.value(E[h]) or 0.0 for h in range(24)]

    hourly: List[Dict[str, float]] = []
    for h in range(24):
        hourly.append({
            "hour": h,
            "grid_kwh": float(grid_vals[h]),
            "solar_used_kwh": float(solar_vals[h]),
            "charge_kwh": float(charge_vals[h]),
            "disch_kwh": float(disch_vals[h]),
            "battery_energy_after_kwh": float(E_vals[h]),
        })

    total_grid = sum(grid_vals)
    total_cost = sum(hours[h].tariff_bdt_per_kwh * grid_vals[h] for h in range(24))
    peak_grid = max(grid_vals) if grid_vals else 0.0

    return OptimizerResult(
        hourly=hourly,
        total_grid_kwh=float(total_grid),
        total_cost_bdt=float(total_cost),
        peak_grid_kwh=float(peak_grid),
        status="optimal",
    )
