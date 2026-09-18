"""Math optimizer stage: ... -> Guardrail Validator -> **Math Optimizer** ->
Final Validator -> API Response.

Builds and solves the 24-hour cost-minimization LP described in Problem
Statement Sections 05.2, 05.3 and 09, given validated directives from
app.guardrails. Uses PuLP with the bundled CBC solver so the result is a
true optimum (not a heuristic), which is what the Optimization Quality
score (`min(1, organizer_optimal_cost / recalculated_team_cost)`) rewards.
"""
from __future__ import annotations

from dataclasses import dataclass

import pulp

from app.directives import BatteryAction, ParsedDirectives
from app.schemas import BatterySpec, HourEntry, HourPlan

_EPS = 1e-6
# Negligible relative to any real tariff*kWh cost; only breaks ties between
# LP solutions that are otherwise identical in grid cost, steering the
# solver away from a degenerate simultaneous-charge-and-discharge solution
# (which the energy-balance equation permits at zero cost impact on its
# own). The net-charge/discharge post-processing below is the real
# guarantee against reporting simultaneous action; this is defense in depth.
_DEGENERACY_PENALTY = 1e-6


class OptimizationInfeasibleError(Exception):
    """Raised when the LP has no feasible solution. Per the Problem
    Statement, organizer-valid scoring scenarios are always feasible, so
    this should only occur on a malformed/contradictory request -- callers
    must map it to a controlled error response, never a crash."""


@dataclass
class OptimizationResult:
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


def _effective_solar(hours: list[HourEntry], directives: ParsedDirectives) -> list[float]:
    effective = [h.solar_kwh for h in hours]
    for reduction in directives.solar_reductions:
        for hour in reduction.hours:
            effective[hour] = hours[hour].solar_kwh * reduction.factor
    return effective


def _hourly_reserve_floor(battery: BatterySpec, directives: ParsedDirectives) -> list[float]:
    floor = [battery.minimum_energy_kwh] * 24
    for reserve in directives.minimum_battery_reserves:
        for hour in reserve.hours:
            floor[hour] = max(floor[hour], reserve.minimum_energy_kwh)
    return floor


def _hourly_grid_cap(directives: ParsedDirectives) -> dict[int, float]:
    caps: dict[int, float] = {}
    for window in directives.max_grid_windows:
        for hour in window.hours:
            caps[hour] = min(caps.get(hour, float("inf")), window.max_grid_kwh)
    return caps


def solve(hours: list[HourEntry], battery: BatterySpec, directives: ParsedDirectives) -> OptimizationResult:
    n = 24
    demand = [h.demand_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]
    effective_solar = _effective_solar(hours, directives)
    reserve_floor = _hourly_reserve_floor(battery, directives)
    grid_cap = _hourly_grid_cap(directives)

    no_charge_hours = {hour for w in directives.no_charge_windows for hour in w.hours}
    no_discharge_hours = {hour for w in directives.no_discharge_windows for hour in w.hours}

    prob = pulp.LpProblem("gridwise_optimize_energy", pulp.LpMinimize)

    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(n)]
    solar_used = [pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=max(effective_solar[h], 0.0)) for h in range(n)]
    charge = [
        pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=0 if h in no_charge_hours else battery.max_charge_kwh_per_hour)
        for h in range(n)
    ]
    discharge = [
        pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=0 if h in no_discharge_hours else battery.max_discharge_kwh_per_hour)
        for h in range(n)
    ]
    batt_after = [pulp.LpVariable(f"batt_after_{h}", lowBound=reserve_floor[h], upBound=battery.capacity_kwh) for h in range(n)]

    prob += (
        pulp.lpSum(grid[h] * tariff[h] for h in range(n))
        + _DEGENERACY_PENALTY * pulp.lpSum(charge[h] + discharge[h] for h in range(n))
    )

    for h in range(n):
        prev_energy = battery.initial_energy_kwh if h == 0 else batt_after[h - 1]
        prob += batt_after[h] == prev_energy + charge[h] - discharge[h]
        prob += grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h]
        if h in grid_cap:
            prob += grid[h] <= grid_cap[h]

    # End-of-day battery neutrality (Section 09.6).
    prob += batt_after[n - 1] == battery.initial_energy_kwh

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationInfeasibleError(f"LP solver status: {pulp.LpStatus[status]}")

    hourly_plan: list[HourPlan] = []
    prev_after = battery.initial_energy_kwh
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in range(n):
        grid_val = max(0.0, grid[h].value() or 0.0)
        solar_val = max(0.0, solar_used[h].value() or 0.0)
        charge_val = max(0.0, charge[h].value() or 0.0)
        discharge_val = max(0.0, discharge[h].value() or 0.0)
        after_val = batt_after[h].value()
        if after_val is None:
            after_val = prev_after

        # Report a net charge/discharge action so simultaneous nonzero
        # charge+discharge (which the balance equation alone permits at
        # zero cost, independent of whether the degeneracy penalty above
        # fully suppressed it) is never surfaced in the response. The
        # balance equation still holds exactly under this net formulation
        # because reported_charge - reported_discharge == charge_val -
        # discharge_val by construction.
        net = charge_val - discharge_val
        if abs(net) < _EPS:
            action = BatteryAction.IDLE
            action_kwh = 0.0
        elif net > 0:
            action = BatteryAction.CHARGE
            action_kwh = net
        else:
            action = BatteryAction.DISCHARGE
            action_kwh = -net

        hourly_plan.append(
            HourPlan(
                hour=h,
                grid_kwh=round(grid_val, 6),
                solar_used_kwh=round(solar_val, 6),
                battery_action=action,
                battery_kwh=round(action_kwh, 6),
                battery_energy_after_kwh=round(after_val, 6),
            )
        )

        total_grid += grid_val
        total_cost += grid_val * tariff[h]
        peak_grid = max(peak_grid, grid_val)
        prev_after = after_val

    return OptimizationResult(
        hourly_plan=hourly_plan,
        total_grid_kwh=round(total_grid, 6),
        total_cost_bdt=round(total_cost, 6),
        peak_grid_kwh=round(peak_grid, 6),
    )
