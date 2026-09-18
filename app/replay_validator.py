"""Final validator stage: ... -> Math Optimizer -> **Final Validator** ->
API Response.

Independently replays the optimizer's own output hour-by-hour against the
GridWise energy/battery rules (Problem Statement Section 09) and every
validated directive, without reusing any optimizer internals. This is
defense in depth against an optimizer bug: if this module ever finds a
violation, the service has produced an invalid plan and must fail in a
controlled way (Section 08 "SAFE FAILURE") rather than return it.

It also doubles as the same logic tests/test_public_samples.py uses to
independently check hourly_plan correctness, so a bug can't hide by only
existing in one place.
"""
from __future__ import annotations

from app.directives import BatteryAction, ParsedDirectives
from app.schemas import BatterySpec, HourEntry, HourPlan

_TOL = 0.01  # Problem Statement Section 11.5 numeric tolerance.


def _effective_solar(hours: list[HourEntry], directives: ParsedDirectives) -> list[float]:
    effective = [h.solar_kwh for h in hours]
    # Must match app/optimizer.py's combination rule exactly (strictest
    # factor wins on overlap) or this independent replay check would
    # disagree with the optimizer and flag a false violation.
    factor_by_hour: dict[int, float] = {}
    for reduction in directives.solar_reductions:
        for hour in reduction.hours:
            factor_by_hour[hour] = min(factor_by_hour.get(hour, 1.0), reduction.factor)
    for hour, factor in factor_by_hour.items():
        effective[hour] = hours[hour].solar_kwh * factor
    return effective


def _reserve_floor(battery: BatterySpec, directives: ParsedDirectives) -> list[float]:
    floor = [battery.minimum_energy_kwh] * 24
    for reserve in directives.minimum_battery_reserves:
        for hour in reserve.hours:
            floor[hour] = max(floor[hour], reserve.minimum_energy_kwh)
    return floor


def replay_validate(
    hours: list[HourEntry],
    battery: BatterySpec,
    directives: ParsedDirectives,
    hourly_plan: list[HourPlan],
    reported_total_grid_kwh: float,
    reported_total_cost_bdt: float,
    reported_peak_grid_kwh: float,
) -> list[str]:
    violations: list[str] = []

    plan_by_hour = {p.hour: p for p in hourly_plan}
    if len(hourly_plan) != 24 or set(plan_by_hour.keys()) != set(range(24)):
        violations.append("hourly_plan does not contain exactly 24 unique hours 0-23")
        return violations  # further checks would be meaningless

    effective_solar = _effective_solar(hours, directives)
    reserve_floor = _reserve_floor(battery, directives)

    no_charge_hours = {hour for w in directives.no_charge_windows for hour in w.hours}
    no_discharge_hours = {hour for w in directives.no_discharge_windows for hour in w.hours}
    grid_cap: dict[int, float] = {}
    for window in directives.max_grid_windows:
        for hour in window.hours:
            grid_cap[hour] = min(grid_cap.get(hour, float("inf")), window.max_grid_kwh)

    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0
    prev_after = battery.initial_energy_kwh

    for h in range(24):
        p = plan_by_hour[h]
        demand = hours[h].demand_kwh
        tariff = hours[h].tariff_bdt_per_kwh

        if p.solar_used_kwh > effective_solar[h] + _TOL:
            violations.append(f"hour {h}: solar_used_kwh {p.solar_used_kwh} exceeds effective solar {effective_solar[h]}")

        if p.battery_action == BatteryAction.CHARGE:
            charge_amt, discharge_amt = p.battery_kwh, 0.0
            if charge_amt > battery.max_charge_kwh_per_hour + _TOL:
                violations.append(f"hour {h}: charge {charge_amt} exceeds max_charge_kwh_per_hour")
            if h in no_charge_hours and charge_amt > _TOL:
                violations.append(f"hour {h}: charging occurred during a no_charge_window hour")
        elif p.battery_action == BatteryAction.DISCHARGE:
            charge_amt, discharge_amt = 0.0, p.battery_kwh
            if discharge_amt > battery.max_discharge_kwh_per_hour + _TOL:
                violations.append(f"hour {h}: discharge {discharge_amt} exceeds max_discharge_kwh_per_hour")
            if h in no_discharge_hours and discharge_amt > _TOL:
                violations.append(f"hour {h}: discharging occurred during a no_discharge_window hour")
        else:
            charge_amt, discharge_amt = 0.0, 0.0
            if p.battery_kwh > _TOL:
                violations.append(f"hour {h}: battery_kwh must be 0 when idle")

        expected_after = prev_after + charge_amt - discharge_amt
        if abs(p.battery_energy_after_kwh - expected_after) > _TOL:
            violations.append(
                f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} != expected {expected_after}"
            )

        floor = reserve_floor[h]
        if p.battery_energy_after_kwh < floor - _TOL:
            violations.append(f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} below reserve floor {floor}")
        if p.battery_energy_after_kwh > battery.capacity_kwh + _TOL:
            violations.append(f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} exceeds capacity")

        balance_lhs = p.grid_kwh + p.solar_used_kwh + discharge_amt
        balance_rhs = demand + charge_amt
        if abs(balance_lhs - balance_rhs) > _TOL:
            violations.append(f"hour {h}: energy balance violated ({balance_lhs} != {balance_rhs})")

        if h in grid_cap and p.grid_kwh > grid_cap[h] + _TOL:
            violations.append(f"hour {h}: grid_kwh {p.grid_kwh} exceeds max_grid_window cap {grid_cap[h]}")

        total_grid += p.grid_kwh
        total_cost += p.grid_kwh * tariff
        peak_grid = max(peak_grid, p.grid_kwh)
        prev_after = p.battery_energy_after_kwh

    if abs(prev_after - battery.initial_energy_kwh) > _TOL:
        violations.append(f"end-of-day battery energy {prev_after} != initial_energy_kwh {battery.initial_energy_kwh}")

    if abs(total_grid - reported_total_grid_kwh) > _TOL:
        violations.append(f"reported total_grid_kwh {reported_total_grid_kwh} != recalculated {total_grid}")
    if abs(total_cost - reported_total_cost_bdt) > _TOL:
        violations.append(f"reported total_cost_bdt {reported_total_cost_bdt} != recalculated {total_cost}")
    if abs(peak_grid - reported_peak_grid_kwh) > _TOL:
        violations.append(f"reported peak_grid_kwh {reported_peak_grid_kwh} != recalculated {peak_grid}")

    return violations
