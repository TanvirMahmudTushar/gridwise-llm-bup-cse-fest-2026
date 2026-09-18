"""Unit tests for app.optimizer -- the PuLP LP model -- exercising each
directive type in isolation, plus a no-directive baseline and an
infeasibility case. Every feasible case is cross-checked with the
independent app.replay_validator, so a bug can't hide by only existing in
one of the two modules.
"""
from __future__ import annotations

import pytest

from app.directives import (
    MaxGridWindow,
    MinimumBatteryReserve,
    NoChargeWindow,
    NoDischargeWindow,
    ParsedDirectives,
    SolarReduction,
)
from app.optimizer import OptimizationInfeasibleError, solve
from app.replay_validator import replay_validate
from app.schemas import BatterySpec, HourEntry

from tests.conftest import make_battery, make_flat_hours


def _hours(entries: list[dict]) -> list[HourEntry]:
    return [HourEntry(**e) for e in entries]


def _assert_no_violations(hours, battery, directives, result):
    violations = replay_validate(
        hours,
        battery,
        directives,
        result.hourly_plan,
        result.total_grid_kwh,
        result.total_cost_bdt,
        result.peak_grid_kwh,
    )
    assert violations == []


def test_baseline_no_directives_is_valid_and_optimal_shape():
    hours = _hours(make_flat_hours(demand=100, solar=0, tariff=10))
    battery = BatterySpec(**make_battery())
    directives = ParsedDirectives.empty()

    result = solve(hours, battery, directives)

    assert len(result.hourly_plan) == 24
    _assert_no_violations(hours, battery, directives, result)
    # No solar, flat tariff, so cost is exactly demand * tariff regardless
    # of battery shuffling (grid always ends up meeting demand net of
    # battery neutrality over the day).
    assert result.total_grid_kwh == pytest.approx(2400, abs=0.01)


def test_no_charge_window_forces_zero_charge_in_listed_hours():
    # Cheap early hours would normally be used to pre-charge the battery
    # before an expensive evening -- block hour 2 specifically.
    entries = make_flat_hours(demand=100, solar=0, tariff=5)
    for h in (18, 19, 20):
        entries[h]["tariff_bdt_per_kwh"] = 30
    hours = _hours(entries)
    battery = BatterySpec(**make_battery(initial_energy_kwh=60, minimum_energy_kwh=20))
    directives = ParsedDirectives.empty()
    directives.no_charge_windows.append(NoChargeWindow(hours=(2,)))

    result = solve(hours, battery, directives)
    _assert_no_violations(hours, battery, directives, result)

    plan_hour_2 = next(p for p in result.hourly_plan if p.hour == 2)
    assert plan_hour_2.battery_action != "charge"


def test_no_discharge_window_forces_zero_discharge_in_listed_hours():
    entries = make_flat_hours(demand=100, solar=0, tariff=5)
    for h in (18, 19):
        entries[h]["tariff_bdt_per_kwh"] = 30  # normally worth discharging here
    hours = _hours(entries)
    battery = BatterySpec(**make_battery(initial_energy_kwh=150))
    directives = ParsedDirectives.empty()
    directives.no_discharge_windows.append(NoDischargeWindow(hours=(18, 19)))

    result = solve(hours, battery, directives)
    _assert_no_violations(hours, battery, directives, result)

    for h in (18, 19):
        plan_hour = next(p for p in result.hourly_plan if p.hour == h)
        assert plan_hour.battery_action != "discharge"


def test_solar_reduction_caps_solar_used_in_listed_hours():
    entries = make_flat_hours(demand=100, solar=80, tariff=10)
    hours = _hours(entries)
    battery = BatterySpec(**make_battery())
    directives = ParsedDirectives.empty()
    directives.solar_reductions.append(SolarReduction(hours=(10, 11), factor=0.25))

    result = solve(hours, battery, directives)
    _assert_no_violations(hours, battery, directives, result)

    for h in (10, 11):
        plan_hour = next(p for p in result.hourly_plan if p.hour == h)
        assert plan_hour.solar_used_kwh <= 20 + 0.01  # 80 * 0.25
    # An hour outside the window should be able to use full solar.
    plan_hour_12 = next(p for p in result.hourly_plan if p.hour == 12)
    assert plan_hour_12.solar_used_kwh <= 80 + 0.01


def test_minimum_battery_reserve_raises_floor_in_listed_hours():
    entries = make_flat_hours(demand=100, solar=0, tariff=10)
    hours = _hours(entries)
    battery = BatterySpec(**make_battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=20))
    directives = ParsedDirectives.empty()
    directives.minimum_battery_reserves.append(MinimumBatteryReserve(hours=(18, 19, 20), minimum_energy_kwh=100))

    result = solve(hours, battery, directives)
    _assert_no_violations(hours, battery, directives, result)

    for h in (18, 19, 20):
        plan_hour = next(p for p in result.hourly_plan if p.hour == h)
        assert plan_hour.battery_energy_after_kwh >= 100 - 0.01


def test_max_grid_window_caps_grid_in_listed_hours():
    entries = make_flat_hours(demand=150, solar=0, tariff=10)
    hours = _hours(entries)
    battery = BatterySpec(**make_battery(capacity_kwh=300, initial_energy_kwh=150, max_charge_kwh_per_hour=80, max_discharge_kwh_per_hour=80))
    directives = ParsedDirectives.empty()
    directives.max_grid_windows.append(MaxGridWindow(hours=(12,), max_grid_kwh=100))

    result = solve(hours, battery, directives)
    _assert_no_violations(hours, battery, directives, result)

    plan_hour_12 = next(p for p in result.hourly_plan if p.hour == 12)
    assert plan_hour_12.grid_kwh <= 100 + 0.01


def test_infeasible_max_grid_window_raises_controlled_error():
    entries = make_flat_hours(demand=100, solar=0, tariff=10)
    hours = _hours(entries)
    # Battery can discharge at most 50 kWh/hr; a 0 kWh grid cap with 100
    # kWh demand and no solar cannot be satisfied.
    battery = BatterySpec(**make_battery(max_discharge_kwh_per_hour=50))
    directives = ParsedDirectives.empty()
    directives.max_grid_windows.append(MaxGridWindow(hours=(0,), max_grid_kwh=0))

    with pytest.raises(OptimizationInfeasibleError):
        solve(hours, battery, directives)


def test_combined_directives_all_hold_simultaneously():
    entries = make_flat_hours(demand=120, solar=40, tariff=10)
    for h in (18, 19, 20):
        entries[h]["tariff_bdt_per_kwh"] = 25
        entries[h]["solar_kwh"] = 0
    hours = _hours(entries)
    battery = BatterySpec(**make_battery(capacity_kwh=250, initial_energy_kwh=130, max_charge_kwh_per_hour=60, max_discharge_kwh_per_hour=60))
    directives = ParsedDirectives.empty()
    directives.minimum_battery_reserves.append(MinimumBatteryReserve(hours=(18, 19, 20), minimum_energy_kwh=70))
    directives.max_grid_windows.append(MaxGridWindow(hours=(19, 20), max_grid_kwh=110))
    directives.no_charge_windows.append(NoChargeWindow(hours=(11, 12)))

    result = solve(hours, battery, directives)
    _assert_no_violations(hours, battery, directives, result)
