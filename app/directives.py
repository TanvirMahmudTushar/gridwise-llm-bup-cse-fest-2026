"""Directive types and their validated, parsed representations.

This module defines the six supported directive types (Problem Statement
Section 04) and small dataclasses that hold *already-guardrail-validated*
structured adjustments, ready to be applied by the optimizer. Nothing in
this module trusts LLM output directly -- guardrails.py is the only place
that turns raw/untrusted LLM JSON into these types.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


ALLOWED_DIRECTIVE_TYPES = {d.value for d in DirectiveType}


class BatteryAction(str, Enum):
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


@dataclass(frozen=True)
class SolarReduction:
    hours: tuple[int, ...]
    factor: float


@dataclass(frozen=True)
class MinimumBatteryReserve:
    hours: tuple[int, ...]
    minimum_energy_kwh: float


@dataclass(frozen=True)
class NoChargeWindow:
    hours: tuple[int, ...]


@dataclass(frozen=True)
class NoDischargeWindow:
    hours: tuple[int, ...]


@dataclass(frozen=True)
class MaxGridWindow:
    hours: tuple[int, ...]
    max_grid_kwh: float


@dataclass
class ParsedDirectives:
    """All validated, applicable directives for one scenario, grouped by
    type so the optimizer can apply them independently and additively."""

    solar_reductions: list[SolarReduction]
    minimum_battery_reserves: list[MinimumBatteryReserve]
    no_charge_windows: list[NoChargeWindow]
    no_discharge_windows: list[NoDischargeWindow]
    max_grid_windows: list[MaxGridWindow]

    @staticmethod
    def empty() -> "ParsedDirectives":
        return ParsedDirectives([], [], [], [], [])
