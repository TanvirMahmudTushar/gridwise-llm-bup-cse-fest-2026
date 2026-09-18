from __future__ import annotations

import json
from pathlib import Path

import pytest

SAMPLE_CASES_PATH = (
    Path(__file__).resolve().parent.parent
    / "Problem Statement"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


@pytest.fixture(scope="session")
def sample_cases() -> list[dict]:
    data = json.loads(SAMPLE_CASES_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def make_hour_entry(hour: int, demand: float, solar: float, tariff: float) -> dict:
    return {"hour": hour, "demand_kwh": demand, "solar_kwh": solar, "tariff_bdt_per_kwh": tariff}


def make_flat_hours(demand: float = 100.0, solar: float = 0.0, tariff: float = 10.0) -> list[dict]:
    """24 identical hours -- useful for isolated directive unit tests where
    the exact demand/solar/tariff shape doesn't matter."""
    return [make_hour_entry(h, demand, solar, tariff) for h in range(24)]


def make_battery(
    capacity_kwh: float = 200.0,
    initial_energy_kwh: float = 100.0,
    minimum_energy_kwh: float = 20.0,
    max_charge_kwh_per_hour: float = 50.0,
    max_discharge_kwh_per_hour: float = 50.0,
) -> dict:
    return {
        "capacity_kwh": capacity_kwh,
        "initial_energy_kwh": initial_energy_kwh,
        "minimum_energy_kwh": minimum_energy_kwh,
        "max_charge_kwh_per_hour": max_charge_kwh_per_hour,
        "max_discharge_kwh_per_hour": max_discharge_kwh_per_hour,
    }
