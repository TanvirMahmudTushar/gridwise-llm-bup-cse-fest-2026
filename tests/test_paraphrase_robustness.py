"""Paraphrase-robustness regression tests against the real Groq LLM.

The public sample pack only exercises AM/PM-style phrasing lifted from its
own 10 cases. The Problem Statement explicitly warns that hidden notes "may
paraphrase the same directive" and calls out 24-hour clock notation as one
concrete example of an equivalent phrasing. These hand-authored cases use
wording that does not appear anywhere in the public pack (24-hour clock
times, indirect/passive phrasing, different units, a novel distractor) to
guard against the system prompt in app/llm_interpreter.py overfitting to
the public wording instead of generalizing.

Like tests/test_public_samples.py, this hits the real Groq API and is
skipped automatically without GROQ_API_KEY.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.guardrails import apply_guardrails
from app.main import app
from app.replay_validator import replay_validate
from app.schemas import BatterySpec, HourEntry, HourPlan

from tests.conftest import make_battery, make_flat_hours

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live-LLM paraphrase robustness test",
)

client = TestClient(app)


def _post_and_validate(payload: dict) -> dict:
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    hours = [HourEntry(**h) for h in payload["hours"]]
    battery = BatterySpec(**payload["battery"])
    reparsed = apply_guardrails(
        {"directive_interpretation": body["directive_interpretation"]}, len(payload["operator_notes"]), battery
    )
    plan = [HourPlan(**h) for h in body["hourly_plan"]]
    violations = replay_validate(
        hours, battery, reparsed.directives, plan, body["total_grid_kwh"], body["total_cost_bdt"], body["peak_grid_kwh"]
    )
    assert violations == [], f"self-consistency violations: {violations}"
    return body


def test_24_hour_clock_and_relative_capacity_phrasing():
    payload = {
        "scenario_id": "PARAPHRASE-A",
        "operator_notes": [
            "Solar output will fall to about 10% of forecast between 13:00 and 15:00 for panel maintenance.",
            "Battery must not drop below a third of its capacity from 20:00 to 23:00.",
            "Do not allow the battery to draw grid charge between 03:00 and 05:00.",
        ],
        "hours": make_flat_hours(demand=120, solar=60, tariff=12),
        "battery": make_battery(
            capacity_kwh=300, initial_energy_kwh=150, minimum_energy_kwh=30, max_charge_kwh_per_hour=60, max_discharge_kwh_per_hour=60
        ),
    }
    body = _post_and_validate(payload)
    by_index = {d["note_index"]: d for d in body["directive_interpretation"]}

    assert by_index[0]["directive_type"] == "solar_reduction"
    assert by_index[0]["structured_adjustment"]["hours"] == [13, 14]
    assert abs(by_index[0]["structured_adjustment"]["factor"] - 0.1) < 0.02

    assert by_index[1]["directive_type"] == "minimum_battery_reserve"
    assert by_index[1]["structured_adjustment"]["hours"] == [20, 21, 22]
    assert abs(by_index[1]["structured_adjustment"]["minimum_energy_kwh"] - 100.0) < 1.0  # a third of 300

    assert by_index[2]["directive_type"] == "no_charge_window"
    assert by_index[2]["structured_adjustment"]["hours"] == [3, 4]


def test_indirect_phrasing_different_units_and_novel_distractor():
    payload = {
        "scenario_id": "PARAPHRASE-B",
        "operator_notes": [
            "The battery must hold its charge, not release it, from 16:00 to 18:00.",
            "Utility feed must be throttled to no more than 120 kilowatt-hours per hour between 18:00 and 20:00.",
            "The IT department will migrate the ticketing system to a new server this weekend.",
        ],
        "hours": make_flat_hours(demand=140, solar=30, tariff=14),
        "battery": make_battery(
            capacity_kwh=250, initial_energy_kwh=120, minimum_energy_kwh=25, max_charge_kwh_per_hour=55, max_discharge_kwh_per_hour=55
        ),
    }
    body = _post_and_validate(payload)
    by_index = {d["note_index"]: d for d in body["directive_interpretation"]}

    assert by_index[0]["directive_type"] == "no_discharge_window"
    assert by_index[0]["structured_adjustment"]["hours"] == [16, 17]

    assert by_index[1]["directive_type"] == "max_grid_window"
    assert by_index[1]["structured_adjustment"]["hours"] == [18, 19]
    assert abs(by_index[1]["structured_adjustment"]["max_grid_kwh"] - 120.0) < 1.0

    assert by_index[2]["applies"] is False
    assert by_index[2]["directive_type"] == "no_op"
