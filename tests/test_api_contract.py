"""API contract tests (Problem Statement Section 06-10). The LLM call is
monkeypatched out so these tests check schema/status-code behavior without
needing network access or a Groq API key -- end-to-end LLM behavior is
covered separately in tests/test_public_samples.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.llm_interpreter import LLMInterpretationError
from app.main import app

from tests.conftest import make_battery, make_flat_hours

client = TestClient(app)


def _valid_payload(notes: list[str] | None = None) -> dict:
    return {
        "scenario_id": "TEST-001",
        "operator_notes": notes or ["The cafeteria menu changes tomorrow."],
        "hours": make_flat_hours(demand=100, solar=20, tariff=10),
        "battery": make_battery(),
    }


def _no_op_llm_output(notes: list[str], battery) -> dict:
    return {
        "directive_interpretation": [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "irrelevant to today's schedule",
            }
            for i in range(len(notes))
        ]
    }


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_optimize_energy_valid_request_shape(monkeypatch):
    monkeypatch.setattr("app.main.interpret_notes", _no_op_llm_output)

    resp = client.post("/optimize-energy", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()

    assert body["scenario_id"] == "TEST-001"
    assert len(body["directive_interpretation"]) == 1
    assert body["directive_interpretation"][0]["directive_type"] == "no_op"
    assert body["directive_interpretation"][0]["applies"] is False
    assert len(body["hourly_plan"]) == 24
    assert {p["hour"] for p in body["hourly_plan"]} == set(range(24))
    assert isinstance(body["total_grid_kwh"], (int, float))
    assert isinstance(body["total_cost_bdt"], (int, float))
    assert isinstance(body["peak_grid_kwh"], (int, float))
    assert isinstance(body["plan_summary"], str) and body["plan_summary"]


def test_optimize_energy_applies_a_real_directive(monkeypatch):
    def fake_interpret(notes, battery):
        return {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [2, 3]},
                    "explanation": "maintenance window",
                }
            ]
        }

    monkeypatch.setattr("app.main.interpret_notes", fake_interpret)

    resp = client.post("/optimize-energy", json=_valid_payload(["Charger maintenance from 2 AM to 4 AM."]))
    assert resp.status_code == 200
    body = resp.json()

    assert body["directive_interpretation"][0]["directive_type"] == "no_charge_window"
    for hour_plan in body["hourly_plan"]:
        if hour_plan["hour"] in (2, 3):
            assert hour_plan["battery_action"] != "charge"


def test_optimize_energy_llm_outage_falls_back_safely(monkeypatch):
    def raise_outage(notes, battery):
        raise LLMInterpretationError("simulated provider outage")

    monkeypatch.setattr("app.main.interpret_notes", raise_outage)

    resp = client.post("/optimize-energy", json=_valid_payload())
    # Safe failure: still a valid 200 response with every note downgraded
    # to no_op, never a crash or a 5xx for an LLM-side outage.
    assert resp.status_code == 200
    body = resp.json()
    assert all(d["directive_type"] == "no_op" for d in body["directive_interpretation"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("battery"),
        lambda p: p.__setitem__("operator_notes", []),
        lambda p: p.__setitem__("operator_notes", ["a", "b", "c", "d"]),
        lambda p: p.__setitem__("hours", p["hours"][:23]),
        lambda p: p.__setitem__("scenario_id", ""),
    ],
)
def test_malformed_requests_return_400(monkeypatch, mutation):
    monkeypatch.setattr("app.main.interpret_notes", _no_op_llm_output)
    payload = _valid_payload()
    mutation(payload)

    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400


def test_infinite_tariff_returns_400(monkeypatch):
    # Python's json module accepts the non-standard `Infinity` literal;
    # a plain `ge=0` constraint alone would let it through (inf >= 0 is
    # true) and poison total_cost_bdt. Must be rejected as malformed.
    import json as _json

    monkeypatch.setattr("app.main.interpret_notes", _no_op_llm_output)
    payload = _valid_payload()
    payload["hours"][5]["tariff_bdt_per_kwh"] = 0  # placeholder, patched below
    raw_body = _json.dumps(payload).replace('"tariff_bdt_per_kwh": 0', '"tariff_bdt_per_kwh": Infinity', 1)

    resp = client.post(
        "/optimize-energy",
        content=raw_body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_malformed_json_body_returns_400():
    resp = client.post(
        "/optimize-energy",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
