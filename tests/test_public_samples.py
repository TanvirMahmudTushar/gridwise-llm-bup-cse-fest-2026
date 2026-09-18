"""End-to-end validation against the 10 public sample cases (Problem
Statement Section 11 / equivalence_note). Exercises the real Groq LLM call,
so it's skipped automatically when GROQ_API_KEY isn't configured (e.g. in a
CI environment without credentials) -- run it locally with a real key
before submitting.

Per the sample pack's own instructions, this does NOT byte-for-byte compare
against expected_output: it (1) replays the returned hourly_plan against
the service's own reported directive_interpretation to check
self-consistency, (2) compares directive_interpretation semantics
(applies/type/hours/numeric values) against each case's ground truth within
tolerance, and (3) compares total_cost_bdt against an independently
recomputed optimum for the ground-truth directives, not against the
packaged reference numbers directly.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.config import NUMERIC_TOLERANCE
from app.guardrails import apply_guardrails
from app.main import app
from app.optimizer import solve
from app.replay_validator import replay_validate
from app.schemas import BatterySpec, HourEntry

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping live-LLM public sample validation",
)

client = TestClient(app)


def _parse_hours_and_battery(case_input: dict) -> tuple[list[HourEntry], BatterySpec]:
    hours = [HourEntry(**h) for h in case_input["hours"]]
    battery = BatterySpec(**case_input["battery"])
    return hours, battery


def test_all_public_sample_cases(sample_cases):
    failures: list[str] = []

    for case in sample_cases:
        case_id = case["id"]
        case_input = case["input"]
        expected = case["expected_output"]

        resp = client.post("/optimize-energy", json=case_input)
        if resp.status_code != 200:
            failures.append(f"{case_id}: HTTP {resp.status_code} - {resp.text[:200]}")
            continue
        body = resp.json()

        hours, battery = _parse_hours_and_battery(case_input)
        num_notes = len(case_input["operator_notes"])

        # (1) Self-consistency: replay the reported hourly_plan against the
        # service's own reported directive_interpretation.
        reparsed = apply_guardrails({"directive_interpretation": body["directive_interpretation"]}, num_notes, battery)
        hourly_plan = [_dict_to_hourplan(h) for h in body["hourly_plan"]]
        violations = replay_validate(
            hours,
            battery,
            reparsed.directives,
            hourly_plan,
            body["total_grid_kwh"],
            body["total_cost_bdt"],
            body["peak_grid_kwh"],
        )
        if violations:
            failures.append(f"{case_id}: self-consistency violations: {violations}")

        # (2) Directive semantics vs. ground truth, within tolerance.
        semantic_errors = _compare_directive_semantics(
            body["directive_interpretation"], expected["directive_interpretation"]
        )
        if semantic_errors:
            failures.append(f"{case_id}: interpretation mismatch: {semantic_errors}")

        # (3) Cost equivalence: recompute the optimum for the GROUND TRUTH
        # directives ourselves and compare total_cost_bdt within a small
        # tolerance, rather than trusting the packaged reference cost.
        ground_truth_guardrails = apply_guardrails(
            {"directive_interpretation": expected["directive_interpretation"]}, num_notes, battery
        )
        ground_truth_optimum = solve(hours, battery, ground_truth_guardrails.directives)
        cost_gap = body["total_cost_bdt"] - ground_truth_optimum.total_cost_bdt
        if cost_gap > 1.0:  # team cost must not exceed the true optimum (allow tiny numeric slack)
            failures.append(
                f"{case_id}: total_cost_bdt {body['total_cost_bdt']} exceeds recomputed optimum "
                f"{ground_truth_optimum.total_cost_bdt} by {cost_gap}"
            )

    assert not failures, "\n".join(failures)


def _dict_to_hourplan(d: dict):
    from app.schemas import HourPlan

    return HourPlan(**d)


def _compare_directive_semantics(actual: list[dict], expected: list[dict]) -> list[str]:
    errors: list[str] = []
    actual_by_index = {a["note_index"]: a for a in actual}

    for exp in expected:
        idx = exp["note_index"]
        act = actual_by_index.get(idx)
        if act is None:
            errors.append(f"note {idx}: missing from response")
            continue
        if act["applies"] != exp["applies"]:
            errors.append(f"note {idx}: applies {act['applies']} != expected {exp['applies']}")
        if act["directive_type"] != exp["directive_type"]:
            errors.append(f"note {idx}: directive_type {act['directive_type']} != expected {exp['directive_type']}")
            continue
        if exp["directive_type"] == "no_op":
            continue

        exp_adj = exp["structured_adjustment"] or {}
        act_adj = act["structured_adjustment"] or {}
        if act_adj.get("hours") != exp_adj.get("hours"):
            errors.append(f"note {idx}: hours {act_adj.get('hours')} != expected {exp_adj.get('hours')}")
        for numeric_field in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if numeric_field in exp_adj:
                act_val = act_adj.get(numeric_field)
                if act_val is None or abs(act_val - exp_adj[numeric_field]) > max(NUMERIC_TOLERANCE, 0.02):
                    errors.append(f"note {idx}: {numeric_field} {act_val} != expected {exp_adj[numeric_field]}")

    return errors
