"""Unit tests for app.guardrails -- deterministic validation of (untrusted)
LLM output. No network/LLM calls; these feed hand-crafted raw JSON directly.
"""
from __future__ import annotations

from app.directives import DirectiveType
from app.schemas import BatterySpec

from tests.conftest import make_battery

BATTERY = BatterySpec(**make_battery())


def _run(raw, num_notes=1):
    from app.guardrails import apply_guardrails

    return apply_guardrails(raw, num_notes, BATTERY)


def test_valid_solar_reduction_parses_cleanly():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
                "explanation": "cleaning window",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == []
    assert len(result.interpretations) == 1
    assert result.interpretations[0].applies is True
    assert result.interpretations[0].directive_type == DirectiveType.SOLAR_REDUCTION
    assert len(result.directives.solar_reductions) == 1
    assert result.directives.solar_reductions[0].hours == (13, 14)
    assert result.directives.solar_reductions[0].factor == 0.2


def test_no_op_note_normalizes_applies_and_adjustment():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,  # contradicts no_op; guardrail must normalize
                "directive_type": "no_op",
                "structured_adjustment": {"hours": [1]},  # should be discarded
                "explanation": "unrelated note",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == []
    entry = result.interpretations[0]
    assert entry.applies is False
    assert entry.directive_type == DirectiveType.NO_OP
    assert entry.structured_adjustment is None


def test_missing_llm_output_falls_back_to_no_op_for_every_note():
    result = _run(None, num_notes=3)
    assert result.fallback_note_indices == [0, 1, 2]
    assert len(result.interpretations) == 3
    for entry in result.interpretations:
        assert entry.directive_type == DirectiveType.NO_OP
        assert entry.applies is False


def test_unsupported_directive_type_falls_back():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "export_to_grid",  # not a supported type
                "structured_adjustment": {"hours": [1, 2]},
                "explanation": "invented type",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]
    assert result.interpretations[0].directive_type == DirectiveType.NO_OP


def test_out_of_range_factor_falls_back():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [1], "factor": 1.5},
                "explanation": "factor > 1 is invalid",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]


def test_reserve_above_capacity_falls_back():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {"hours": [18], "minimum_energy_kwh": 9999},
                "explanation": "exceeds capacity",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]


def test_hours_are_deduped_and_sorted():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [15, 14, 14, 13]},
                "explanation": "unsorted with a duplicate",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == []
    assert result.directives.no_charge_windows[0].hours == (13, 14, 15)


def test_duplicate_note_index_keeps_first_occurrence_only():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [1]},
                "explanation": "first",
            },
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "no_discharge_window",
                "structured_adjustment": {"hours": [2]},
                "explanation": "duplicate, should be ignored",
            },
        ]
    }
    result = _run(raw)
    assert len(result.interpretations) == 1
    assert result.interpretations[0].directive_type == DirectiveType.NO_CHARGE_WINDOW


def test_hours_out_of_0_23_range_fall_back():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [23, 24], "max_grid_kwh": 100},
                "explanation": "hour 24 is invalid",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]


def test_directive_type_as_non_string_does_not_crash():
    # A dict is unhashable; a naive `x not in allowed_set` membership check
    # would raise TypeError instead of failing safe. Must downgrade to
    # no_op instead of crashing the whole request.
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": {"nested": "object"},
                "structured_adjustment": {"hours": [1]},
                "explanation": "malformed directive_type",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]
    assert result.interpretations[0].directive_type == DirectiveType.NO_OP


def test_infinite_factor_falls_back():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [1], "factor": float("inf")},
                "explanation": "infinite factor",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]


def test_infinite_max_grid_kwh_falls_back():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [1], "max_grid_kwh": float("inf")},
                "explanation": "infinite cap is not a real constraint",
            }
        ]
    }
    result = _run(raw)
    assert result.fallback_note_indices == [0]


def test_multiple_notes_mixed_applicable_and_no_op():
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [18, 19], "max_grid_kwh": 150},
                "explanation": "feeder cap",
            },
            {
                "note_index": 1,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "distractor",
            },
        ]
    }
    result = _run(raw, num_notes=2)
    assert result.fallback_note_indices == []
    assert [i.directive_type for i in result.interpretations] == [
        DirectiveType.MAX_GRID_WINDOW,
        DirectiveType.NO_OP,
    ]
    assert len(result.directives.max_grid_windows) == 1
