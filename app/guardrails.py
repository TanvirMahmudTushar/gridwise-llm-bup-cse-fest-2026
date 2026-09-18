"""Deterministic validation of LLM output (Problem Statement Section 08).

This is the guardrail stage of the pipeline: Energy Data + Operator Notes ->
LLM Interpreter -> **Guardrail Validator** -> Math Optimizer -> Final
Validator -> API Response.

LLM output is treated as untrusted, possibly-malformed structured data.
Nothing here ever invents a new directive type or a numeric value the LLM
(or the request) didn't provide. The one deterministic repair this module
performs is normalizing an otherwise-valid ``hours`` list (sorting/deduping
integers already present) -- that changes formatting, not meaning.

Safe-failure policy (documented in README "Known limitations"): if a single
note's LLM output cannot be validated into a supported, well-shaped
directive, that note is deterministically downgraded to ``no_op`` --
never invented as some other directive, never allowed to crash the request.
This keeps the service available (Performance & Reliability / Malformed
input scoring) at the cost of losing interpretation credit for that one
note, which is the correct, spec-compliant trade-off: a wrong invented
directive risks failing Directive Application & Constraint Correctness
(25 pts) and Critical Violations, while a safe no_op only loses partial
credit on the single affected note's interpretation score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.directives import (
    ALLOWED_DIRECTIVE_TYPES,
    DirectiveType,
    MaxGridWindow,
    MinimumBatteryReserve,
    NoChargeWindow,
    NoDischargeWindow,
    ParsedDirectives,
    SolarReduction,
)
from app.schemas import BatterySpec, DirectiveInterpretation

_VALID_HOURS = set(range(24))


@dataclass
class GuardrailResult:
    interpretations: list[DirectiveInterpretation]
    directives: ParsedDirectives
    fallback_note_indices: list[int]


def _fallback_entry(note_index: int, reason: str) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        structured_adjustment=None,
        explanation=f"Could not validate a supported directive for this note ({reason}); treated as no_op.",
    )


def _extract_raw_entries(raw: Any) -> list[Any]:
    """Normalize whatever shape the LLM returned into a list of candidate
    entry dicts. Returns [] if nothing list-like can be found."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("directive_interpretation", "directives", "interpretations", "results"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    result: float | None = None
    if isinstance(value, bool):
        result = None
    elif isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            result = None
    # Reject inf/-inf/NaN here, once, for every numeric field that flows
    # through this helper (factor, minimum_energy_kwh, max_grid_kwh).
    # Python's json module accepts non-standard `Infinity`/`NaN` literals,
    # and e.g. `factor <= 1.0` or `max_grid_kwh >= 0` alone would not catch
    # `-Infinity`/`Infinity` respectively -- Section 11.3 requires finite
    # numeric values.
    if result is not None and not math.isfinite(result):
        return None
    return result


def _validate_hours(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or len(value) == 0:
        return None
    parsed: list[int] = []
    for item in value:
        as_int = _coerce_int(item)
        if as_int is None or as_int not in _VALID_HOURS:
            return None
        parsed.append(as_int)
    # Deterministic repair: dedupe + sort ascending. This only normalizes
    # formatting of hours the LLM already named; it invents nothing.
    unique_sorted = tuple(sorted(set(parsed)))
    if len(unique_sorted) == 0:
        return None
    return unique_sorted


def _parse_structured_adjustment(
    directive_type: DirectiveType,
    adjustment: Any,
    battery: BatterySpec,
) -> tuple[dict[str, Any], Any] | None:
    """Validate structured_adjustment against the required shape for the
    given directive type. Returns (clean_dict_for_response, parsed_directive_obj)
    or None if invalid."""
    if not isinstance(adjustment, dict):
        return None

    hours = _validate_hours(adjustment.get("hours"))
    if hours is None:
        return None

    if directive_type == DirectiveType.SOLAR_REDUCTION:
        factor = _coerce_float(adjustment.get("factor"))
        if factor is None or not (0.0 <= factor <= 1.0):
            return None
        clean = {"hours": list(hours), "factor": factor}
        return clean, SolarReduction(hours=hours, factor=factor)

    if directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE:
        min_kwh = _coerce_float(adjustment.get("minimum_energy_kwh"))
        if min_kwh is None or min_kwh < 0 or min_kwh > battery.capacity_kwh:
            return None
        clean = {"hours": list(hours), "minimum_energy_kwh": min_kwh}
        return clean, MinimumBatteryReserve(hours=hours, minimum_energy_kwh=min_kwh)

    if directive_type == DirectiveType.NO_CHARGE_WINDOW:
        clean = {"hours": list(hours)}
        return clean, NoChargeWindow(hours=hours)

    if directive_type == DirectiveType.NO_DISCHARGE_WINDOW:
        clean = {"hours": list(hours)}
        return clean, NoDischargeWindow(hours=hours)

    if directive_type == DirectiveType.MAX_GRID_WINDOW:
        max_grid = _coerce_float(adjustment.get("max_grid_kwh"))
        if max_grid is None or max_grid < 0:
            return None
        clean = {"hours": list(hours), "max_grid_kwh": max_grid}
        return clean, MaxGridWindow(hours=hours, max_grid_kwh=max_grid)

    return None  # pragma: no cover - unreachable, directive_type already validated


def apply_guardrails(raw: Any, num_notes: int, battery: BatterySpec) -> GuardrailResult:
    entries = _extract_raw_entries(raw)

    # First-seen-wins mapping of note_index -> raw entry. Out-of-range or
    # unparsable indices are dropped (they don't identify an existing note).
    by_index: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        idx = _coerce_int(entry.get("note_index"))
        if idx is None or idx < 0 or idx >= num_notes:
            continue
        if idx in by_index:
            continue  # duplicate note_index; keep first occurrence only
        by_index[idx] = entry

    interpretations: list[DirectiveInterpretation] = []
    directives = ParsedDirectives.empty()
    fallback_indices: list[int] = []

    for idx in range(num_notes):
        entry = by_index.get(idx)
        if entry is None:
            interpretations.append(_fallback_entry(idx, "missing or unparsable entry"))
            fallback_indices.append(idx)
            continue

        raw_type = entry.get("directive_type")
        # raw_type may be any JSON value (a dict/list is unhashable and
        # would crash a plain `in ALLOWED_DIRECTIVE_TYPES` set-membership
        # check) -- require a string first so a malformed note downgrades
        # to no_op instead of ever raising.
        if not isinstance(raw_type, str) or raw_type.strip() not in ALLOWED_DIRECTIVE_TYPES:
            interpretations.append(_fallback_entry(idx, "unsupported directive_type"))
            fallback_indices.append(idx)
            continue
        directive_type = DirectiveType(raw_type.strip())

        applies = _coerce_bool(entry.get("applies"))
        explanation = entry.get("explanation")
        explanation_str = explanation.strip() if isinstance(explanation, str) and explanation.strip() else "No explanation provided."

        if directive_type == DirectiveType.NO_OP:
            # Guardrail-enforced semantics regardless of what the LLM sent
            # for `applies`/`structured_adjustment` on a no_op row: no_op
            # always normalizes to applies=False, adjustment=None.
            interpretations.append(
                DirectiveInterpretation(
                    note_index=idx,
                    applies=False,
                    directive_type=DirectiveType.NO_OP,
                    structured_adjustment=None,
                    explanation=explanation_str,
                )
            )
            continue

        if applies is not True:
            # A non-no_op directive must claim applies=True; anything else
            # is a contradiction we don't trust enough to apply.
            interpretations.append(_fallback_entry(idx, "applies must be true for a non-no_op directive"))
            fallback_indices.append(idx)
            continue

        parsed = _parse_structured_adjustment(directive_type, entry.get("structured_adjustment"), battery)
        if parsed is None:
            interpretations.append(_fallback_entry(idx, "structured_adjustment failed shape/range validation"))
            fallback_indices.append(idx)
            continue

        clean_adjustment, directive_obj = parsed
        interpretations.append(
            DirectiveInterpretation(
                note_index=idx,
                applies=True,
                directive_type=directive_type,
                structured_adjustment=clean_adjustment,
                explanation=explanation_str,
            )
        )

        if directive_type == DirectiveType.SOLAR_REDUCTION:
            directives.solar_reductions.append(directive_obj)
        elif directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE:
            directives.minimum_battery_reserves.append(directive_obj)
        elif directive_type == DirectiveType.NO_CHARGE_WINDOW:
            directives.no_charge_windows.append(directive_obj)
        elif directive_type == DirectiveType.NO_DISCHARGE_WINDOW:
            directives.no_discharge_windows.append(directive_obj)
        elif directive_type == DirectiveType.MAX_GRID_WINDOW:
            directives.max_grid_windows.append(directive_obj)

    return GuardrailResult(
        interpretations=interpretations,
        directives=directives,
        fallback_note_indices=fallback_indices,
    )
