"""Pydantic request/response models matching the GridWise API contract
exactly (Problem Statement Sections 06-10). These are the canonical schema
-- deviating from field names/types here breaks the judge harness."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.directives import DirectiveType, BatteryAction


# ---------------------------------------------------------------------------
# Request models (Section 07)
# ---------------------------------------------------------------------------


class HourEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    # allow_inf_nan=False: plain `ge=0` alone lets `Infinity` through (inf >= 0
    # is true, and Python's json.loads accepts the non-standard `Infinity`/
    # `NaN` literals by default) which would silently poison total_cost_bdt
    # and make the LP unbounded. Section 11.3 requires numeric values to be
    # finite and non-negative -- this enforces "finite" explicitly.
    demand_kwh: float = Field(ge=0, allow_inf_nan=False)
    solar_kwh: float = Field(ge=0, allow_inf_nan=False)
    tariff_bdt_per_kwh: float = Field(ge=0, allow_inf_nan=False)


class BatterySpec(BaseModel):
    capacity_kwh: float = Field(gt=0, allow_inf_nan=False)
    initial_energy_kwh: float = Field(ge=0, allow_inf_nan=False)
    minimum_energy_kwh: float = Field(ge=0, allow_inf_nan=False)
    max_charge_kwh_per_hour: float = Field(ge=0, allow_inf_nan=False)
    max_discharge_kwh_per_hour: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_consistency(self) -> "BatterySpec":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh must not exceed capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh must not exceed capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh must not be below minimum_energy_kwh")
        return self


class OptimizeRequest(BaseModel):
    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourEntry] = Field(min_length=24, max_length=24)
    battery: BatterySpec

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, v: list[str]) -> list[str]:
        for note in v:
            if not note or not note.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def _hours_cover_0_23(cls, v: list[HourEntry]) -> list[HourEntry]:
        hour_values = [h.hour for h in v]
        if sorted(hour_values) != list(range(24)):
            raise ValueError("hours must contain exactly 24 unique entries for hours 0 through 23")
        return v


# ---------------------------------------------------------------------------
# Response models (Section 10)
# ---------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict[str, Any]] = None
    explanation: str


class HourPlan(BaseModel):
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: str = "ok"
