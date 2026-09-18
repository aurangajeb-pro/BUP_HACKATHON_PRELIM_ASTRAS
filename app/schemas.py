"""
Pydantic schemas for GridWise — BUP CSE Fest 2026.

These mirror the official sample case schema (see
BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json `_meta.schema_notes`).

The strictness here is deliberate: Pydantic's own validation rejects malformed
input *before* it reaches the optimizer.
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


# ----- Allowed enum values (per spec _meta.allowed_enums) -----

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


# ----- Hour, Battery, Request -----

class HourEntry(StrictModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class Battery(StrictModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @field_validator("minimum_energy_kwh")
    @classmethod
    def _min_le_capacity(cls, v: float, info) -> float:
        cap = info.data.get("capacity_kwh")
        if cap is not None and v > cap:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        return v

    @model_validator(mode="after")
    def _init_in_range(self) -> "Battery":
        # Field validators cannot see fields declared later in the class.
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError("initial_energy_kwh must be between minimum_energy_kwh and capacity_kwh")
        return self


class OptimizeRequest(StrictModel):
    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry]
    battery: Battery

    @field_validator("hours")
    @classmethod
    def _hours_complete(cls, v: List[HourEntry]) -> List[HourEntry]:
        if len(v) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        seen = sorted(h.hour for h in v)
        if seen != list(range(24)):
            raise ValueError("hours must cover 0..23 exactly once each")
        return sorted(v, key=lambda entry: entry.hour)

    @field_validator("operator_notes")
    @classmethod
    def _notes_nonempty(cls, v: List[str]) -> List[str]:
        for n in v:
            if not n or not n.strip():
                raise ValueError("operator_notes must be non-empty strings")
        return v


# ----- Structured adjustments (tagged union, discriminator = kind) -----

class _AdjustmentBase(StrictModel):
    hours: List[int]

    @field_validator("hours")
    @classmethod
    def _hours_valid(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("hours list must not be empty")
        if sorted(v) != v:
            raise ValueError("hours must be ascending")
        if len(set(v)) != len(v):
            raise ValueError("hours must be unique")
        for h in v:
            if not (0 <= h <= 23):
                raise ValueError(f"hour {h} out of range 0..23")
        return v


class SolarReduction(_AdjustmentBase):
    kind: Literal["solar_reduction"] = "solar_reduction"
    factor: float = Field(ge=0.0, le=1.0)


class MinimumBatteryReserve(_AdjustmentBase):
    kind: Literal["minimum_battery_reserve"] = "minimum_battery_reserve"
    minimum_energy_kwh: float = Field(ge=0)


class NoChargeWindow(_AdjustmentBase):
    kind: Literal["no_charge_window"] = "no_charge_window"


class NoDischargeWindow(_AdjustmentBase):
    kind: Literal["no_discharge_window"] = "no_discharge_window"


class MaxGridWindow(_AdjustmentBase):
    kind: Literal["max_grid_window"] = "max_grid_window"
    max_grid_kwh: float = Field(ge=0)


StructuredAdjustment = Annotated[
    Union[
        SolarReduction,
        MinimumBatteryReserve,
        NoChargeWindow,
        NoDischargeWindow,
        MaxGridWindow,
    ],
    Field(discriminator="kind"),
]


# ----- Directive interpretation, hourly plan, response -----

class DirectiveInterpretation(StrictModel):
    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[StructuredAdjustment]
    explanation: str = Field(min_length=1)


class HourlyPlanRow(StrictModel):
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)


class OptimizeResponse(StrictModel):
    scenario_id: str = Field(min_length=1)
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanRow]
    total_grid_kwh: float = Field(ge=0)
    total_cost_bdt: float = Field(ge=0)
    peak_grid_kwh: float = Field(ge=0)
    plan_summary: str = Field(min_length=1)

    @field_validator("hourly_plan")
    @classmethod
    def _plan_complete(cls, v: List[HourlyPlanRow]) -> List[HourlyPlanRow]:
        if len(v) != 24:
            raise ValueError("hourly_plan must contain exactly 24 entries")
        seen = sorted(r.hour for r in v)
        if seen != list(range(24)):
            raise ValueError("hourly_plan must cover 0..23 exactly once each")
        return v
