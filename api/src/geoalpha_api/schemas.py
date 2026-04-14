"""Pydantic v2 schemas exposed by the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "geoalpha-api"
    version: str


# --- regimes -----------------------------------------------------------

class RegimeRequest(BaseModel):
    series: List[float] = Field(..., min_length=12)
    n_states: int = Field(2, ge=2, le=6)


class RegimeResponse(BaseModel):
    states: List[int]
    posterior: List[List[float]]
    log_likelihood: float
    means: List[float]
    variances: List[float]


class ChangePointRequest(BaseModel):
    series: List[float] = Field(..., min_length=12)
    hazard_lambda: float = 250.0


class ChangePointResponse(BaseModel):
    cp_prob: List[float]


# --- tasking -----------------------------------------------------------

class TaskingTarget(BaseModel):
    name: str
    value: float
    dwell_min: float = 0.0
    dwell_max: float = 60.0
    priority: str = "ROUTINE"


class TaskingRequest(BaseModel):
    targets: List[TaskingTarget]
    total_budget_s: float = Field(120.0, gt=0)
    risk_aversion: float = Field(0.5, ge=0)


class TaskingResponse(BaseModel):
    assignment: dict
    total_value: float
    solver: str


# --- uncertainty -------------------------------------------------------

class UncertaintyRequest(BaseModel):
    toa: List[float] = Field(..., min_length=1)
    n_samples: int = Field(1024, gt=0, le=20000)
    sun_zenith_mean: float = 35.0
    sun_zenith_std: float = 1.5


class UncertaintyResponse(BaseModel):
    summary: dict


# --- anomaly -----------------------------------------------------------

class AnomalyRequest(BaseModel):
    scores: List[float]
    threshold_quantile: float = 0.95
    target_far: float = 1e-4


class AnomalyResponse(BaseModel):
    threshold: float
    xi: float
    sigma: float
    n_alarms: int
