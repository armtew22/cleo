"""Pydantic schemas for the Phase A HTTP surface (multipart only)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RegionActivationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    z_score: float
    direction: str
    description: str = ""


class ReportResponse(BaseModel):
    """Mirror of `tribe_backend.parcellation.unit.Report.to_json()` shape."""

    model_config = ConfigDict(extra="ignore")

    top_regions: list[RegionActivationResponse] = Field(default_factory=list)
    text: str
    method: str
    z_threshold: float
    window_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    inference: str
    out_dir: str
