"""Pydantic schemas for the HTTP surface.

Phase A: HealthResponse, ReportResponse (multipart-only Report payload).
Phase B: SubmitResponse (202 body), JobStatusResponse, ErrorEnvelope, ErrorInfo.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

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


# ---------------------------------------------------------- Phase B schemas

class SubmitResponse(BaseModel):
    """202 Accepted body for async POST /v1/runs."""
    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: Literal["queued"] = "queued"


class ErrorInfo(BaseModel):
    code: str
    message: str


class JobStatusResponse(BaseModel):
    """Body for GET /v1/runs/{job_id}.

    `report` is populated only when status == "done".
    `error` is populated only when status == "failed".
    `mesh` is the URL of the mesh manifest, populated only when status == "done".
    """
    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    submitted_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    report: Optional[ReportResponse] = None
    error: Optional[ErrorInfo] = None
    mesh: Optional[str] = None


# ---------------------------------------------------------- Phase C schemas

class MeshArtifacts(BaseModel):
    """The four artifact URLs enumerated by the per-job mesh manifest."""
    model_config = ConfigDict(extra="forbid")

    meta: str
    colors: str
    vertices: str
    faces: str


class MeshManifest(BaseModel):
    """Body for GET /v1/runs/{job_id}/mesh.

    Lets the frontend discover artifact URLs from a job_id without
    hand-constructing them.
    """
    model_config = ConfigDict(extra="forbid")

    job_id: str
    artifacts: MeshArtifacts
