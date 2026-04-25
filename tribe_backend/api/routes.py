"""HTTP routes.

Phase A:
    GET  /v1/health
    POST /v1/runs?wait=true   (sync, multipart-only)

Phase B:
    POST   /v1/runs           (default async — 202 + job_id)
    GET    /v1/runs/{job_id}  (status + report-when-done)
    DELETE /v1/runs/{job_id}  (cancel queued)
"""
from __future__ import annotations

import json as _json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from tribe_backend.api import artifacts as _artifacts
from tribe_backend.api.decoding import DecodeError, decode_mp4, probe_video_fps
from tribe_backend.api.deps import get_control_unit, get_settings
from tribe_backend.api.errors import (
    CODE_INFERENCE_ARTIFACTS_MISSING,
    CODE_JOB_FAILED,
    CODE_JOB_NOT_READY,
    APIError,
)
from tribe_backend.api.jobs import (
    Job,
    JobStore,
    NotCancellable,
    QueueFull,
    UnknownJob,
)
from tribe_backend.api.schemas import (
    HealthResponse,
    JobStatusResponse,
    MeshArtifacts,
    MeshManifest,
    ReportResponse,
    SubmitResponse,
)
from tribe_backend.control.unit import ControlUnit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


_ACCEPTED_VIDEO_TYPES = {"video/mp4", "video/webm", "application/octet-stream"}


def _get_jobs(request: Request) -> JobStore:
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:  # pragma: no cover — lifespan always sets this
        raise RuntimeError("JobStore not initialized on app.state")
    return jobs


@router.get("/health", response_model=HealthResponse)
def health(settings=Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        inference=settings.inference,
        out_dir=settings.out_dir,
    )


def _decode_request(media: UploadFile, text: str, z_threshold: float | None = None) -> dict:
    """Read+decode the multipart body into a payload dict the worker can call.

    Raises APIError(400) on bad media. Performed synchronously in the request
    handler so a malformed upload never enqueues a doomed job.
    """
    content_type = (media.content_type or "").lower()
    if content_type and content_type not in _ACCEPTED_VIDEO_TYPES:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="decode_error",
            message=f"unsupported media content-type {content_type!r}; expected video/mp4 or video/webm",
        )

    data = media.file.read()
    try:
        video, audio, audio_sr = decode_mp4(data)
    except DecodeError as e:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="decode_error",
            message=str(e),
        )

    fps = probe_video_fps(data) or 25.0

    if audio.size == 0:
        audio = _silence_for(video, audio_sr)

    payload: dict = {
        "video": video,
        "audio": audio,
        "audio_sr": audio_sr,
        "text": text,
        "video_fps": fps,
    }
    if z_threshold is not None:
        payload["z_threshold"] = z_threshold
    return payload


@router.post("/runs")
def submit_run(
    request: Request,
    media: Annotated[UploadFile, File(description="mp4 stimulus video")],
    text: Annotated[str, Form(description="caption / transcript for the window")],
    wait: Annotated[bool, Form(description="if true, return Report inline (Phase A sync mode)")] = False,
    z_threshold: Annotated[
        float | None,
        Form(
            description=(
                "z-score threshold for ranking parcels in the report. "
                "Default (None) uses the parcellation unit's default of 1.5. "
                "Lower values (e.g. 0.1) surface the long tail of weak activations."
            ),
            ge=0.0,
        ),
    ] = None,
    cu: ControlUnit = Depends(get_control_unit),
):
    # Phase E: refuse `wait=true` in production GPU mode. A real GPU window
    # takes ~5 minutes; a synchronous request would block uvicorn's event
    # loop slot and the user's UI for that whole time. Fake-mode dev wait
    # remains permitted (sub-second) for ergonomic testing.
    if wait:
        settings = get_settings(request)
        if settings.inference == "gpu":
            raise APIError(
                status_code=status.HTTP_400_BAD_REQUEST,
                code="WAIT_DISABLED_GPU",
                message=(
                    "wait=true is disabled when TRIBE_INFERENCE=gpu (5-minute "
                    "blocking request). Submit asynchronously (omit wait or "
                    "set wait=false) and poll GET /v1/runs/{job_id}."
                ),
            )

    payload = _decode_request(media, text, z_threshold=z_threshold)

    if wait:
        # Phase A sync path: run inline, return Report directly.
        report = cu.process_window(**payload)
        if report is None:
            raise APIError(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="report_unavailable",
                message="parcellation produced no report; see server logs",
            )
        body = _json.loads(report.to_json())
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ReportResponse.model_validate(body).model_dump(),
        )

    # Async path: enqueue, return 202.
    jobs = _get_jobs(request)
    try:
        # We can't await here (this handler is sync). FastAPI runs sync routes
        # in a threadpool, so use anyio to bridge into the running event loop.
        import anyio
        job_id = anyio.from_thread.run(jobs.submit, payload)
    except QueueFull as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"code": "QUEUE_FULL", "message": str(e)}},
            headers={"Retry-After": "30"},
        )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=SubmitResponse(job_id=job_id, status="queued").model_dump(),
    )


def _fetch_job(request: Request, job_id: str) -> Job:
    """Look up a job, raising 404 JOB_NOT_FOUND if unknown.

    The id is also pre-validated for shape so a path-traversal attempt
    never reaches the JobStore (and thus never gets close to the filesystem
    via any future code path that consumes the same id).
    """
    if not _artifacts.is_safe_job_id(job_id):
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message=f"unknown job_id {job_id!r}",
        )
    jobs = _get_jobs(request)
    import anyio
    try:
        return anyio.from_thread.run(jobs.get, job_id)
    except UnknownJob:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message=f"unknown job_id {job_id!r}",
        )


def _require_done_job(job: Job) -> str:
    """Map a job's status to either its on-disk window_id (success) or an APIError.

    Returns the disk directory key (== window_id from the report).
    """
    if job.status in ("queued", "running"):
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code=CODE_JOB_NOT_READY,
            message=f"job {job.job_id} is {job.status}; mesh artifacts are not ready",
        )
    if job.status == "failed":
        raise APIError(
            status_code=status.HTTP_410_GONE,
            code=CODE_JOB_FAILED,
            message=f"job {job.job_id} failed; mesh artifacts will never exist",
        )
    if job.status == "cancelled":
        raise APIError(
            status_code=status.HTTP_410_GONE,
            code=CODE_JOB_FAILED,
            message=f"job {job.job_id} was cancelled; mesh artifacts will never exist",
        )
    # status == "done" — pull the on-disk dir key from the report.
    if not job.report or not job.report.get("window_id"):
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=CODE_INFERENCE_ARTIFACTS_MISSING,
            message=f"job {job.job_id} is done but report has no window_id",
        )
    return str(job.report["window_id"])


@router.get("/runs/{job_id}")
def get_run(request: Request, job_id: str):
    job = _fetch_job(request, job_id)

    report_model = None
    mesh_url = None
    if job.report is not None:
        # job.report is the Report.to_json() dict shape; coerce through the
        # response schema so we drop unknown keys / coerce types consistently.
        report_model = ReportResponse.model_validate(job.report)
    if job.status == "done":
        mesh_url = f"/v1/runs/{job.job_id}/mesh"

    body = JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        report=report_model,
        error=job.error,  # type: ignore[arg-type]  (pydantic coerces dict→ErrorInfo)
        mesh=mesh_url,
    ).model_dump(mode="json")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=body,
        headers={"Cache-Control": _artifacts.JSON_CACHE_HEADER},
    )


# ----------------------------------------------------------------- Phase C
# Mesh artifact endpoints: per-job manifest + four GETs serving the binary
# / JSON files written by ControlUnit's mesh exporter.

def _resolve_artifact_path(request: Request, job_id: str, name: str):
    """Run the full job → window_id → on-disk path resolution chain.

    Raises APIError on any of the documented failure modes:
        - 404 JOB_NOT_FOUND        (unknown / unsafe job_id)
        - 409 JOB_NOT_READY        (still queued or running)
        - 410 JOB_FAILED           (failed or cancelled)
        - 500 INFERENCE_ARTIFACTS_MISSING (done but missing on disk)
    """
    job = _fetch_job(request, job_id)
    window_id = _require_done_job(job)

    settings = get_settings(request)
    try:
        path = _artifacts.artifact_path(settings.out_dir, window_id, name)
    except _artifacts.UnsafeArtifactRequest:
        # window_id should already be safe (it comes from our own ControlUnit
        # output), but treat it as 404 if it's not — defense in depth.
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message=f"unknown job_id {job_id!r}",
        )
    if not path.is_file():
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=CODE_INFERENCE_ARTIFACTS_MISSING,
            message=f"artifact {name!r} for job {job.job_id} not present on disk",
        )
    return path


@router.get("/runs/{job_id}/mesh/meta")
def get_mesh_meta(request: Request, job_id: str):
    """Parsed brain_meta.json for the job. JSON; no caching (in-flight ref)."""
    path = _resolve_artifact_path(request, job_id, "brain_meta.json")
    body = _json.loads(path.read_text())
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=body,
        headers={"Cache-Control": _artifacts.JSON_CACHE_HEADER},
    )


@router.get("/runs/{job_id}/mesh/colors")
def get_mesh_colors(request: Request, job_id: str):
    path = _resolve_artifact_path(request, job_id, "brain_colors.bin")
    return _artifacts.binary_file_response(path, "brain_colors.bin")


@router.get("/runs/{job_id}/mesh/vertices")
def get_mesh_vertices(request: Request, job_id: str):
    path = _resolve_artifact_path(request, job_id, "brain_vertices.bin")
    return _artifacts.binary_file_response(path, "brain_vertices.bin")


@router.get("/runs/{job_id}/mesh/faces")
def get_mesh_faces(request: Request, job_id: str):
    path = _resolve_artifact_path(request, job_id, "brain_faces.bin")
    return _artifacts.binary_file_response(path, "brain_faces.bin")


@router.get("/runs/{job_id}/mesh", response_model=MeshManifest)
def get_mesh_manifest(request: Request, job_id: str):
    """Per-job artifact manifest. Requires status == done."""
    job = _fetch_job(request, job_id)
    _require_done_job(job)  # reject queued/running/failed/cancelled
    body = MeshManifest(
        job_id=job.job_id,
        artifacts=MeshArtifacts(
            meta=f"/v1/runs/{job.job_id}/mesh/meta",
            colors=f"/v1/runs/{job.job_id}/mesh/colors",
            vertices=f"/v1/runs/{job.job_id}/mesh/vertices",
            faces=f"/v1/runs/{job.job_id}/mesh/faces",
        ),
    ).model_dump()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=body,
        headers={"Cache-Control": _artifacts.JSON_CACHE_HEADER},
    )


@router.delete("/runs/{job_id}")
def cancel_run(request: Request, job_id: str):
    jobs = _get_jobs(request)
    import anyio
    try:
        anyio.from_thread.run(jobs.cancel, job_id)
    except UnknownJob:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message=f"unknown job_id {job_id!r}",
        )
    except NotCancellable as e:
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="JOB_NOT_CANCELLABLE",
            message=str(e),
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _silence_for(video, audio_sr: int):
    """Build a silent audio buffer matching the video duration.

    StimulusWindow.duration_s is derived inside ControlUnit from the audio
    length, and must fall in [25, 35]s. If the source mp4 has no audio
    track, synthesize silence so the contract holds.
    """
    import numpy as np
    n_frames = video.shape[0]
    duration_s = max(25.0, min(35.0, n_frames / 25.0))
    return np.zeros(int(duration_s * audio_sr), dtype=np.float32)
