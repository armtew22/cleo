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

from tribe_backend.api.decoding import DecodeError, decode_mp4, probe_video_fps
from tribe_backend.api.deps import get_control_unit, get_settings
from tribe_backend.api.errors import APIError
from tribe_backend.api.jobs import (
    JobStore,
    NotCancellable,
    QueueFull,
    UnknownJob,
)
from tribe_backend.api.schemas import (
    HealthResponse,
    JobStatusResponse,
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


def _decode_request(media: UploadFile, text: str) -> dict:
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

    return {
        "video": video,
        "audio": audio,
        "audio_sr": audio_sr,
        "text": text,
        "video_fps": fps,
    }


@router.post("/runs")
def submit_run(
    request: Request,
    media: Annotated[UploadFile, File(description="mp4 stimulus video")],
    text: Annotated[str, Form(description="caption / transcript for the window")],
    wait: Annotated[bool, Form(description="if true, return Report inline (Phase A sync mode)")] = False,
    cu: ControlUnit = Depends(get_control_unit),
):
    payload = _decode_request(media, text)

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


@router.get("/runs/{job_id}")
def get_run(request: Request, job_id: str):
    jobs = _get_jobs(request)
    import anyio
    try:
        job = anyio.from_thread.run(jobs.get, job_id)
    except UnknownJob:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="JOB_NOT_FOUND",
            message=f"unknown job_id {job_id!r}",
        )

    report_model = None
    if job.report is not None:
        # job.report is the Report.to_json() dict shape; coerce through the
        # response schema so we drop unknown keys / coerce types consistently.
        report_model = ReportResponse.model_validate(job.report)

    body = JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        report=report_model,
        error=job.error,  # type: ignore[arg-type]  (pydantic coerces dict→ErrorInfo)
    ).model_dump(mode="json")
    return JSONResponse(status_code=status.HTTP_200_OK, content=body)


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
