"""HTTP routes — Phase A: GET /v1/health and POST /v1/runs (sync, multipart)."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from tribe_backend.api.decoding import DecodeError, decode_mp4, probe_video_fps
from tribe_backend.api.deps import get_control_unit, get_settings
from tribe_backend.api.errors import APIError
from tribe_backend.api.schemas import HealthResponse, ReportResponse
from tribe_backend.control.unit import ControlUnit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


_ACCEPTED_VIDEO_TYPES = {"video/mp4", "video/webm", "application/octet-stream"}


@router.get("/health", response_model=HealthResponse)
def health(settings=Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        inference=settings.inference,
        out_dir=settings.out_dir,
    )


@router.post("/runs", response_model=ReportResponse)
def submit_run(
    media: Annotated[UploadFile, File(description="mp4 stimulus video")],
    text: Annotated[str, Form(description="caption / transcript for the window")],
    wait: Annotated[bool, Form(description="must be true in Phase A")] = True,
    cu: ControlUnit = Depends(get_control_unit),
):
    if not wait:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="async_not_implemented",
            message="async mode lands in Phase B; pass wait=true",
        )

    # Validate content-type quickly so we don't burn CPU on garbage.
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

    report = cu.process_window(
        video=video,
        audio=audio if audio.size > 0 else _silence_for(video, audio_sr),
        audio_sr=audio_sr,
        text=text,
        video_fps=fps,
    )

    if report is None:
        # Parcellation failed inside ControlUnit (failure-isolated). Surface a
        # clean 500 so the frontend retries instead of getting a stale empty.
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="report_unavailable",
            message="parcellation produced no report; see server logs",
        )

    # Serialize via the dataclass's own to_json() so the wire shape matches
    # the offline pipeline byte-for-byte.
    import json as _json
    payload = _json.loads(report.to_json())
    return ReportResponse.model_validate(payload)


def _silence_for(video, audio_sr: int):
    """Build a silent audio buffer matching the video duration.

    StimulusWindow.duration_s is derived inside ControlUnit from the audio
    length, and must fall in [25, 35]s. If the source mp4 has no audio
    track, synthesize silence so the contract holds.
    """
    import numpy as np
    n_frames = video.shape[0]
    # Best effort: assume 25fps if probing already failed.
    duration_s = max(25.0, min(35.0, n_frames / 25.0))
    return np.zeros(int(duration_s * audio_sr), dtype=np.float32)
