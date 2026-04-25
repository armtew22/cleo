"""End-to-end pipeline: clip in → caregiver brief out, via the remote TRIBE backend.

  caption clip (Claude vision)
        │
        ├─────────────────────►  POST /v1/runs (remote GPU box)
        │                        ── poll ── ~5 min ──►  Glasser top_regions
        │                                                       │
        │                                                       ▼
        │                              bucket regions → auditory / visual / limbic
        │                                                       │
        ▼                                                       ▼
  scene caption  ─────────────────────► caregiver brief (Claude, feedback.py)
                                        + 1-10 score, dominant channel

Subcortical activations (amygdala/hippocampus/thalamus) are zero-filled in the
upstream model for v1, so we don't pipe them into the caregiver brief yet.
The hook is in ``feedback.generate_feedback(..., subcortical_z=...)`` — flip
it on the moment the upstream model exposes a non-zero subcortical head.

Env knobs (forwarded to ``cleo.tribe_client.TribeClient``):
  TRIBE_BACKEND_URL    base URL of the remote service
  TRIBE_POLL_INTERVAL  seconds between polls
  TRIBE_TIMEOUT        per-job ceiling
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from cleo.captioning import caption_clip
from cleo.glasser_groups import aggregate_top_regions
from cleo.tribe_client import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    TribeClient,
)

StatusCallback = Callable[[str, dict[str, Any]], None]


def rank_groups(roi_stats: dict[str, dict[str, float]]) -> list[str]:
    """Rank Destrieux groups by sustained z-score, descending."""
    return sorted(roi_stats, key=lambda g: roi_stats[g]["z_sustained"], reverse=True)


def run_forecast(
    video_path: Path,
    *,
    include_feedback: bool = True,
    tribe_url: str | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_TIMEOUT,
    on_status: StatusCallback | None = None,
) -> dict[str, Any]:
    """Run the full pipeline on a single clip and return a JSON-ready forecast.

    Steps
    -----
    1. Generate an objective caption of the scene via Claude vision.
    2. POST the video + caption to the remote TRIBE backend; poll for the report.
    3. Bucket the upstream Glasser ``top_regions`` into our three Destrieux
       groups (auditory / visual / limbic-adjacent).
    4. Generate a caregiver-facing markdown brief via Claude.

    Parameters
    ----------
    video_path:
        Path to a ≈30 s mp4 clip.
    include_feedback:
        Generate caregiver markdown (requires ``ANTHROPIC_API_KEY``).
    tribe_url:
        Override the TRIBE backend URL (default reads ``TRIBE_BACKEND_URL``).
    poll_interval, timeout:
        Forwarded to ``TribeClient.wait``.
    on_status:
        Optional ``cb(stage: str, info: dict)`` for live progress updates.
        Stages: ``captioning`` → ``submitting`` → upstream statuses
        (``queued``/``running``/``done``) → ``feedback`` → ``complete``.

    Returns
    -------
    dict with keys:
        ``video``, ``caption``, ``tribe_job_id``, ``tribe_window_id``,
        ``tribe_text``, ``top_regions``, ``roi_stats``,
        ``ranking_by_sustained_z``, ``mesh_url``,
        and optionally ``feedback`` / ``feedback_error``.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    def _ping(stage: str, info: dict[str, Any] | None = None) -> None:
        if on_status is not None:
            try:
                on_status(stage, info or {})
            except Exception:  # noqa: BLE001
                pass

    # ── 1. caption ────────────────────────────────────────────────────────────
    _ping("captioning", {"video": str(video_path)})
    caption = caption_clip(video_path)

    # ── 2. remote TRIBE ───────────────────────────────────────────────────────
    _ping("submitting", {"caption_chars": len(caption)})
    client = TribeClient(tribe_url)
    job = client.submit_and_wait(
        video_path,
        caption,
        poll_interval=poll_interval,
        timeout=timeout,
        on_status=on_status,
    )

    report = job.get("report") or {}
    top_regions = report.get("top_regions") or []

    # ── 3. bucket Glasser regions → Destrieux 3 groups ────────────────────────
    roi_stats = aggregate_top_regions(top_regions)
    ranking = rank_groups(roi_stats)

    out: dict[str, Any] = {
        "video": str(video_path),
        "caption": caption,
        "tribe_job_id": job.get("job_id"),
        "tribe_window_id": report.get("window_id"),
        "tribe_z_threshold": report.get("z_threshold"),
        "tribe_text": report.get("text"),
        "top_regions": top_regions,
        "roi_stats": roi_stats,
        "ranking_by_sustained_z": ranking,
        "mesh_url": job.get("mesh"),
    }

    # ── 4. caregiver brief ────────────────────────────────────────────────────
    if include_feedback and os.environ.get("ANTHROPIC_API_KEY"):
        _ping("feedback", {"top_group": ranking[0] if ranking else None})
        from cleo.feedback import generate_feedback

        try:
            # Subcortical zero-filled upstream → don't pass subcortical_z yet.
            out["feedback"] = generate_feedback(caption, roi_stats, ranking)
        except Exception as e:  # noqa: BLE001
            out["feedback_error"] = f"{e.__class__.__name__}: {e}"

    _ping("complete", {})
    return out
