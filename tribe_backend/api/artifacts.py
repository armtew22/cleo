"""Pure helpers for serving mesh artifact files written by ControlUnit.

Phase C scope. The routing layer (routes.py) is a thin shim over these helpers
so all path-validation logic lives in one auditable place.

ControlUnit writes one directory per processed window:

    <out_dir>/<window_id>/brain_meta.json
    <out_dir>/<window_id>/brain_colors.bin
    <out_dir>/<window_id>/brain_vertices.bin
    <out_dir>/<window_id>/brain_faces.bin

Note that `window_id` is NOT the same as the API's `job_id` — the JobStore
generates a uuid hex per submit, while ControlUnit generates an ISO-timestamp
window_id inside `_make_window_id`. The disk path key is `window_id`, which
is recorded in the Report and surfaced in the job's `report["window_id"]`
field. The routing layer is responsible for resolving job_id → window_id via
the JobStore before calling `artifact_path()` here.

This module never reads `app.state` or the JobStore directly — it just
validates inputs and constructs paths.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.responses import FileResponse

__all__ = [
    "ARTIFACT_NAMES",
    "MEDIA_TYPES",
    "BINARY_CACHE_HEADER",
    "JSON_CACHE_HEADER",
    "UnsafeArtifactRequest",
    "is_allowed_artifact_name",
    "is_safe_job_id",
    "artifact_path",
    "binary_file_response",
]


# Allowlist of files we are willing to serve. Anything else 404s without
# touching the filesystem.
ARTIFACT_NAMES: frozenset[str] = frozenset({
    "brain_meta.json",
    "brain_colors.bin",
    "brain_vertices.bin",
    "brain_faces.bin",
})

MEDIA_TYPES: dict[str, str] = {
    "brain_meta.json": "application/json",
    "brain_colors.bin": "application/octet-stream",
    "brain_vertices.bin": "application/octet-stream",
    "brain_faces.bin": "application/octet-stream",
}

# 1 year, immutable. Vertices/faces are pure functions of fsaverage5 and
# colors are tied to a specific window_id and never change once written.
BINARY_CACHE_HEADER = "public, max-age=31536000, immutable"
# Manifest + meta reference in-flight job state — never cache.
JSON_CACHE_HEADER = "no-store"


class UnsafeArtifactRequest(ValueError):
    """Raised when a job_id or artifact name fails validation.

    The routing layer translates this into a 404 — we deliberately do NOT
    distinguish "bad name" from "bad id" in the public response so that the
    surface gives no information about the on-disk layout.
    """


# job_id may be:
#   - a uuid4 hex (32 lowercase hex chars), per JobStore.submit
#   - a window_id like "2026-04-25T12:34:56Z-abc123" (ISO ts + 6 hex)
# Both fit in: alnum, ".", "_", ":", "-", "T". No "/" or ".." possible.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_:T.\-]{1,128}$")


def is_allowed_artifact_name(name: str) -> bool:
    return name in ARTIFACT_NAMES


def is_safe_job_id(job_id: str) -> bool:
    if not isinstance(job_id, str):
        return False
    if not job_id or job_id in (".", ".."):
        return False
    if "/" in job_id or "\\" in job_id or "\x00" in job_id:
        return False
    return _JOB_ID_RE.match(job_id) is not None


def artifact_path(out_dir: str | Path, job_id: str, name: str) -> Path:
    """Build the on-disk path for an artifact, validating both inputs.

    Parameters
    ----------
    out_dir
        Base directory where ControlUnit's ArtifactSink writes per-window
        subdirectories. Typically `Settings.out_dir`.
    job_id
        The directory name (== window_id on disk). Must satisfy
        `is_safe_job_id`.
    name
        Artifact filename. Must be in `ARTIFACT_NAMES`.

    Raises
    ------
    UnsafeArtifactRequest
        If either input fails validation, OR if the resolved path escapes
        `out_dir` (defense-in-depth against symlinks).
    """
    if not is_safe_job_id(job_id):
        raise UnsafeArtifactRequest(f"unsafe job_id {job_id!r}")
    if not is_allowed_artifact_name(name):
        raise UnsafeArtifactRequest(f"unknown artifact name {name!r}")

    base = Path(out_dir)
    candidate = base / job_id / name

    # Defense in depth: if symlinks (or anything else) cause the resolved path
    # to escape `base`, refuse. We use `os.path.commonpath` semantics via
    # Path.resolve(strict=False) so missing files are still validated.
    base_resolved = base.resolve(strict=False)
    cand_resolved = candidate.resolve(strict=False)
    try:
        cand_resolved.relative_to(base_resolved)
    except ValueError as e:
        raise UnsafeArtifactRequest(
            f"resolved artifact path escapes out_dir: {cand_resolved}"
        ) from e

    return candidate


def binary_file_response(path: Path, name: str) -> FileResponse:
    """Build a FileResponse for a binary artifact with the right headers.

    Caller is responsible for verifying `path.exists()` first — this helper
    does not check, it just wraps the response.
    """
    media_type = MEDIA_TYPES.get(name, "application/octet-stream")
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=name,
        headers={"Cache-Control": BINARY_CACHE_HEADER},
    )
