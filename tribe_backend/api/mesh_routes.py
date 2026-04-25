"""FastAPI mesh endpoints — Phase 6b.

New routes (all registered on a separate router, included by app.py):

    GET  /v1/mesh/static?surface=pial
         One-time bootstrap. Returns a tar archive containing the four
         static mesh files (vertices/normals/faces/meta). ETag-cacheable,
         immutable. Honors If-None-Match → 304.

    POST /v1/inference/colors
         Multipart: image (required, image/jpeg or video/mp4), audio
         (optional), text (optional), method, vmin, vmax, cmap.
         Returns brain_colors.bin bytes (uint8 RGBA × 20484 = 81936 bytes).

    POST /v1/inference/full
         Same multipart input; returns JSON envelope with window_id,
         colors_url, report, vmin, vmax, method.

    GET  /v1/inference/colors/{window_id}
         Serves cached colors bytes (5-min TTL). 404 if missing/expired.

Stub inference path:
    When TRIBE_INFERENCE != "gpu" (i.e. the default "fake") or when
    app.state has no registered ``inference_fn``, a deterministic stub
    TribeOutput is generated from a seeded hash of the image bytes. This
    path is always active in tests.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
import uuid
from typing import Annotated, Optional

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    SUBCORTICAL_VOXELS,
    TribeOutput,
)
from tribe_backend.mesh.colormap import activation_to_rgba
from tribe_backend.mesh.exporter import aggregate_temporal
from tribe_backend.mesh.geometry import _compute_vertex_normals, load_fsaverage5

mesh_router = APIRouter(prefix="/v1")

# ETag for the static bootstrap (version-pinned; bump when exporter changes).
_STATIC_ETAG_TEMPLATE = '"fsaverage5-{surface}-v1"'

# Number of bytes for a single RGBA frame over all cortical vertices.
_COLORS_FRAME_BYTES = CORTICAL_VERTICES * 4  # 81 936

# Colors cache entry lifetime in seconds.
_COLORS_TTL_S = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_stub_output(image_bytes: bytes, *, seed_extra: int = 0) -> TribeOutput:
    """Deterministic stub TribeOutput derived from image_bytes hash + seed."""
    digest = int(hashlib.sha256(image_bytes).hexdigest(), 16)
    seed = (digest ^ seed_extra) % (2**31)
    rng = np.random.default_rng(seed)
    T = 31
    cortical = rng.standard_normal((T, CORTICAL_VERTICES)).astype(np.float32)
    subcortical = rng.standard_normal((T, SUBCORTICAL_VOXELS)).astype(np.float32)
    wid = hashlib.sha256(image_bytes).hexdigest()[:16]
    return TribeOutput(
        cortical=cortical,
        subcortical=subcortical,
        window_id=wid,
    )


def _run_inference(app_state, image_bytes: bytes) -> TribeOutput:
    """Run inference via registered callable or fall back to stub."""
    fn = getattr(app_state, "inference_fn", None)
    if fn is not None:
        return fn(image_bytes)
    return _make_stub_output(image_bytes)


def _build_static_tar(surface: str) -> bytes:
    """Serialize static mesh for the given surface into an in-memory tar."""
    coords, faces = load_fsaverage5(surface)
    normals = _compute_vertex_normals(coords, faces)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        def _add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        _add("brain_vertices.bin", coords.astype(np.float32, copy=False).tobytes())
        _add("brain_normals.bin",  normals.astype(np.float32, copy=False).tobytes())
        _add("brain_faces.bin",    faces.astype(np.int32, copy=False).tobytes())

        meta = {
            "format": "tribe_brain_mesh_v1",
            "surface_type": surface,
            "vertex_count": int(coords.shape[0]),
            "face_count": int(faces.shape[0]),
            "bytes_per_vertex": 12,
            "bytes_per_normal": 12,
            "bytes_per_face": 12,
            "bytes_per_color": 4,
            "files": {
                "vertices": "brain_vertices.bin",
                "normals":  "brain_normals.bin",
                "faces":    "brain_faces.bin",
            },
        }
        meta_bytes = json.dumps(meta, indent=2).encode()
        _add("brain_meta.json", meta_bytes)

    return buf.getvalue()


def _colors_bytes(
    output: TribeOutput,
    method: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> tuple[bytes, int]:
    """Return (raw_bytes, frame_count)."""
    # Map method alias from plan contract → exporter literals
    _METHOD_MAP = {
        "window_mean": "mean",
        "peak": "peak",
        "peak_window": "peak",  # closest available in exporter
    }
    agg_method = _METHOD_MAP.get(method, "mean")
    per_vertex = aggregate_temporal(output.cortical, method=agg_method)
    rgba = activation_to_rgba(per_vertex, vmin=vmin, vmax=vmax, cmap=cmap)
    raw = rgba.astype(np.uint8, copy=False).tobytes()
    return raw, 1  # single-frame for now


def _get_colors_cache(app_state) -> dict:
    if not hasattr(app_state, "colors_cache"):
        app_state.colors_cache = {}
    return app_state.colors_cache


def _cache_colors(app_state, window_id: str, data: bytes) -> None:
    cache = _get_colors_cache(app_state)
    cache[window_id] = {"data": data, "expires": time.monotonic() + _COLORS_TTL_S}


def _fetch_colors(app_state, window_id: str) -> Optional[bytes]:
    cache = _get_colors_cache(app_state)
    entry = cache.get(window_id)
    if entry is None:
        return None
    if time.monotonic() > entry["expires"]:
        del cache[window_id]
        return None
    return entry["data"]


def _get_or_build_static(app_state, surface: str) -> bytes:
    """Return cached tar bytes for the surface, building on first call."""
    key = f"_mesh_static_{surface}"
    cached = getattr(app_state, key, None)
    if cached is None:
        data = _build_static_tar(surface)
        setattr(app_state, key, data)
    else:
        data = cached
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@mesh_router.get("/mesh/static")
def get_mesh_static(
    request: Request,
    surface: str = "pial",
) -> Response:
    """Bootstrap endpoint: returns tar archive with static mesh files."""
    etag = _STATIC_ETAG_TEMPLATE.format(surface=surface)

    # Honor If-None-Match.
    if_none_match = request.headers.get("if-none-match", "")
    if if_none_match and if_none_match.strip() == etag:
        return Response(status_code=304, headers={"ETag": etag})

    tar_bytes = _get_or_build_static(request.app.state, surface)

    return Response(
        content=tar_bytes,
        media_type="application/octet-stream",
        headers={
            "ETag": etag,
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Disposition": f'attachment; filename="brain_static_{surface}.tar"',
        },
    )


@mesh_router.post("/inference/colors")
def post_inference_colors(
    request: Request,
    image: Annotated[UploadFile, File(description="image/jpeg or video/mp4 frame")],
    audio: Annotated[Optional[UploadFile], File(description="optional audio/wav")] = None,
    text: Annotated[Optional[str], Form()] = None,
    method: Annotated[str, Form()] = "window_mean",
    vmin: Annotated[float, Form()] = -3.0,
    vmax: Annotated[float, Form()] = 3.0,
    cmap: Annotated[str, Form()] = "RdBu_r",
) -> Response:
    """Accept multipart input; return brain_colors.bin binary."""
    image_bytes = image.file.read()

    output = _run_inference(request.app.state, image_bytes)
    window_id = output.window_id or str(uuid.uuid4())

    raw, frame_count = _colors_bytes(output, method, vmin, vmax, cmap)

    return Response(
        content=raw,
        media_type="application/octet-stream",
        headers={
            "X-Window-Id": window_id,
            "X-Tribe-Method": method,
            "X-Vmin": str(vmin),
            "X-Vmax": str(vmax),
            "X-Frame-Count": str(frame_count),
        },
    )


@mesh_router.post("/inference/full")
def post_inference_full(
    request: Request,
    image: Annotated[UploadFile, File(description="image/jpeg or video/mp4 frame")],
    audio: Annotated[Optional[UploadFile], File(description="optional audio/wav")] = None,
    text: Annotated[Optional[str], Form()] = None,
    method: Annotated[str, Form()] = "window_mean",
    vmin: Annotated[float, Form()] = -3.0,
    vmax: Annotated[float, Form()] = 3.0,
    cmap: Annotated[str, Form()] = "RdBu_r",
):
    """Same input as /inference/colors; returns JSON envelope with report."""
    from fastapi.responses import JSONResponse
    from tribe_backend.parcellation.unit import GlasserParcellationUnit

    image_bytes = image.file.read()

    output = _run_inference(request.app.state, image_bytes)
    window_id = output.window_id or str(uuid.uuid4())

    raw, frame_count = _colors_bytes(output, method, vmin, vmax, cmap)

    # Cache colors for the follow-up GET.
    _cache_colors(request.app.state, window_id, raw)

    # Generate report.
    gpu = GlasserParcellationUnit()
    report = gpu.generate_report(output, method=method if method in ("window_mean", "peak", "peak_window") else "window_mean")

    return JSONResponse(
        status_code=200,
        content={
            "window_id": window_id,
            "colors_url": f"/v1/inference/colors/{window_id}",
            "report": json.loads(report.to_json()),
            "vmin": vmin,
            "vmax": vmax,
            "method": method,
        },
    )


@mesh_router.get("/inference/colors/{window_id}")
def get_cached_colors(request: Request, window_id: str) -> Response:
    """Serve cached colors bytes (5-min TTL). 404 if missing or expired."""
    data = _fetch_colors(request.app.state, window_id)
    if data is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": {"code": "NOT_FOUND", "message": f"window_id {window_id!r} not found or expired"}},
        )
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Window-Id": window_id,
            "Content-Length": str(len(data)),
        },
    )
