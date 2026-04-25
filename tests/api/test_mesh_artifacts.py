"""GET /v1/runs/{job_id}/mesh/{colors,vertices,faces,meta} — artifact serving.

For fsaverage5 the cortical mesh is 20484 vertices and 40960 faces. Each
binary file's exact byte size is determined by the dtype layout the mesh
exporter writes:

    brain_colors.bin   = 20484 * 4 bytes  (rgba uint8)
    brain_vertices.bin = 20484 * 12 bytes (3 float32 per vertex)
    brain_faces.bin    = 40960 * 12 bytes (3 int32 per face)
"""
from __future__ import annotations

import json
import time
from pathlib import Path


VERTEX_COUNT = 20484
FACE_COUNT = 40960
COLORS_BYTES = VERTEX_COUNT * 4
VERTICES_BYTES = VERTEX_COUNT * 12
FACES_BYTES = FACE_COUNT * 12


def _run_until_done(client, real_30s_clip_bytes, *, text="artifacts"):
    submit = client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": text},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    deadline = time.time() + 15.0
    body = None
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{job_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert body and body["status"] == "done", body
    return job_id, body


def test_colors_endpoint_byte_exact(test_client, real_30s_clip_bytes):
    job_id, _ = _run_until_done(test_client, real_30s_clip_bytes)
    r = test_client.get(f"/v1/runs/{job_id}/mesh/colors")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert int(r.headers["content-length"]) == COLORS_BYTES
    assert len(r.content) == COLORS_BYTES


def test_vertices_endpoint_byte_exact(test_client, real_30s_clip_bytes):
    job_id, _ = _run_until_done(test_client, real_30s_clip_bytes)
    r = test_client.get(f"/v1/runs/{job_id}/mesh/vertices")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert int(r.headers["content-length"]) == VERTICES_BYTES
    assert len(r.content) == VERTICES_BYTES


def test_faces_endpoint_byte_exact(test_client, real_30s_clip_bytes):
    job_id, _ = _run_until_done(test_client, real_30s_clip_bytes)
    r = test_client.get(f"/v1/runs/{job_id}/mesh/faces")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert int(r.headers["content-length"]) == FACES_BYTES
    assert len(r.content) == FACES_BYTES


def test_meta_endpoint_returns_json(test_client, real_30s_clip_bytes):
    job_id, _ = _run_until_done(test_client, real_30s_clip_bytes)
    r = test_client.get(f"/v1/runs/{job_id}/mesh/meta")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["vertex_count"] == VERTEX_COUNT
    assert body["face_count"] == FACE_COUNT
    # files block should reference the four sibling artifacts.
    assert body["files"]["colors"] == "brain_colors.bin"
    assert body["files"]["vertices"] == "brain_vertices.bin"
    assert body["files"]["faces"] == "brain_faces.bin"


def test_artifacts_match_disk(test_client, real_30s_clip_bytes, tmp_path):
    """Bytes served over HTTP must equal bytes on disk."""
    # The test_client fixture sets OUT_DIR to <tmp>/out — recover that path
    # by reading it back from the manifest's window_id (via report).
    job_id, status_body = _run_until_done(test_client, real_30s_clip_bytes)
    window_id = status_body["report"]["window_id"]

    # Find which OUT_DIR this client is using by walking up to find it.
    # Cleaner: read OUT_DIR from env (test_client fixture set it).
    import os
    out_dir = Path(os.environ["OUT_DIR"])
    disk_colors = (out_dir / window_id / "brain_colors.bin").read_bytes()
    http_colors = test_client.get(f"/v1/runs/{job_id}/mesh/colors").content
    assert http_colors == disk_colors

    disk_meta = json.loads((out_dir / window_id / "brain_meta.json").read_text())
    http_meta = test_client.get(f"/v1/runs/{job_id}/mesh/meta").json()
    assert http_meta == disk_meta
