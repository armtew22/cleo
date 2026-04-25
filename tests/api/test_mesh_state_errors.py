"""Mesh endpoint behavior across all non-done job states.

Matrix:
    unknown   →  404 JOB_NOT_FOUND
    queued    →  409 JOB_NOT_READY
    running   →  409 JOB_NOT_READY  (covered by the gated worker)
    failed    →  410 JOB_FAILED
    cancelled →  410 JOB_FAILED
    done but artifact missing on disk → 500 INFERENCE_ARTIFACTS_MISSING

The Phase C handlers all funnel through the same _require_done_job
helper, so we test each endpoint with one representative case rather
than full N×M coverage.
"""
from __future__ import annotations

import time

import pytest


# All four mesh endpoints route through the same state-checking helper —
# parametrize so a future endpoint addition gets coverage for free.
MESH_ENDPOINTS = [
    "/mesh",
    "/mesh/meta",
    "/mesh/colors",
    "/mesh/vertices",
    "/mesh/faces",
]


# -------------------------------------------------------------------- 404

@pytest.mark.parametrize("suffix", MESH_ENDPOINTS)
def test_unknown_job_returns_404(test_client, suffix):
    r = test_client.get(f"/v1/runs/abcdef0123/{suffix.lstrip('/')}".replace("//", "/"))
    # The path is /v1/runs/<id>/mesh... — build it cleanly:
    r = test_client.get(f"/v1/runs/abcdef0123{suffix}")
    assert r.status_code == 404, r.text
    body = r.json()
    assert body["error"]["code"] == "JOB_NOT_FOUND"


# -------------------------------------------------------------------- 409

@pytest.mark.parametrize("suffix", MESH_ENDPOINTS)
def test_queued_job_returns_409(test_client_paused_worker, tiny_synthetic_mp4, suffix):
    client, resume = test_client_paused_worker
    try:
        submit = client.post(
            "/v1/runs",
            files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
            data={"text": "queued mesh"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]
        # Worker is paused, so the job stays queued.
        status_body = client.get(f"/v1/runs/{job_id}").json()
        assert status_body["status"] == "queued", status_body

        r = client.get(f"/v1/runs/{job_id}{suffix}")
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["error"]["code"] == "JOB_NOT_READY"
        assert "queued" in body["error"]["message"] or job_id in body["error"]["message"]
    finally:
        resume()


# -------------------------------------------------------------------- 410 failed

@pytest.fixture
def client_with_failing_inference(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TRIBE_INFERENCE", "fake")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")

    from tribe_backend.api import worker as worker_mod
    from tribe_backend.control.factory import InferenceFailure

    real_run_worker = worker_mod.run_worker

    async def failing_run_worker(store, _process_window, stop, **kw):
        def boom(**_payload):
            raise InferenceFailure("synthetic failure")
        return await real_run_worker(store, boom, stop, **kw)

    monkeypatch.setattr(worker_mod, "run_worker", failing_run_worker)
    from tribe_backend.api.app import get_app
    app = get_app()
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("suffix", MESH_ENDPOINTS)
def test_failed_job_returns_410(client_with_failing_inference, tiny_synthetic_mp4, suffix):
    client = client_with_failing_inference
    submit = client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"text": "failed mesh"},
    )
    job_id = submit.json()["job_id"]
    deadline = time.time() + 10.0
    body = None
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{job_id}").json()
        if body["status"] == "failed":
            break
        time.sleep(0.05)
    assert body and body["status"] == "failed", body

    r = client.get(f"/v1/runs/{job_id}{suffix}")
    assert r.status_code == 410, r.text
    err = r.json()["error"]
    assert err["code"] == "JOB_FAILED"


# -------------------------------------------------------------------- 410 cancelled

def test_cancelled_job_returns_410(test_client_paused_worker, tiny_synthetic_mp4):
    client, resume = test_client_paused_worker
    try:
        submit = client.post(
            "/v1/runs",
            files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
            data={"text": "cancel mesh"},
        )
        job_id = submit.json()["job_id"]
        # Cancel while queued.
        cancel = client.delete(f"/v1/runs/{job_id}")
        assert cancel.status_code == 204
        r = client.get(f"/v1/runs/{job_id}/mesh")
        assert r.status_code == 410, r.text
        assert r.json()["error"]["code"] == "JOB_FAILED"
    finally:
        resume()


# -------------------------------------------------------------------- 500 missing on disk

def test_done_job_with_missing_artifact_returns_500(
    test_client, real_30s_clip_bytes, monkeypatch
):
    """If the report is present but the .bin file is gone (rm'd, never written,
    etc.), the endpoint must surface 500 INFERENCE_ARTIFACTS_MISSING rather
    than serving stale data or crashing."""
    import os
    import shutil

    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "missing artifact"},
    )
    job_id = submit.json()["job_id"]
    deadline = time.time() + 15.0
    body = None
    while time.time() < deadline:
        body = test_client.get(f"/v1/runs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert body and body["status"] == "done", body

    window_id = body["report"]["window_id"]
    out_dir = os.environ["OUT_DIR"]
    # Wipe one specific file.
    target = os.path.join(out_dir, window_id, "brain_colors.bin")
    assert os.path.isfile(target), target
    os.remove(target)

    r = test_client.get(f"/v1/runs/{job_id}/mesh/colors")
    assert r.status_code == 500, r.text
    assert r.json()["error"]["code"] == "INFERENCE_ARTIFACTS_MISSING"
