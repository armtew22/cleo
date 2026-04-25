"""When inference raises InferenceFailure, the job is marked failed.

The HTTP transport never returns 500 for in-job failures — `GET /v1/runs/{id}`
still returns 200, with status="failed" and a sanitized error envelope.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def test_client_failing_inference(tmp_path, monkeypatch):
    """A FastAPI test client whose inference always raises InferenceFailure."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TRIBE_INFERENCE", "fake")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")

    from tribe_backend.api import worker as worker_mod
    from tribe_backend.control.factory import InferenceFailure

    real_run_worker = worker_mod.run_worker

    async def failing_run_worker(store, _process_window, stop, **kw):
        # Replace the process_window callable with one that always raises.
        def boom(**_payload):
            raise InferenceFailure("synthetic failure")
        return await real_run_worker(store, boom, stop, **kw)

    monkeypatch.setattr(worker_mod, "run_worker", failing_run_worker)

    from tribe_backend.api.app import get_app
    app = get_app()
    with TestClient(app) as client:
        yield client


def test_inference_failure_marks_job_failed(test_client_failing_inference, tiny_synthetic_mp4):
    client = test_client_failing_inference
    submit = client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"text": "boom"},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    deadline = time.time() + 10.0
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{job_id}").json()
        if body["status"] in ("failed", "done"):
            break
        time.sleep(0.05)

    assert body["status"] == "failed", body
    assert "error" in body
    assert body["error"]["code"] == "INFERENCE_FAILURE"
    assert "synthetic failure" in body["error"]["message"]
    # No traceback / stack leaked:
    assert "Traceback" not in body["error"]["message"]
