"""DELETE /v1/runs/{job_id} — cancel queued jobs.

Phase B v1: queued jobs can be cancelled (204); running/done/failed jobs
cannot (409). Unknown jobs → 404.
"""
from __future__ import annotations

import time

import pytest


def test_delete_unknown_returns_404(test_client):
    r = test_client.delete("/v1/runs/nope")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "JOB_NOT_FOUND"


def test_delete_queued_returns_204_and_marks_cancelled(test_client_paused_worker, tiny_synthetic_mp4):
    """Submit while the worker is paused so the job stays in 'queued',
    then cancel it."""
    client, resume = test_client_paused_worker
    try:
        submit = client.post(
            "/v1/runs",
            files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
            data={"text": "to be cancelled"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        r = client.delete(f"/v1/runs/{job_id}")
        assert r.status_code == 204
        assert r.text == ""

        # Subsequent GET shows status="cancelled".
        g = client.get(f"/v1/runs/{job_id}")
        assert g.status_code == 200
        assert g.json()["status"] == "cancelled"
    finally:
        resume()


def test_delete_done_returns_409(test_client, real_30s_clip_bytes):
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "x"},
    )
    job_id = submit.json()["job_id"]
    # Wait for done.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        body = test_client.get(f"/v1/runs/{job_id}").json()
        if body["status"] == "done":
            break
        time.sleep(0.05)
    assert body["status"] == "done"

    r = test_client.delete(f"/v1/runs/{job_id}")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_NOT_CANCELLABLE"
