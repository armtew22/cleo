"""GET /v1/runs/{job_id} — status polling and terminal report shape."""
from __future__ import annotations

import time


def _poll_until_done(client, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/runs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate within {timeout}s; last={body}")


def test_get_unknown_job_returns_404(test_client):
    r = test_client.get("/v1/runs/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "JOB_NOT_FOUND"
    assert "message" in body["error"]


def test_get_job_lifecycle_queued_then_done(test_client, real_30s_clip_bytes):
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "lifecycle"},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    # Right after submit, the job exists.
    r = test_client.get(f"/v1/runs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("queued", "running", "done")
    assert "submitted_at" in body
    # started_at and finished_at may be null at this stage; they must be present keys.
    assert "started_at" in body
    assert "finished_at" in body

    final = _poll_until_done(test_client, job_id)
    assert final["status"] == "done", final
    assert final["started_at"] is not None
    assert final["finished_at"] is not None

    # When done, the report payload is included with the same shape Phase A returned.
    assert "report" in final
    report = final["report"]
    assert "top_regions" in report and isinstance(report["top_regions"], list)
    assert "text" in report
    assert "method" in report
    assert "z_threshold" in report


def test_done_response_has_no_error_field(test_client, real_30s_clip_bytes):
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "no error"},
    )
    job_id = submit.json()["job_id"]
    final = _poll_until_done(test_client, job_id)
    assert final["status"] == "done"
    assert final.get("error") is None
