"""GET /v1/runs/{job_id}/mesh — manifest enumerating artifact URLs.

The manifest is a per-job JSON document so the frontend can discover
mesh artifact URLs from the job_id without hand-constructing them.
"""
from __future__ import annotations

import time


def _poll_until_done(client, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        r = client.get(f"/v1/runs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate within {timeout}s; last={body}")


def test_manifest_returns_four_artifact_urls(test_client, real_30s_clip_bytes):
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "manifest"},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    final = _poll_until_done(test_client, job_id)
    assert final["status"] == "done", final

    r = test_client.get(f"/v1/runs/{job_id}/mesh")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    arts = body["artifacts"]
    assert set(arts.keys()) == {"meta", "colors", "vertices", "faces"}
    # URLs must be absolute paths under the same job_id namespace.
    expected = {
        "meta":     f"/v1/runs/{job_id}/mesh/meta",
        "colors":   f"/v1/runs/{job_id}/mesh/colors",
        "vertices": f"/v1/runs/{job_id}/mesh/vertices",
        "faces":    f"/v1/runs/{job_id}/mesh/faces",
    }
    assert arts == expected


def test_manifest_uses_no_store_cache(test_client, real_30s_clip_bytes):
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "manifest cache"},
    )
    job_id = submit.json()["job_id"]
    _poll_until_done(test_client, job_id)
    r = test_client.get(f"/v1/runs/{job_id}/mesh")
    assert r.status_code == 200
    assert r.headers.get("cache-control", "").lower() == "no-store"
