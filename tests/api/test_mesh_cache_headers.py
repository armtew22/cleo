"""Cache-Control discipline on mesh endpoints.

Binary artifacts (vertices/faces are pure functions of fsaverage5; colors
are tied to a window_id and never change once written) → immutable cache
for one year. JSON endpoints (manifest + meta) reference job state and
must not be cached.
"""
from __future__ import annotations

import time

IMMUTABLE = "public, max-age=31536000, immutable"
NO_STORE = "no-store"


def _run_to_done(client, mp4_bytes):
    submit = client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", mp4_bytes, "video/mp4")},
        data={"text": "cache headers"},
    )
    job_id = submit.json()["job_id"]
    deadline = time.time() + 15.0
    body = None
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert body and body["status"] == "done"
    return job_id


def test_binary_endpoints_have_immutable_cache(test_client, real_30s_clip_bytes):
    job_id = _run_to_done(test_client, real_30s_clip_bytes)
    for suffix in ("colors", "vertices", "faces"):
        r = test_client.get(f"/v1/runs/{job_id}/mesh/{suffix}")
        assert r.status_code == 200, (suffix, r.text)
        cc = r.headers.get("cache-control", "").lower()
        assert cc == IMMUTABLE, (suffix, cc)


def test_json_endpoints_have_no_store(test_client, real_30s_clip_bytes):
    job_id = _run_to_done(test_client, real_30s_clip_bytes)
    for suffix in ("", "/meta"):
        r = test_client.get(f"/v1/runs/{job_id}/mesh{suffix}")
        assert r.status_code == 200, (suffix, r.text)
        cc = r.headers.get("cache-control", "").lower()
        assert cc == NO_STORE, (suffix, cc)
