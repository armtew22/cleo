"""GET /v1/runs/{job_id} carries a top-level "mesh" URL when status==done.

Lets the frontend follow a single link to the per-job artifact manifest
without hand-constructing URLs.
"""
from __future__ import annotations

import time


def _poll_until_terminal(client, job_id, *, timeout=15.0):
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{job_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate; last={body}")


def test_status_done_includes_mesh_url(test_client, real_30s_clip_bytes):
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "mesh url"},
    )
    job_id = submit.json()["job_id"]
    body = _poll_until_terminal(test_client, job_id)
    assert body["status"] == "done"
    assert body["mesh"] == f"/v1/runs/{job_id}/mesh"

    # Sanity: the URL actually resolves.
    r = test_client.get(body["mesh"])
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id


def test_status_queued_does_not_include_mesh_url(
    test_client_paused_worker, tiny_synthetic_mp4
):
    client, resume = test_client_paused_worker
    try:
        submit = client.post(
            "/v1/runs",
            files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
            data={"text": "queued no mesh"},
        )
        job_id = submit.json()["job_id"]
        body = client.get(f"/v1/runs/{job_id}").json()
        assert body["status"] == "queued"
        # mesh is None / absent for non-done states.
        assert body.get("mesh") is None
    finally:
        resume()


def test_status_response_uses_no_store(test_client):
    """Even an unknown-id 404 isn't relevant here — we just want to confirm
    that a real status payload carries no-store. Use a quick submit→done."""
    # Build a minimal request via the synthetic fixture so this test is fast
    # and doesn't depend on the 30s clip being present.
    import shutil, subprocess, tempfile, os
    if shutil.which("ffmpeg") is None:
        import pytest
        pytest.skip("ffmpeg required")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "tiny.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "color=c=red:s=16x16:d=1:r=25",
             "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
             "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", out],
            check=True,
        )
        with open(out, "rb") as f:
            mp4 = f.read()
    submit = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", mp4, "video/mp4")},
        data={"text": "no-store check"},
    )
    job_id = submit.json()["job_id"]
    r = test_client.get(f"/v1/runs/{job_id}")
    assert r.status_code == 200
    assert r.headers.get("cache-control", "").lower() == "no-store"
