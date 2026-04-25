"""POST /v1/runs returns 503 + Retry-After when the queue is full."""
from __future__ import annotations


def test_queue_full_returns_503(test_client_paused_worker, tiny_synthetic_mp4):
    """With queue depth=2 and the worker paused, the third submit must 503."""
    client, resume = test_client_paused_worker
    try:
        # Fill the queue.
        for i in range(2):
            r = client.post(
                "/v1/runs",
                files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
                data={"text": f"job-{i}"},
            )
            assert r.status_code == 202, r.text

        # Third should be rejected.
        r = client.post(
            "/v1/runs",
            files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
            data={"text": "overflow"},
        )
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "QUEUE_FULL"
        assert "Retry-After" in r.headers
        assert int(r.headers["Retry-After"]) > 0
    finally:
        resume()
