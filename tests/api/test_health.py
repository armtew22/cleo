"""GET /v1/health."""
from __future__ import annotations


def test_health_returns_inference_backend_and_out_dir(test_client):
    r = test_client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["inference"] == "fake"
    assert isinstance(body["out_dir"], str) and body["out_dir"]
