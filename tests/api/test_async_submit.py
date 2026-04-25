"""POST /v1/runs without ?wait=true — async submission.

Default mode is async (Phase A returned 400, Phase B returns 202 + job_id).
"""
from __future__ import annotations


def test_async_submit_returns_202_and_job_id(test_client, tiny_synthetic_mp4):
    r = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"text": "async test"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert "job_id" in body and isinstance(body["job_id"], str) and body["job_id"]
    assert body["status"] == "queued"


def test_async_submit_default_is_async(test_client, tiny_synthetic_mp4):
    """No `wait` field at all → async (202)."""
    r = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"text": "default async"},
    )
    assert r.status_code == 202


def test_async_submit_wait_false_explicit_returns_202(test_client, tiny_synthetic_mp4):
    """`wait=false` is now valid (Phase B): returns 202."""
    r = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"text": "explicit false", "wait": "false"},
    )
    assert r.status_code == 202


def test_async_submit_garbage_mp4_returns_400(test_client):
    """Decode happens synchronously before queueing, so a bad mp4 is still 400."""
    r = test_client.post(
        "/v1/runs",
        files={"media": ("fake.mp4", b"not an mp4", "video/mp4")},
        data={"text": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "decode_error"


def test_async_submit_missing_text_returns_422(test_client, tiny_synthetic_mp4):
    r = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
    )
    assert r.status_code == 422
