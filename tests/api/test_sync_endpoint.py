"""POST /v1/runs (sync, multipart-only)."""
from __future__ import annotations


def test_submit_returns_report(test_client, real_30s_clip_bytes):
    r = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "The quick brown fox.", "wait": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Report shape mirrors Report.to_json()
    assert "window_id" in body and body["window_id"] is not None
    assert "top_regions" in body and isinstance(body["top_regions"], list)
    assert "text" in body and isinstance(body["text"], str)
    assert "method" in body
    assert "z_threshold" in body


def test_submit_missing_media_returns_422(test_client):
    r = test_client.post("/v1/runs", data={"text": "hello", "wait": "true"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert "message" in body["error"]


def test_submit_missing_text_returns_422(test_client, tiny_synthetic_mp4):
    r = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"wait": "true"},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"


def test_submit_wait_false_returns_202(test_client, tiny_synthetic_mp4):
    """Phase B: wait=false (default) is async — returns 202 + job_id."""
    r = test_client.post(
        "/v1/runs",
        files={"media": ("tiny.mp4", tiny_synthetic_mp4, "video/mp4")},
        data={"text": "hi", "wait": "false"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert "job_id" in body


def test_submit_non_mp4_content_type_rejected(test_client):
    """Non-video content-types are rejected with 400 decode_error.

    Documented behavior: the endpoint inspects content-type before attempting
    to decode. This avoids spending CPU on garbage and gives the frontend a
    crisp signal.
    """
    r = test_client.post(
        "/v1/runs",
        files={"media": ("note.txt", b"hello world", "text/plain")},
        data={"text": "hi", "wait": "true"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "decode_error"


def test_submit_garbage_mp4_returns_400(test_client):
    r = test_client.post(
        "/v1/runs",
        files={"media": ("fake.mp4", b"not an mp4", "video/mp4")},
        data={"text": "hi", "wait": "true"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "decode_error"
