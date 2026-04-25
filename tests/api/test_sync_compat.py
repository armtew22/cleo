"""Phase A regression: ?wait=true still returns the report inline."""
from __future__ import annotations


def test_wait_true_returns_report_inline(test_client, real_30s_clip_bytes):
    r = test_client.post(
        "/v1/runs",
        files={"media": ("clip_30s.mp4", real_30s_clip_bytes, "video/mp4")},
        data={"text": "sync regression", "wait": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Phase A response shape — direct Report payload at the top level.
    assert "top_regions" in body
    assert "text" in body
    assert "method" in body
    assert "z_threshold" in body
    assert "window_id" in body
