"""Tests for the z_threshold form field on POST /v1/runs.

Frontend may pass `z_threshold=0.1` to surface weak activations the default
of 1.5 would suppress. Verified end-to-end through the sync path; the async
path receives the same payload dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tribe_backend.api.app import get_app
from tests.conftest import make_synthetic


pytestmark = pytest.mark.unit


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIBE_INFERENCE", "fake")
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    return get_app()


@pytest.fixture
def fake_mp4(tmp_path) -> bytes:
    """Synthesize a tiny mp4 via ffmpeg from raw frames + silent audio."""
    import subprocess
    import wave

    F, H, W = 60, 64, 64  # 30s @ 2fps
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(F, H, W, 3), dtype=np.uint8)

    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(np.zeros(16000 * 30, dtype=np.int16).tobytes())

    mp4 = tmp_path / "v.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "2",
        "-i", "pipe:0",
        "-i", str(wav),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(mp4),
    ]
    subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)
    return mp4.read_bytes()


def _post(client, mp4_bytes, **form):
    return client.post(
        "/v1/runs",
        files={"media": ("v.mp4", mp4_bytes, "video/mp4")},
        data={"text": "z_threshold smoke test", "wait": "true", **form},
    )


def test_default_threshold_used_when_field_omitted(app, fake_mp4):
    with TestClient(app) as client:
        r = _post(client, fake_mp4)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["z_threshold"] == 1.5  # parcellation default propagated


def test_explicit_low_threshold_overrides_default(app, fake_mp4):
    with TestClient(app) as client:
        r = _post(client, fake_mp4, z_threshold="0.1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["z_threshold"] == pytest.approx(0.1)


def test_explicit_high_threshold_overrides_default(app, fake_mp4):
    with TestClient(app) as client:
        r = _post(client, fake_mp4, z_threshold="3.0")
        assert r.status_code == 200, r.text
        assert r.json()["z_threshold"] == pytest.approx(3.0)


def test_negative_threshold_rejected(app, fake_mp4):
    with TestClient(app) as client:
        r = _post(client, fake_mp4, z_threshold="-0.5")
        assert r.status_code == 422


def test_lower_threshold_yields_more_or_equal_top_regions(app, fake_mp4):
    """Monotonicity: a lower threshold cannot return fewer regions."""
    with TestClient(app) as client:
        high = _post(client, fake_mp4, z_threshold="2.0").json()["top_regions"]
        low  = _post(client, fake_mp4, z_threshold="0.05").json()["top_regions"]
    assert len(low) >= len(high)
