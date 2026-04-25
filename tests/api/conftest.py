"""API-test fixtures.

Provides:
    - real_30s_clip_bytes : the bench fixture if present (else skip)
    - tiny_synthetic_mp4  : a dynamically generated mp4 blob (no audio) for
                            tests that just need a parseable upload
    - test_client         : FastAPI TestClient with TRIBE_INFERENCE=fake and a
                            tmp out_dir, sharing a single ControlUnit
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

REAL_30S_CLIP = Path("/home/md2292/tribev2-bench/stimuli/clip_30s.mp4")


@pytest.fixture
def real_30s_clip_bytes() -> bytes:
    if not REAL_30S_CLIP.is_file():
        pytest.skip("real 30s clip not present")
    return REAL_30S_CLIP.read_bytes()


@pytest.fixture
def tiny_synthetic_mp4(tmp_path: Path) -> bytes:
    """Generate a ~1s, 16x16, 25fps mp4 with a silent audio track via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required to synthesize a fixture mp4")
    out = tmp_path / "tiny.mp4"
    # 1s of color bars + 1s of silent audio at 16k mono.
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=red:s=16x16:d=1:r=25",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        pytest.skip(f"ffmpeg synth failed: {res.stderr.decode('utf-8', errors='replace')[:200]}")
    return out.read_bytes()


@pytest.fixture
def test_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("TRIBE_INFERENCE", "fake")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    # Re-import the app factory each time so it reads fresh env.
    from tribe_backend.api.app import get_app
    app = get_app()
    with TestClient(app) as client:
        yield client
