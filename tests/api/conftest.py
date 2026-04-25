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


@pytest.fixture
def test_client_paused_worker(tmp_path, monkeypatch):
    """Like `test_client` but the worker is paused on a gate so jobs stay
    in `queued` for the duration of the test. The fixture yields
    `(client, resume)`; the test calls `resume()` to release the worker
    (typically inside a `try/finally`).

    Queue depth is set to 2 so the queue-full test can saturate it cheaply.
    """
    import asyncio as _asyncio
    from tribe_backend.api import worker as worker_mod

    monkeypatch.setenv("TRIBE_INFERENCE", "fake")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("TRIBE_QUEUE_DEPTH", "2")

    real_run_worker = worker_mod.run_worker
    gate_holder: dict = {}

    async def gated_run_worker(store, process_window, stop, **kw):
        gate = _asyncio.Event()
        gate_holder["gate"] = gate
        gate_holder["loop"] = _asyncio.get_running_loop()
        gate_task = _asyncio.create_task(gate.wait())
        stop_task = _asyncio.create_task(stop.wait())
        try:
            done, pending = await _asyncio.wait(
                {gate_task, stop_task}, return_when=_asyncio.FIRST_COMPLETED
            )
        except _asyncio.CancelledError:
            for t in (gate_task, stop_task):
                t.cancel()
            raise
        for t in pending:
            t.cancel()
        if stop.is_set():
            return
        return await real_run_worker(store, process_window, stop, **kw)

    monkeypatch.setattr(worker_mod, "run_worker", gated_run_worker)

    from tribe_backend.api.app import get_app
    app = get_app()

    def resume():
        gate = gate_holder.get("gate")
        loop = gate_holder.get("loop")
        if gate and loop and loop.is_running():
            loop.call_soon_threadsafe(gate.set)
        elif gate:
            gate.set()

    with TestClient(app) as client:
        yield client, resume
