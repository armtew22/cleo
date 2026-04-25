"""Phase E.1 — `?wait=true` is rejected when TRIBE_INFERENCE=gpu.

A 5-minute synchronous request would be a UX disaster on the GPU path. The
fake inference path is permissive (synchronous wait is convenient for dev),
but production must refuse it with a structured 400 / `WAIT_DISABLED_GPU`.

We exercise the route directly without actually loading torch/weights by
faking the lifespan with a stub inference. The `Settings.inference == "gpu"`
flag is what gates the rejection — that flag is taken from env at app build
time, so we set TRIBE_INFERENCE=gpu but monkeypatch `build_inference` so the
gpu adapter is never constructed in this CPU test.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def gpu_mode_test_client(tmp_path, monkeypatch):
    """TestClient where Settings.inference == "gpu" but the inference impl is a fake.

    Lets us test the route-level wait-disable enforcement without touching torch.
    """
    monkeypatch.setenv("TRIBE_INFERENCE", "gpu")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")

    from tests.conftest import make_synthetic
    from tribe_backend.control import factory as factory_mod

    real_build = factory_mod.build_inference

    def fake_build(name, *, fake_fixture=None):
        # Force the "fake" implementation regardless of name so we never
        # touch torch in this CPU test, but Settings.inference still reads "gpu".
        return real_build("fake", fake_fixture=make_synthetic())

    monkeypatch.setattr(factory_mod, "build_inference", fake_build)
    # app.py imports build_inference into its own namespace; patch there too.
    from tribe_backend.api import app as app_mod
    monkeypatch.setattr(app_mod, "build_inference", fake_build)

    from tribe_backend.api.app import get_app
    app = get_app()
    with TestClient(app) as client:
        yield client


def test_health_reports_gpu_in_gpu_mode(gpu_mode_test_client):
    r = gpu_mode_test_client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["inference"] == "gpu"


def test_wait_true_rejected_in_gpu_mode(gpu_mode_test_client, tiny_synthetic_mp4):
    """POST /v1/runs with wait=true must 400 in gpu mode BEFORE any decoding/inference.

    Using the tiny synthetic mp4 is fine here: the wait-disable check must
    fire ahead of contract validation. If someone reorders the route so
    decoding runs first, this test catches it (a 400 decode_error or 5xx
    contract failure would surface instead of WAIT_DISABLED_GPU).
    """
    files = {"media": ("clip.mp4", io.BytesIO(tiny_synthetic_mp4), "video/mp4")}
    data = {"text": "smoke", "wait": "true"}
    r = gpu_mode_test_client.post("/v1/runs", files=files, data=data)
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "WAIT_DISABLED_GPU"
    # Don't leak operational details — just a clear message.
    assert "gpu" in body["error"]["message"].lower()


def test_wait_false_still_works_in_gpu_mode(gpu_mode_test_client, tiny_synthetic_mp4):
    """Async submission (default wait=false) must still succeed in gpu mode."""
    files = {"media": ("clip.mp4", io.BytesIO(tiny_synthetic_mp4), "video/mp4")}
    data = {"text": "smoke"}
    r = gpu_mode_test_client.post("/v1/runs", files=files, data=data)
    assert r.status_code == 202, r.text
    assert "job_id" in r.json()


def test_wait_true_still_works_in_fake_mode(test_client, real_30s_clip_bytes):
    """The fake-mode wait=true convenience path is untouched.

    Uses the real 30s clip (≥25s required by StimulusWindow contract).
    """
    files = {"media": ("clip.mp4", io.BytesIO(real_30s_clip_bytes), "video/mp4")}
    data = {"text": "smoke", "wait": "true"}
    r = test_client.post("/v1/runs", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "top_regions" in body
