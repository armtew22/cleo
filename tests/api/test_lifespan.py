"""App startup spawns the worker; shutdown stops it cleanly."""
from __future__ import annotations

import asyncio
import warnings


def test_lifespan_starts_and_stops_worker(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("TRIBE_INFERENCE", "fake")
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))

    from tribe_backend.api.app import get_app
    app = get_app()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TestClient(app) as client:
            r = client.get("/v1/health")
            assert r.status_code == 200
            # Worker task must be present and running:
            assert hasattr(app.state, "worker_task")
            assert app.state.worker_task is not None
            assert not app.state.worker_task.done()
            assert hasattr(app.state, "jobs")
            assert hasattr(app.state, "stop_event")
        # After exiting the TestClient context, the lifespan shutdown ran.
        assert app.state.worker_task.done()

    # No "Task was destroyed but it is pending" warnings.
    leak = [w for w in caught if "was destroyed" in str(w.message)]
    assert not leak, [str(w.message) for w in leak]


def test_health_still_works(test_client):
    r = test_client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["inference"] == "fake"
