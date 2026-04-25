"""Phase E.1 — `python -m tribe_backend.api` invokes uvicorn with workers=1.

We don't actually start a server; we monkeypatch ``uvicorn.run`` and
inspect the call. The single-worker rule is the load-bearing assertion:
multiple workers would each load the GPU weights into the same device
and OOM, and would split the in-memory JobStore.
"""
from __future__ import annotations

import sys

import pytest


def test_cli_invokes_uvicorn_with_workers_one(monkeypatch):
    captured: dict = {}

    import uvicorn  # noqa: F401  (real import target we patch)

    def fake_run(app_target, **kwargs):
        captured["app_target"] = app_target
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    # Import & invoke the CLI's main() — re-import in case a prior test
    # cached it with a different uvicorn binding.
    sys.modules.pop("tribe_backend.api.__main__", None)
    from tribe_backend.api.__main__ import main

    main()

    assert captured["app_target"] == "tribe_backend.api.app:get_app"
    assert captured["factory"] is True
    assert captured["workers"] == 1, "single-GPU constraint requires --workers 1"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8000
    assert captured["reload"] is False


def test_cli_respects_host_and_port_env(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8004")

    sys.modules.pop("tribe_backend.api.__main__", None)
    from tribe_backend.api.__main__ import main

    main()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8004
    assert captured["workers"] == 1


def test_cli_rejects_non_integer_port(monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    monkeypatch.setenv("PORT", "not-a-number")

    sys.modules.pop("tribe_backend.api.__main__", None)
    from tribe_backend.api.__main__ import main

    with pytest.raises(SystemExit):
        main()
