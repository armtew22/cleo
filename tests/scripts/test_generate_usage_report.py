"""Phase E.2 — `scripts/generate_usage_report.py`.

Spins a real uvicorn-equivalent server (uvicorn in a thread) backed by
TestClient's ASGI transport — actually we bind the FastAPI app to a real
TCP socket via ``uvicorn.Server`` running in a thread so the script's
``urllib.request`` probes hit a real HTTP listener (TestClient's ASGI
transport doesn't expose a TCP port).

Verifies:
    - report renders to disk with zero unresolved placeholders,
    - every endpoint from openapi.json is listed in the appendix table,
    - cluster host (or hostname fallback) appears in the §A block,
    - script exits non-zero on missing live server.
"""
from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "generate_usage_report.py"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@contextlib.contextmanager
def _live_server(tmp_path, monkeypatch_env):
    """Run uvicorn in a thread on a free port. Yields ``(base_url, port)``.

    Using uvicorn (not TestClient) so the script's urllib probes hit a
    real TCP socket. We use TRIBE_INFERENCE=fake so no GPU is needed.
    """
    import uvicorn
    from tribe_backend.api.app import get_app

    # Per-server env (fake mode + isolated out_dir).
    import os
    old = {k: os.environ.get(k) for k in ("TRIBE_INFERENCE", "OUT_DIR", "CORS_ALLOW_ORIGINS")}
    os.environ["TRIBE_INFERENCE"] = "fake"
    os.environ["OUT_DIR"] = str(tmp_path / "out")
    os.environ["CORS_ALLOW_ORIGINS"] = "*"
    try:
        app = get_app()
        port = _free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        # Wait for /v1/health to be reachable.
        import urllib.request, urllib.error
        deadline = time.monotonic() + 15.0
        last_err = None
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=1.0).read()
                break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                time.sleep(0.1)
        else:
            server.should_exit = True
            raise RuntimeError(f"test server did not start: {last_err!r}")

        yield f"http://127.0.0.1:{port}", port
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_script(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**__import__("os").environ}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env=env, timeout=30,
    )


def test_script_writes_populated_report(tmp_path, monkeypatch):
    """End-to-end: probe the live server, render to disk, verify contents."""
    out = tmp_path / "out.md"
    with _live_server(tmp_path, monkeypatch) as (base_url, port):
        proc = _run_script(
            "--base-url", base_url,
            "--out", str(out),
            "--cluster-host", "test-node-99",
            "--port", str(port),
        )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert out.exists()

    text = out.read_text()
    # No unresolved placeholders.
    for tok in ("{CLUSTER_HOST}", "{PORT}", "<TODO>"):
        assert tok not in text, f"placeholder {tok!r} survived rendering"

    # Sections A-G must be present.
    for section in (
        "## A. Endpoint base URL",
        "## B. Request schema",
        "## C. Response schema",
        "## D. Worked curl examples",
        "## E. CORS guidance",
        "## F. Latency expectations",
        "## G. Error catalog",
    ):
        assert section in text, f"missing section {section!r}"

    # Cluster host substituted in.
    assert "test-node-99" in text
    assert f":{port}" in text

    # Every openapi path appears in the appendix table.
    for path in (
        "/v1/health",
        "/v1/runs",
        "/v1/runs/{job_id}",
        "/v1/runs/{job_id}/mesh",
        "/v1/runs/{job_id}/mesh/meta",
        "/v1/runs/{job_id}/mesh/colors",
        "/v1/runs/{job_id}/mesh/vertices",
        "/v1/runs/{job_id}/mesh/faces",
    ):
        assert f"`{path}`" in text, f"openapi path {path!r} missing from appendix"

    # JSON+base64 path must NOT appear (was dropped per user decision).
    assert "video_b64" not in text
    assert "audio_b64" not in text

    # WAIT_DISABLED_GPU error must be documented.
    assert "WAIT_DISABLED_GPU" in text


def test_script_fails_when_server_unreachable(tmp_path):
    out = tmp_path / "out.md"
    proc = _run_script(
        "--base-url", "http://127.0.0.1:1",  # nothing listens here
        "--out", str(out),
        "--cluster-host", "x",
        "--port", "1",
    )
    assert proc.returncode != 0
    assert "probing live server failed" in proc.stderr.lower() or "failed" in proc.stderr.lower()
    assert not out.exists()


def test_resolve_cluster_host_returns_a_string():
    """The resolver must always return a non-empty (host, source) pair."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import generate_usage_report as gur
    finally:
        sys.path.pop(0)

    host, source = gur.resolve_cluster_host()
    assert isinstance(host, str) and host
    assert source in ("SLURM_JOB_NODELIST", "scontrol", "hostname")


def test_render_rejects_openapi_missing_paths(tmp_path):
    """If openapi.json lacks any required endpoint, render must raise ProbeError."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import generate_usage_report as gur
    finally:
        sys.path.pop(0)

    truncated = {"paths": {"/v1/health": {"get": {}}}, "info": {"version": "0"}}
    with pytest.raises(gur.ProbeError):
        gur.render_report(
            base_url="http://x",
            cluster_host="x", cluster_host_source="hostname",
            port=8000,
            health={"status": "ok", "inference": "fake", "out_dir": "/tmp"},
            openapi=truncated,
            today="2026-04-25",
        )
