"""Path-traversal hardening for /v1/runs/{job_id}/mesh/* endpoints.

A malicious or malformed job_id (URL-encoded ../, raw slashes, nulls,
etc.) must 404 BEFORE the routing layer touches the filesystem. We
verify both the response code AND that no on-disk read of /etc/passwd
or other escape-target paths could possibly succeed (the artifact name
is a closed allowlist anyway, but defense in depth matters).
"""
from __future__ import annotations

import pytest


# Path components that should all 404 — none of these are valid uuid hex
# nor valid window_id (ISO timestamp + 6-hex suffix).
TRAVERSAL_IDS = [
    "..%2F..%2Fetc%2Fpasswd",   # URL-encoded
    "..%5C..%5Cwindows",        # backslash variant
    "..",
    ".",
    "%2E%2E",
    "%00",
    "abc%2Fdef",                # encoded slash
    "..;/passwd",
    "%252e%252e%252f",          # double-encoded
]


@pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
def test_path_traversal_in_job_id_returns_404(test_client, bad_id):
    r = test_client.get(f"/v1/runs/{bad_id}/mesh/colors")
    # Either Starlette decodes and we hit JOB_NOT_FOUND, OR it 404s on the
    # routing layer because the path doesn't match. Either way: never 200,
    # never 5xx, never an actual file read.
    assert r.status_code == 404, (bad_id, r.status_code, r.text)


@pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
def test_path_traversal_on_manifest_returns_404(test_client, bad_id):
    r = test_client.get(f"/v1/runs/{bad_id}/mesh")
    assert r.status_code == 404, (bad_id, r.status_code, r.text)


def test_etc_passwd_attempt_yields_safe_envelope(test_client):
    """The error envelope must never hint at the on-disk layout.

    Two acceptable outcomes:
      1. Starlette's router refuses the URL before our handler runs and
         returns the default 404 envelope (code "not_found").
      2. Our handler runs and returns the JOB_NOT_FOUND envelope.
    Both are safe — the prohibited outcome is leaking a filesystem path
    in the message body.
    """
    r = test_client.get("/v1/runs/..%2F..%2Fetc%2Fpasswd/mesh/colors")
    assert r.status_code == 404
    body = r.json()
    if "error" in body:
        assert body["error"]["code"] in ("JOB_NOT_FOUND", "not_found")
        # Message must not echo a real filesystem layout (we explicitly never
        # mention "out_dir", "/etc/passwd", or absolute paths in handler msgs).
        msg = body["error"]["message"].lower()
        assert "/etc/" not in msg
        assert "out_dir" not in msg


def test_traversal_id_does_not_reach_jobstore(test_client, monkeypatch):
    """Spy on JobStore.get to confirm it's never called for a malformed id."""
    from tribe_backend.api import jobs as jobs_mod

    real_get = jobs_mod.JobStore.get
    calls: list[str] = []

    async def spy_get(self, job_id):
        calls.append(job_id)
        return await real_get(self, job_id)

    monkeypatch.setattr(jobs_mod.JobStore, "get", spy_get)

    test_client.get("/v1/runs/..%2F..%2Fetc%2Fpasswd/mesh/colors")
    test_client.get("/v1/runs/foo%2Fbar/mesh/meta")

    # Neither malformed id should have reached JobStore.get; if any did, it
    # means the sanitizer is bypassed. (A passing case where the framework
    # 404s before our handler runs also yields zero calls, which is fine.)
    for c in calls:
        # If anything reached JobStore, it must be a safe id (which would not
        # match these attack strings).
        assert "/" not in c and ".." not in c and "%" not in c
