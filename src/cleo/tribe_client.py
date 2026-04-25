"""HTTP client for the remote TRIBE v2 backend (ellis-compute-02.cs.cornell.edu).

The backend exposes (per ``instructions_for-GPU.md``):
  GET    /v1/health            → {"status":"ok","inference":"gpu"|"fake","out_dir":"..."}
  POST   /v1/runs              → multipart {media, text} → 202 {"job_id":...,"status":"queued"}
  GET    /v1/runs/{id}         → {status, report?, error?, mesh?, ...}
  DELETE /v1/runs/{id}         → 204 (cancellable) | 409 | 404
  GET    /v1/runs/{id}/mesh    → manifest of artifact URLs

A real GPU forward pass takes ≈5 min, so the client polls and tolerates 503
(queue full) with one short retry. Errors are normalised to ``TribeBackendError``.

Env knobs:
  TRIBE_BACKEND_URL     base URL (default ellis-compute-02:8000)
  TRIBE_POLL_INTERVAL   seconds between polls (default 5)
  TRIBE_TIMEOUT         hard ceiling for a single job (default 900s = 15 min)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx

DEFAULT_TRIBE_URL = os.environ.get(
    "TRIBE_BACKEND_URL",
    "http://ellis-compute-02.cs.cornell.edu:8004",
)
DEFAULT_POLL_INTERVAL = float(os.environ.get("TRIBE_POLL_INTERVAL", "5"))
DEFAULT_TIMEOUT = float(os.environ.get("TRIBE_TIMEOUT", "900"))

# Status callback signature: ``cb(status: str, job: dict) -> None``
StatusCallback = Callable[[str, dict[str, Any]], None]


class TribeBackendError(RuntimeError):
    """Raised when the remote backend reports an error or is unreachable."""


class TribeClient:
    """Thin sync HTTP wrapper around the remote TRIBE v2 service."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        connect_timeout: float = 10.0,
        request_timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_TRIBE_URL).rstrip("/")
        self._timeout = httpx.Timeout(request_timeout, connect=connect_timeout)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{self.base_url}/v1/health")
            r.raise_for_status()
            return r.json()

    def submit(self, video_path: Path, caption: str) -> str:
        """POST /v1/runs — returns ``job_id``. Retries once on 503 QUEUE_FULL."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        for attempt in range(2):
            with httpx.Client(timeout=self._timeout) as c:
                with video_path.open("rb") as fh:
                    files = {"media": (video_path.name, fh, "video/mp4")}
                    data = {"text": caption}
                    try:
                        r = c.post(f"{self.base_url}/v1/runs", files=files, data=data)
                    except httpx.HTTPError as e:
                        raise TribeBackendError(
                            f"could not reach TRIBE backend at {self.base_url}: {e}"
                        ) from e

            if r.status_code == 503 and attempt == 0:
                retry_after = float(r.headers.get("Retry-After", "5"))
                time.sleep(retry_after)
                continue
            if r.status_code >= 400:
                raise TribeBackendError(
                    f"submit failed (HTTP {r.status_code}): {_safe_error(r)}"
                )
            payload = r.json()
            job_id = payload.get("job_id")
            if not job_id:
                raise TribeBackendError(f"submit returned no job_id: {payload}")
            return job_id

        raise TribeBackendError("submit failed after retry on 503 QUEUE_FULL")

    def get_job(self, job_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{self.base_url}/v1/runs/{job_id}")
            if r.status_code == 404:
                raise TribeBackendError(f"job {job_id} not found")
            r.raise_for_status()
            return r.json()

    def cancel(self, job_id: str) -> bool:
        """DELETE /v1/runs/{id} — returns True if cancelled, False if not cancellable."""
        with httpx.Client(timeout=self._timeout) as c:
            r = c.delete(f"{self.base_url}/v1/runs/{job_id}")
        if r.status_code == 204:
            return True
        if r.status_code in (404, 409):
            return False
        r.raise_for_status()
        return False

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        on_status: StatusCallback | None = None,
    ) -> dict[str, Any]:
        """Poll until terminal. Returns the final job dict on ``done``; raises otherwise."""
        deadline = time.monotonic() + timeout
        last_status: str | None = None
        while True:
            if time.monotonic() > deadline:
                raise TribeBackendError(
                    f"job {job_id} did not complete within {timeout:.0f}s"
                )
            job = self.get_job(job_id)
            status = str(job.get("status") or "")
            if status != last_status:
                last_status = status
                if on_status is not None:
                    try:
                        on_status(status, job)
                    except Exception:  # noqa: BLE001 — callback errors must not kill the wait
                        pass
            if status == "done":
                return job
            if status == "failed":
                err = job.get("error") or {}
                raise TribeBackendError(
                    f"TRIBE inference failed: "
                    f"{err.get('code', 'UNKNOWN')} — {err.get('message', '(no message)')}"
                )
            if status == "cancelled":
                raise TribeBackendError(f"job {job_id} was cancelled")
            time.sleep(poll_interval)

    def submit_and_wait(
        self,
        video_path: Path,
        caption: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        on_status: StatusCallback | None = None,
    ) -> dict[str, Any]:
        job_id = self.submit(video_path, caption)
        if on_status is not None:
            try:
                on_status("submitted", {"job_id": job_id})
            except Exception:  # noqa: BLE001
                pass
        return self.wait(
            job_id,
            poll_interval=poll_interval,
            timeout=timeout,
            on_status=on_status,
        )

    # ── mesh artifacts ────────────────────────────────────────────────────────

    def fetch_mesh_manifest(self, job_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{self.base_url}/v1/runs/{job_id}/mesh")
            r.raise_for_status()
            return r.json()

    def fetch_mesh_artifact(self, job_id: str, kind: str) -> bytes:
        """Fetch one of: meta | colors | vertices | faces. Returns raw bytes (or JSON for meta)."""
        if kind not in ("meta", "colors", "vertices", "faces"):
            raise ValueError(f"unknown mesh artifact kind: {kind!r}")
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{self.base_url}/v1/runs/{job_id}/mesh/{kind}")
            r.raise_for_status()
            return r.content


def _safe_error(r: httpx.Response) -> str:
    """Return the backend error envelope ``{error: {code, message}}`` if present."""
    try:
        payload = r.json()
    except ValueError:
        return r.text[:200]
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict):
        return f"{err.get('code', '?')}: {err.get('message', '')}"
    return str(payload)[:200]
