"""Generate the frontend usage report (plan §9).

Probes a live tribe-backend HTTP server (default ``http://127.0.0.1:8000``),
resolves the cluster host using the Slurm discovery procedure (§9.1), and
renders a self-contained markdown document the frontend team can read
without ever opening this codebase.

The script is **deploy-time** (not plan-time): the cluster host is
unknown until the Slurm allocation lands. The script exits non-zero on
any of:

* the live server is unreachable (``GET /v1/health`` fails),
* the live OpenAPI is unreachable (``GET /openapi.json`` fails),
* the rendered report contains an unresolved ``{CLUSTER_HOST}`` /
  ``{PORT}`` placeholder,
* the OpenAPI spec is missing any of the documented endpoints.

A partially-populated report is worse than no report.

Usage:

    python scripts/generate_usage_report.py
    python scripts/generate_usage_report.py --base-url http://127.0.0.1:8004
    python scripts/generate_usage_report.py --out agent/reports/backend-usage-2026-04-25.md

Per the user decision (Phase E), the **JSON+base64 path was DROPPED** —
the report documents only ``multipart/form-data``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "agent" / "reports" / f"backend-usage-{_dt.date.today().isoformat()}.md"


# --------------------------------------------------------------- discovery

def resolve_cluster_host() -> tuple[str, str]:
    """Return ``(host, source)`` per the §9.1 discovery procedure.

    Tries, in order:
      1. ``$SLURM_JOB_NODELIST`` (single-node form only — does not expand
         a comma/range list, since the API binds to one node).
      2. ``scontrol show node $(hostname)`` — pull NodeAddr.
      3. ``socket.gethostname()`` fallback (and the report flags this as
         "not Slurm-routed" so the frontend dev knows the address may not
         be reachable from outside this box).

    The ``source`` value is one of ``"SLURM_JOB_NODELIST"``,
    ``"scontrol"``, ``"hostname"``.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "").strip()
    if nodelist and "," not in nodelist and "[" not in nodelist:
        return nodelist, "SLURM_JOB_NODELIST"

    try:
        proc = subprocess.run(
            ["scontrol", "show", "node", socket.gethostname()],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if proc.returncode == 0:
            for tok in proc.stdout.split():
                if tok.startswith("NodeAddr="):
                    addr = tok.split("=", 1)[1].strip()
                    if addr:
                        return addr, "scontrol"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return socket.gethostname(), "hostname"


# --------------------------------------------------------------- HTTP probes

class ProbeError(RuntimeError):
    pass


def http_get_json(url: str, timeout: float = 10.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                raise ProbeError(f"GET {url} returned HTTP {resp.status}")
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ProbeError(f"GET {url} failed: {e}") from e


def probe_server(base_url: str) -> tuple[dict, dict]:
    """Return (health_json, openapi_json), raising ProbeError on any miss."""
    base = base_url.rstrip("/")
    health = http_get_json(f"{base}/v1/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise ProbeError(f"/v1/health returned unexpected body: {health!r}")
    openapi = http_get_json(f"{base}/openapi.json")
    if not isinstance(openapi, dict) or "paths" not in openapi:
        raise ProbeError("/openapi.json missing 'paths'")
    return health, openapi


# --------------------------------------------------------------- rendering

REQUIRED_PATHS = (
    "/v1/health",
    "/v1/runs",
    "/v1/runs/{job_id}",
    "/v1/runs/{job_id}/mesh",
    "/v1/runs/{job_id}/mesh/meta",
    "/v1/runs/{job_id}/mesh/colors",
    "/v1/runs/{job_id}/mesh/vertices",
    "/v1/runs/{job_id}/mesh/faces",
)


def _verify_openapi_paths(openapi: dict) -> None:
    have = set(openapi.get("paths", {}).keys())
    missing = [p for p in REQUIRED_PATHS if p not in have]
    if missing:
        raise ProbeError(
            f"OpenAPI spec is missing required endpoints: {missing!r} "
            f"(template would be inconsistent with running server)"
        )


def _render_endpoint_table(openapi: dict) -> str:
    """Render a markdown table of every path/method from openapi.json.

    Drives off the live spec so the report cannot drift from the running
    server. Each row: ``| METHOD | path | summary |``.
    """
    rows = ["| Method | Path | Summary |", "|--------|------|---------|"]
    for path, methods in sorted(openapi.get("paths", {}).items()):
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "delete", "put", "patch"):
                continue
            summary = (op.get("summary") or op.get("operationId") or "").strip()
            rows.append(f"| {method.upper()} | `{path}` | {summary} |")
    return "\n".join(rows)


def render_report(
    *,
    base_url: str,
    cluster_host: str,
    cluster_host_source: str,
    port: int,
    health: dict,
    openapi: dict,
    today: str,
) -> str:
    """Return the populated markdown body. No placeholders may survive."""
    _verify_openapi_paths(openapi)

    inference = health.get("inference", "unknown")
    out_dir = health.get("out_dir", "<unknown>")
    endpoint_table = _render_endpoint_table(openapi)
    health_pretty = json.dumps(health, indent=2)
    openapi_version = openapi.get("info", {}).get("version", "unknown")

    not_slurm_note = ""
    if cluster_host_source == "hostname":
        not_slurm_note = (
            "\n> **Note:** The cluster host above was derived from "
            "``socket.gethostname()`` because ``$SLURM_JOB_NODELIST`` is "
            "unset and ``scontrol show node`` was unavailable. This is **not** "
            "a Slurm-routed hostname; the frontend may need to use an SSH "
            "tunnel or reverse proxy to reach it.\n"
        )

    return f"""# tribe-backend HTTP API — Frontend Usage ({today})

This document is generated by ``scripts/generate_usage_report.py`` and probes
the live server at the time of generation. It is the single source of truth
for the frontend team — you do **not** need to read the codebase, the plan,
or the OpenAPI schema to use the API.

> Server probed: ``{base_url}``  ·  Inference backend: **{inference}**  ·  OpenAPI ``{openapi_version}``  ·  Generated {today}

---

## A. Endpoint base URL

- **Cluster host:** ``{cluster_host}`` (resolved via ``{cluster_host_source}``)
- **Port:** ``{port}``
- **BASE_URL:** ``http://{cluster_host}:{port}``
{not_slurm_note}
**Live ``GET /v1/health`` response captured at generation:**

```json
{health_pretty}
```

If the frontend network cannot reach the compute node directly (the common
case on shared HPC), set up a local-port-forward tunnel:

```bash
ssh -L {port}:{cluster_host}:{port} <head-node>
# then use http://localhost:{port} from the frontend
```

---

## B. Request schema — `POST /v1/runs`

> **multipart/form-data only.** The JSON+base64 convenience path was
> dropped before E ship because it added a second contract surface for no
> measurable frontend benefit.

| field      | type   | required | example                          | notes                                          |
|------------|--------|----------|----------------------------------|------------------------------------------------|
| `media`    | file   | yes      | `clip.mp4` (30s, H.264, ~12 MB)  | mp4 or webm, ≤200 MB, audio track preferred    |
| `text`     | str    | yes      | `"The quick brown fox..."`       | UTF-8, no length cap in v1                     |
| `wait`     | bool   | no       | `false` (default)                | **400 if true and inference=gpu** — see error catalog |

Server-derived fields (no client input needed): ``audio_sr``, ``video_fps``,
``t_start``, ``window_id``. The video FPS is probed from the mp4 container;
audio is decoded from the embedded track or synthesized as silence if
absent.

---

## C. Response schema

### 202 Accepted (async submit, default)

```json
{{
  "job_id": "550e8400e29b41d4a716446655440000",
  "status": "queued"
}}
```

### 200 OK on `GET /v1/runs/{{job_id}}`

```json
{{
  "job_id": "550e8400...",
  "status": "done",
  "submitted_at": "2026-04-25T14:02:11Z",
  "started_at":   "2026-04-25T14:02:12Z",
  "finished_at":  "2026-04-25T14:07:13Z",
  "report": {{
    "window_id": "550e8400...",
    "text": "The quick brown fox.",
    "method": "z_mean",
    "z_threshold": 1.0,
    "top_regions": [
      {{ "name": "L_FFC_ROI",  "z_score": 3.42, "direction": "positive",
         "description": "..." }},
      {{ "name": "R_V1_ROI",   "z_score": 2.18, "direction": "positive",
         "description": "..." }}
    ]
  }},
  "mesh": "/v1/runs/550e8400.../mesh"
}}
```

`status ∈ {{queued, running, done, failed, cancelled}}`. On `failed` the body
carries an ``error: {{code, message}}`` instead of ``report``.

### 200 OK on `GET /v1/runs/{{job_id}}/mesh`

```json
{{
  "job_id": "550e8400...",
  "artifacts": {{
    "meta":     "/v1/runs/{{job_id}}/mesh/meta",
    "colors":   "/v1/runs/{{job_id}}/mesh/colors",
    "vertices": "/v1/runs/{{job_id}}/mesh/vertices",
    "faces":    "/v1/runs/{{job_id}}/mesh/faces"
  }}
}}
```

Content-types: ``application/json`` for ``meta``; ``application/octet-stream``
for ``colors``, ``vertices``, ``faces``.

---

## D. Worked curl examples

All commands assume ``BASE_URL=http://{cluster_host}:{port}`` (or
``http://localhost:{port}`` if you set up the SSH tunnel from §A).

**1. Submit a job (multipart):**

```bash
curl -s -X POST $BASE_URL/v1/runs \\
  -F "media=@clip.mp4" \\
  -F "text=The quick brown fox jumps over the lazy dog."
# → 202 {{ "job_id": "...", "status": "queued" }}
```

**2. Poll status:**

```bash
curl -s $BASE_URL/v1/runs/$JOB_ID
# → 200 {{ "status": "running", ... }}
# ... ~5 min later ...
# → 200 {{ "status": "done", "report": {{...}}, "mesh": "..." }}
```

**3. Fetch the mesh manifest, then artifacts:**

```bash
curl -s $BASE_URL/v1/runs/$JOB_ID/mesh
# → 200 {{ "artifacts": {{ "meta": "...", "colors": "...", ... }} }}

curl -s  $BASE_URL/v1/runs/$JOB_ID/mesh/meta              > meta.json
curl -so brain_colors.bin   $BASE_URL/v1/runs/$JOB_ID/mesh/colors
curl -so brain_vertices.bin $BASE_URL/v1/runs/$JOB_ID/mesh/vertices
curl -so brain_faces.bin    $BASE_URL/v1/runs/$JOB_ID/mesh/faces
```

**4. Cancel a queued job:**

```bash
curl -s -X DELETE $BASE_URL/v1/runs/$JOB_ID -o /dev/null -w "%{{http_code}}\\n"
# → 204 (queued)  |  409 (running, not cancellable)  |  404 (unknown)
```

---

## E. CORS guidance

- Default dev allowlist: ``*`` (controlled by ``CORS_ALLOW_ORIGINS`` env var).
- In production GPU mode the server still reads ``CORS_ALLOW_ORIGINS``; the
  Slurm operator should set it to your frontend origin (e.g.
  ``CORS_ALLOW_ORIGINS=https://your-frontend.example``).
- To request a change: open a PR editing ``tribe_backend/api/settings.py`` or
  set the env var on the Slurm job.
- Preflight ``OPTIONS`` is handled by ``fastapi.middleware.cors.CORSMiddleware``
  — the frontend does not need to do anything beyond a normal browser fetch.

---

## F. Latency expectations

- **GPU mode (``TRIBE_INFERENCE=gpu``):** ~5 minutes per 30s clip on a single
  A6000. Use the async path (POST → 202 → poll). ``wait=true`` is **rejected
  with HTTP 400 / ``WAIT_DISABLED_GPU``** in this mode.
- **Fake mode (``TRIBE_INFERENCE=fake``):** sub-second. ``wait=true`` is
  permitted and convenient for frontend integration tests.
- Implication: any UI showing a progress indicator should plan for a
  5-minute spinner with a cancellation affordance.

---

## G. Error catalog

Every error response uses the envelope ``{{"error": {{"code": "...", "message": "..."}}}}``.
Tracebacks are never returned.

| Status | `error.code`               | When                                             | Frontend action                                  |
|--------|----------------------------|--------------------------------------------------|--------------------------------------------------|
| 400    | `decode_error`             | Uploaded media isn't a valid mp4/webm            | "Couldn't read your video, try re-encoding"      |
| 400    | `WAIT_DISABLED_GPU`        | `wait=true` with ``TRIBE_INFERENCE=gpu``         | Drop the wait flag and use the async path        |
| 422    | `validation_error`         | Pydantic validation (missing `text` / `media`)   | Show field-level message from ``error.message``  |
| 404    | `JOB_NOT_FOUND`            | Unknown ``job_id`` on GET / DELETE / mesh        | Treat as expired; restart submission             |
| 409    | `JOB_NOT_CANCELLABLE`      | DELETE on a running/done/failed job              | Disable cancel button once status="running"      |
| 409    | `JOB_NOT_READY`            | Mesh artifact requested while queued/running     | Poll job status first, then fetch mesh           |
| 410    | `JOB_FAILED`               | Mesh artifact requested on a failed/cancelled job| Surface the original failure; mesh will not exist|
| 500    | `INFERENCE_ARTIFACTS_MISSING` | Job done but artifact missing on disk         | Report a server bug; restart submission          |
| 500    | `INFERENCE_FAILURE`        | Inference crashed mid-run (rare)                 | Show "Processing failed", offer retry            |
| 503    | `QUEUE_FULL`               | Submission past `TRIBE_QUEUE_DEPTH`              | "Server busy, retry in a few minutes"            |

---

## Appendix — Live OpenAPI endpoint inventory

The following table is rendered directly from the running server's
``/openapi.json`` at report-generation time. If your client breaks because a
path here is missing, the server you are pointed at is older than this
document.

{endpoint_table}
"""


# --------------------------------------------------------------- main

def _check_no_unresolved_placeholders(text: str) -> None:
    bad_tokens = ("{CLUSTER_HOST}", "{PORT}", "<TODO>")
    for tok in bad_tokens:
        if tok in text:
            raise ProbeError(f"rendered report still contains placeholder {tok!r}")


def _port_from_url(url: str, default: int = 8000) -> int:
    m = re.search(r":(\d+)(?:/|$)", url)
    return int(m.group(1)) if m else default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the frontend usage report.")
    parser.add_argument("--base-url", default=os.environ.get("TRIBE_BASE_URL", DEFAULT_BASE_URL),
                        help=f"Live API URL to probe (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output markdown path (default: agent/reports/backend-usage-{today}.md)")
    parser.add_argument("--cluster-host", default=None,
                        help="Override cluster host (otherwise resolved via SLURM/scontrol/hostname)")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port (otherwise parsed from --base-url)")
    args = parser.parse_args(argv)

    try:
        health, openapi = probe_server(args.base_url)
    except ProbeError as e:
        print(f"ERROR: probing live server failed: {e}", file=sys.stderr)
        return 2

    if args.cluster_host:
        cluster_host, source = args.cluster_host, "cli-override"
    else:
        cluster_host, source = resolve_cluster_host()

    port = args.port if args.port is not None else _port_from_url(args.base_url)

    today = _dt.date.today().isoformat()

    try:
        body = render_report(
            base_url=args.base_url,
            cluster_host=cluster_host,
            cluster_host_source=source,
            port=port,
            health=health,
            openapi=openapi,
            today=today,
        )
        _check_no_unresolved_placeholders(body)
    except ProbeError as e:
        print(f"ERROR: rendering report failed: {e}", file=sys.stderr)
        return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
