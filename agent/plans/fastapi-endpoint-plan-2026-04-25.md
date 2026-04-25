# FastAPI HTTP Endpoint Layer — Implementation Plan

> **PREREQUISITE — Before executing any phase of this plan, run the `/prime` command to initialize your understanding of the codebase architecture. Do not proceed until priming is complete.**

**Goal**: Add a thin FastAPI HTTP layer at `tribe_backend/api/` that exposes the existing `ControlUnit.process_window` flow over JSON/HTTP so the frontend can submit a stimulus window and receive a `Report` plus mesh artifact links — without re-implementing inference, parcellation, or mesh export.

**Context & Assumptions**:
- The 5-phase TDD backend plan (`agent/plans/tdd-pipeline-plan.md`) has been merged. `default_control_unit()` is the composition root, `ControlUnit.process_window(...)` is the single entry point, and the Phase-5 GPU adapter (`GpuTribeInference`) loads weights once and keeps them in VRAM.
- The contract is frozen: `StimulusWindow(video, audio, audio_sr, text)`, `TribeOutput((T, 20484))`, `Report` with `to_json/from_json`. The API layer must respect this contract — no widening, no changes.
- A real GPU forward pass takes ~5 minutes per 30s window on an A6000, so synchronous request/response is not viable for production.
- Only one model fits in VRAM, so a single-worker job queue is sufficient for v1.
- The existing import-linter contracts forbid `parcellation`/`mesh`/`inference` from importing `control`. We will add a new contract that forbids `api/` from importing `parcellation`/`mesh`/`inference` directly — it must go through `control` and `contracts`.
- FakeTribeInference is the default for local dev so the API is fully testable with no GPU.

**Risks & Unknowns**:
- **Job persistence**: an in-memory job dict loses state on restart. Acceptable for v1 (jobs take 5 min, restarts are rare), but document the upgrade path.
- **Backpressure**: with one GPU worker, a queue of N pending jobs means the Nth client waits N×5 min. Need a max-queue-depth and a 503 on overflow.
- **Memory pressure from multipart uploads**: a 30s mp4 can be tens of MB; we must spool to disk, not buffer in RAM.
- **Video decoding**: `process_window` expects a raw `(F, H, W, 3) uint8` array. The frontend sends an mp4 blob. We need a decode step (probably `imageio-ffmpeg` or `av`); confirm what the existing pipeline's test fixtures use and reuse that.
- **CORS in dev vs prod**: dev allowlist (`http://localhost:5173` etc.) must not leak into prod.
- **Timeouts**: uvicorn default keepalive may kill long sync requests in `?wait=true` mode against the real GPU. Document but discourage that combination.
- **GPU worker crash**: if the worker task dies, jobs hang in `running`. Need a watchdog or at least a clear error path.

---

## §0 Context

This plan extends the work merged from `agent/plans/tdd-pipeline-plan.md`. That plan delivered the offline pipeline: parcellation, mesh export, GPU inference adapter, and the `ControlUnit` composition root. This plan adds **only** the HTTP transport — every brain-related computation stays where it already lives. The user's directive is "the flow should be identical": the API handler unpacks the request, calls `ControlUnit.process_window(...)`, and serializes the resulting `Report`. Nothing more.

---

## §1 API Surface

### Routes

| Method | Path                                  | Purpose                                              | Success | Errors                  |
|--------|---------------------------------------|------------------------------------------------------|---------|-------------------------|
| GET    | `/v1/health`                          | Liveness + which inference backend is wired          | 200     | —                       |
| POST   | `/v1/runs`                            | Submit a window. `?wait=true` → sync; default async  | 202 / 200 | 400, 422, 503, 500    |
| GET    | `/v1/runs/{job_id}`                   | Job status + Report when done                        | 200     | 404                     |
| GET    | `/v1/runs/{job_id}/stream`            | SSE: progress events (Phase D, optional)             | 200     | 404                     |
| GET    | `/v1/runs/{job_id}/mesh/meta`         | Static `brain_meta.json`                             | 200     | 404                     |
| GET    | `/v1/runs/{job_id}/mesh/colors`       | Static `brain_colors.bin`                            | 200     | 404                     |
| GET    | `/v1/runs/{job_id}/mesh/animation/{t}`| Per-frame color file                                 | 200     | 404                     |
| DELETE | `/v1/runs/{job_id}`                   | Cancel a queued job (running jobs are not cancelable in v1) | 204 | 404, 409          |

### Request — `POST /v1/runs`

**Recommended primary path: `multipart/form-data`**

```
media:       file (mp4 or webm, ≤200 MB, spooled to tmp)
text:        form field, str
audio_sr:    form field, int (default 16000; only used if media has no audio track)
video_fps:   form field, int (default 25)
t_start:     form field, float (default 0.0)
window_id:   form field, str (optional; server generates UUID if absent)
```

**Convenience path (small payloads / scripted clients): pure JSON**

```json
{
  "video_b64": "<base64 mp4>",
  "audio_b64": "<base64 wav, optional if mp4 has audio>",
  "audio_sr": 16000,
  "video_fps": 25,
  "text": "...",
  "t_start": 0.0,
  "window_id": "optional-client-supplied-id"
}
```

The handler decodes the mp4 into the raw `(F, H, W, 3) uint8` ndarray and the audio waveform array internally (in `tribe_backend/api/decoding.py`) and then calls `process_window(video, audio, audio_sr, text, video_fps, t_start)` exactly as today.

**Alternatives considered**: presigned-URL upload + reference-by-ID. Defer to v2 — adds an object store dependency, and we have no current storage tier.

### Response

**202 Accepted** (async submit):
```json
{ "job_id": "uuid", "status": "queued", "submitted_at": "2026-04-25T..." }
```

**200 OK** on `GET /v1/runs/{job_id}` (terminal):
```json
{
  "job_id": "uuid",
  "status": "done",
  "submitted_at": "...", "started_at": "...", "finished_at": "...",
  "report": { /* ReportResponse, mirrors Report.to_json() */ },
  "artifacts": {
    "mesh_meta":      "/v1/runs/{job_id}/mesh/meta",
    "mesh_colors":    "/v1/runs/{job_id}/mesh/colors",
    "animation_base": "/v1/runs/{job_id}/mesh/animation"
  }
}
```

`status ∈ {queued, running, done, failed, canceled}`. On `failed`, include `error: {code, message}`; never leak tracebacks.

### Status code map

| Condition                              | Code |
|----------------------------------------|------|
| Submission accepted (async)            | 202  |
| Sync run completed                     | 200  |
| Validation failure (Pydantic)          | 422  |
| Bad media (decode error)               | 400  |
| Queue full                             | 503  |
| Unknown job_id                         | 404  |
| Inference failure during run           | reflected in job `status=failed`, GET returns 200 |
| Server misconfiguration at startup     | process exits non-zero (fail-fast)                |

---

## §2 Module Layout

New package `tribe_backend/api/`:

```
tribe_backend/api/
├── __init__.py
├── app.py          # FastAPI() factory + lifespan (build ControlUnit once)
├── routes.py       # APIRouter with the §1 endpoints
├── schemas.py      # Pydantic models: SubmitForm, ReportResponse, JobStatusResponse, ...
├── jobs.py         # JobStore + GpuWorker (asyncio.Queue + single consumer task)
├── decoding.py     # mp4/wav bytes → (np.ndarray video, np.ndarray audio, sr)
├── deps.py         # FastAPI Depends() providers: get_control_unit, get_job_store
├── errors.py       # exception classes + handlers (InferenceFailure → 500-in-job)
└── settings.py     # pydantic-settings: TRIBE_INFERENCE, TRIBE_OUT_DIR, CORS, queue depth
```

**Import-linter additions (append to `pyproject.toml`):**

```toml
[[tool.importlinter.contracts]]
name = "API talks only to control + contracts"
type = "forbidden"
source_modules = ["tribe_backend.api"]
forbidden_modules = [
    "tribe_backend.parcellation",
    "tribe_backend.mesh",
    "tribe_backend.inference",
]
```

Rationale: the API must stay a thin transport. Only `tribe_backend.control` and `tribe_backend.contracts` are imported from handlers. `decoding.py` may use `numpy` and an ffmpeg lib but does **not** touch parcellation/mesh/inference.

**Composition root** (`app.py`) on startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    inference = GpuTribeInference() if settings.inference == "gpu" else FakeTribeInference()
    cu = default_control_unit(out_dir=settings.out_dir, inference=inference)
    app.state.control_unit = cu
    app.state.jobs = JobStore()
    app.state.worker = asyncio.create_task(gpu_worker(cu, app.state.jobs))
    yield
    app.state.worker.cancel()
```

`GpuTribeInference` and `FakeTribeInference` are the only places where `inference/` is touched, and they are inside `app.py`'s lifespan — which is allowed only if we put the import behind a runtime helper or, cleaner, expose a factory in `tribe_backend.control.factory` that takes a string and returns the inference impl. **Recommendation**: extend `default_control_unit` (or add `build_inference(name: str)`) inside `tribe_backend.control` so `api/` never imports `inference` at all and the import-linter rule above passes cleanly.

---

## §3 Job Lifecycle & Async Worker

### State machine

```
queued ──▶ running ──▶ done
   │           │
   │           └──▶ failed
   └──▶ canceled  (only from queued; running is uninterruptible in v1)
```

### `JobStore` (in-memory, v1)

```python
@dataclass
class Job:
    id: str
    status: Literal["queued", "running", "done", "failed", "canceled"]
    submitted_at: datetime
    started_at:  datetime | None
    finished_at: datetime | None
    report: Report | None
    error: ErrorInfo | None
    payload_path: Path           # tmpdir holding the decoded inputs until run

class JobStore:
    def __init__(self): self._jobs: dict[str, Job] = {}
    def create(self, payload_path) -> Job: ...
    def get(self, job_id) -> Job | None: ...
    def update(self, job_id, **fields) -> None: ...
```

### Worker

A single asyncio task pulls one `Job` at a time and runs `process_window` in a thread (it is CPU/GPU-bound and blocking):

```python
async def gpu_worker(cu, jobs, queue):
    while True:
        job = await queue.get()
        jobs.update(job.id, status="running", started_at=now())
        try:
            payload = load_payload(job.payload_path)
            report = await asyncio.to_thread(cu.process_window, **payload)
            jobs.update(job.id, status="done", report=report, finished_at=now())
        except InferenceFailure as e:
            jobs.update(job.id, status="failed", error=ErrorInfo("inference", str(e)))
        except Exception as e:
            jobs.update(job.id, status="failed", error=ErrorInfo("internal", "unhandled"))
            log.exception("worker crash")
```

### Constraints & policies

- `queue.maxsize = settings.queue_depth` (default 4). Submission past max → 503.
- The worker is **per-process**. We do not run multiple uvicorn workers (`--workers 1`) — multiple processes would each try to load weights into the same GPU and OOM.
- `?wait=true` on submission: enqueue, then `await job_done_event` with a configurable timeout. Reject in production (return 400 if `settings.inference == "gpu"`) — the 5-min hold is intentional to keep dev ergonomic but unsafe in prod.

### Persistence upgrade path (documented, not implemented)

- v1: in-process dict, lost on restart.
- v2: SQLite at `out_dir/jobs.db` (single file, no infra) — restartable, single-node.
- v3: Postgres + Redis queue if we ever need horizontal scaling. Given the single-GPU constraint this is unlikely soon.

---

## §4 TDD Phases

Strict red-green-refactor. Every phase ends with `pytest -m "not gpu and not slow"` green and import-linter clean. Tests use `fastapi.testclient.TestClient` (sync) plus `httpx.AsyncClient` for the async + SSE paths.

### Phase A — Schemas + sync endpoint backed by `FakeTribeInference`

Goal: prove the wiring with the fake, no queue yet. `?wait=true` is the **only** mode in this phase.

**Tests (red first):**
1. `test_health_returns_inference_backend` → `GET /v1/health` → `{"status":"ok","inference":"fake"}`
2. `test_submit_multipart_returns_report` → POST a small fixture mp4 + text → 200 with a `ReportResponse` whose `top_regions` is non-empty (FakeTribeInference produces deterministic output).
3. `test_submit_json_b64_returns_report` → same, JSON variant.
4. `test_validation_error_on_missing_text` → 422.
5. `test_decode_error_returns_400` → POST `media=b"not-an-mp4"` → 400.
6. `test_report_response_round_trips_dataclass` → `ReportResponse.model_validate(report.to_json())` succeeds and re-serializes byte-for-byte equal.
7. `test_importlinter_clean` → run lint-imports as a test; api package does not import parcellation/mesh/inference.

**Green deliverables:** `app.py`, `routes.py` (just `POST /v1/runs?wait=true` and `GET /v1/health`), `schemas.py`, `decoding.py`, `deps.py`. Single `ControlUnit` built in lifespan.

**Refactor:** extract `tribe_backend.control.build_inference(name)` so `api/` is import-linter-clean.

### Phase B — Async job queue + status polling

**Tests:**
1. `test_async_submit_returns_202_and_job_id`
2. `test_get_job_queued_then_running_then_done` — drive the FakeTribeInference under a slow stub (monkeypatch a `sleep`) and assert state transitions.
3. `test_unknown_job_returns_404`
4. `test_queue_full_returns_503` — set `queue_depth=1`, submit twice without draining.
5. `test_cancel_queued_job` → 204; subsequent GET shows `canceled`.
6. `test_cannot_cancel_running_job` → 409.
7. `test_inference_failure_marks_job_failed_not_500` — inject a fake that raises `InferenceFailure`; HTTP stays 200, body shows `status:"failed"` with sanitized error.
8. `test_concurrent_submissions_serialize` — submit N=3, assert exactly one running at a time.

**Deliverables:** `jobs.py` with `JobStore` + `gpu_worker` + `asyncio.Queue`. `routes.py` gains `GET /v1/runs/{id}` and `DELETE /v1/runs/{id}`. Default mode is async (202).

### Phase C — Mesh artifact endpoints

**Tests:**
1. `test_mesh_meta_served_after_done` — submit, wait, `GET /v1/runs/{id}/mesh/meta` returns the same JSON written by the mesh exporter.
2. `test_mesh_colors_returns_octet_stream` — content-type `application/octet-stream`, byte-equal to the file in `out_dir`.
3. `test_mesh_animation_frame` — `GET .../animation/0` returns frame 0; out-of-range → 404.
4. `test_artifacts_404_before_done` — querying mesh of a queued/running job → 404 (never partial).
5. `test_path_traversal_blocked` — `GET .../animation/..%2F..%2Fetc%2Fpasswd` → 404.

**Deliverables:** static-file routes scoped under `out_dir / job_id /`. Use `FileResponse`. Validate that `job_id` is a UUID before touching the filesystem.

### Phase D — SSE progress stream (optional, can defer)

**Tests:**
1. `test_sse_emits_state_change_events` — client subscribes; receives `queued → running → done`.
2. `test_sse_closes_on_terminal_state`
3. `test_sse_404_for_unknown_job`

**Deliverables:** `sse-starlette`-based generator that subscribes to a per-job `asyncio.Event`. The worker sets the event on each transition.

Alternatives considered: WebSocket. Reject for v1 — SSE is simpler, reconnect-friendly, frontend-friendly through `EventSource`, and we have no bidirectional needs.

### Phase E — GPU production wiring + smoke

Marked `@pytest.mark.gpu`, only run on the box with weights.

**Tests:**
1. `test_gpu_smoke_one_window_end_to_end` — POST a real 30s fixture, poll until done (timeout 10 min), assert `report.top_regions` non-empty and mesh artifacts exist.
2. `test_gpu_inference_failure_surfaces_as_failed_job` — inject a malformed window; assert `status:"failed"` not 500.
3. `test_weights_missing_fail_fast` — set `TRIBE_INFERENCE=gpu` with weights env unset; uvicorn startup must exit non-zero (lifespan raises).

**Deliverables:** `Settings.inference="gpu"` path, `--workers 1` enforced (assert at startup). README snippet for running.

**Post-startup deliverable (deploy-time):** After uvicorn starts, run `scripts/generate-usage-report.py` which probes the running server (GET `/healthz`, GET `/openapi.json`) and writes the populated report to `agent/reports/backend-usage-{date}.md`, with `{CLUSTER_HOST}:{PORT}` already filled in (resolved via the §9 discovery procedure). The script must fail loudly if either probe fails — a missing or unreachable server means the report would mislead the frontend team.

---

## §5 Integration & User Flow Walkthrough

Frontend (browser, dev origin `http://localhost:5173`):

```bash
# 1. submit
curl -X POST http://localhost:8000/v1/runs \
  -F "media=@clip.mp4" \
  -F "text=The quick brown fox..." \
  -F "video_fps=25"
# → 202 { "job_id": "abc...", "status": "queued" }

# 2. poll
curl http://localhost:8000/v1/runs/abc...
# → 200 { "status": "running", ... }
# (later)
# → 200 { "status": "done", "report": {...}, "artifacts": {...} }

# 3. fetch mesh artifacts (frontend renderer hits these URLs directly)
curl http://localhost:8000/v1/runs/abc.../mesh/meta
curl http://localhost:8000/v1/runs/abc.../mesh/colors -o brain_colors.bin
curl http://localhost:8000/v1/runs/abc.../mesh/animation/0 -o frame_0.bin
```

Frontend pseudo-flow:

```ts
const { job_id } = await fetch("/v1/runs", { method: "POST", body: form }).then(r => r.json());
const evt = new EventSource(`/v1/runs/${job_id}/stream`);   // Phase D
evt.onmessage = (e) => { /* update progress UI */ };
evt.addEventListener("done", async () => {
  const job = await fetch(`/v1/runs/${job_id}`).then(r => r.json());
  renderReport(job.report);
  loadMesh(job.artifacts.mesh_meta, job.artifacts.mesh_colors);
});
```

---

## §6 Acceptance Criteria

- All Phase-A through Phase-C tests green under `pytest -m "not gpu and not slow"`.
- `lint-imports` passes with the new contract.
- `TRIBE_INFERENCE=fake uvicorn tribe_backend.api.app:app` starts cleanly; `/v1/health` responds in <100ms.
- Submitting a fixture window via curl returns a Report whose `top_regions` matches the offline pipeline's output for the same input (byte-equivalent JSON, given the FakeTribeInference is deterministic).
- Mesh endpoints serve byte-identical files to what `BrainMeshExporter` wrote on disk.
- Phase-E GPU smoke test passes on the A6000 box (run manually before declaring done).
- No regression in existing `tests/` suite.
- README section added under `tribe_backend/api/README.md` documenting env vars, dev run, and curl examples.
- The populated frontend usage report exists at `agent/reports/backend-usage-{YYYY-MM-DD}.md` with the actual cluster URL filled in, and a frontend dev unfamiliar with the project can successfully POST a video and receive a Report by following only the report's instructions.

---

## §7 Dependencies (add to `pyproject.toml`)

```toml
dependencies = [
    # existing...
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "python-multipart>=0.0.9",     # multipart form parsing
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "imageio-ffmpeg>=0.4.9",       # decode mp4 → ndarray (or `av` if already used in tests)
    "soundfile>=0.12",             # audio decode for the JSON-b64 path
    "sse-starlette>=2.0",          # only if Phase D is built
]

[project.optional-dependencies.dev]
# add:
#   "httpx>=0.27"  # async test client
```

Confirm whether `imageio-ffmpeg` or `pyav` is already a transitive dep before adding — reuse what the existing test fixtures use.

---

## §8 Carry-over / Future Work

- **Auth.** v1 trusts the network. Before public exposure: API keys (header) → OAuth/JWT.
- **Rate limiting.** Add `slowapi` once auth identifies callers.
- **Job persistence.** SQLite at `out_dir/jobs.db` for restart-safety; only worth it once a real frontend exists.
- **Batch jobs.** `POST /v1/batch` taking N windows; same worker, but a job becomes a parent with N children. Defer until UX requires it.
- **Multi-GPU.** Replace single asyncio worker with one worker process per GPU device, fronted by a lightweight queue (Redis). Only if we ever have >1 GPU.
- **Cleanup / retention.** Background task to GC `out_dir/{job_id}/` after N days. Trivial but not in v1.
- **Observability.** Structured logs (`structlog`) + Prometheus metrics (`prometheus-fastapi-instrumentator`): job count by status, queue depth, p95 inference duration.
- **Streaming inputs.** Eventually the frontend may want to stream video as it's recorded; that's a transport rewrite (WebSocket + chunked windowing) and out of scope.

---

## Open Questions

1. Does the existing test suite already use `imageio-ffmpeg`, `pyav`, or `decord` for video decode? Reuse rather than add a new dep.
2. Should the JSON-b64 convenience path stay in v1 or be cut to keep the surface minimal? (Recommend keep — it's ~30 lines and unblocks scripted clients/tests.)
3. Does `out_dir` already namespace per-window via `window_id`? If yes, we map `job_id → window_id` 1:1 and reuse the directory. If no, we add a `job_id` directory layer.
4. Confirm `InferenceFailure` is the only domain-specific exception raised by `process_window`. If there are others (parcellation errors, mesh export errors), each needs an explicit error mapping.
5. What is the expected max `media` upload size? Sets the multipart spool threshold and the nginx/ingress config in front of uvicorn.

---

## §9 Frontend Usage Report

After the API is implemented and uvicorn is running on the Slurm box, the project must produce a **standalone, self-contained document** that the frontend team can consume without ever reading this plan, the source code, or the OpenAPI schema. The document lives at:

```
agent/reports/backend-usage-{YYYY-MM-DD}.md
```

It is generated at deploy time (not plan time) by `scripts/generate-usage-report.py`, which probes the live server (`GET /v1/health`, `GET /openapi.json`) and renders a markdown file from a template with all placeholders resolved. The frontend dev opens this one file and can immediately curl, fetch, or wire up an `EventSource`.

### §9.1 Deploy-time URL discovery (Slurm cluster)

**The deployment URL is not `http://localhost:8000`.** This codebase runs inside a Slurm allocation on a compute node whose hostname is only known at run time. The frontend cannot reach the compute node directly without knowing the routing topology (head node? SSH tunnel? reverse proxy?). The report generator must resolve `{CLUSTER_HOST}:{PORT}` to a real, reachable address before writing the report.

**Discovery procedure** (executed by the operator when running `scripts/generate-usage-report.py`, *not* baked into this plan):

1. **Identify the compute node** the API is bound to:
   ```bash
   # Inside the Slurm job that runs uvicorn:
   echo "$SLURM_JOB_NODELIST"                          # e.g., "gpu-node-07"
   scontrol show node "$(hostname)" | grep NodeAddr    # e.g., "NodeAddr=10.32.4.7"
   squeue -u "$USER" --states=R -h -o "%N %B"          # confirm node + batch host
   ```
2. **Determine the access pattern** the frontend will use. Document whichever applies for this cluster — the operator confirms by trying each in order:
   - **Direct from frontend network**: if compute nodes are routable from the frontend's LAN/VPN (rare on shared HPC), use `NodeAddr:PORT` directly.
   - **SSH local-port-forward tunnel**: most common for HPC. Frontend dev runs `ssh -L 8000:gpu-node-07:8000 head.cluster.example` and uses `http://localhost:8000`. The report must include the exact tunnel command with the resolved node name substituted in.
   - **Head/login-node reverse proxy**: if a head node fronts compute (e.g., nginx on `head.cluster.example` proxying `/tribe/ → gpu-node-07:8000`), document the public URL `https://head.cluster.example/tribe/` and confirm CORS + path-prefix are configured.
3. **Fill the placeholder.** The report template carries a literal `{CLUSTER_HOST}:{PORT}` token; the generator replaces it with the resolved address (or, for the tunnel case, with `localhost:{LOCAL_PORT}` plus a sibling "Tunnel setup" subsection containing the exact `ssh -L ...` command).
4. **Sanity-check.** The generator must `curl -fsS http://{CLUSTER_HOST}:{PORT}/v1/health` from the same network the frontend will use, and embed the actual JSON response in the report. If the probe fails, the script aborts non-zero and the report is not written.

This is a **deploy-time step, not a plan-time step.** The plan cannot know `gpu-node-07`; only the operator running the deploy script can.

### §9.2 Report contents

The generated `backend-usage-{date}.md` must contain the sections below, in order. Every section is rendered with concrete, resolved values — no `<TODO>` placeholders may survive into the final file.

#### A. Endpoint base URL

- Resolved `BASE_URL = http://{CLUSTER_HOST}:{PORT}` (or tunnel/proxy variant).
- The exact `curl /v1/health` response captured at generation time, with the inference backend (`fake` vs `gpu`) called out.
- If a tunnel is required: the literal `ssh -L ...` command the frontend dev runs, with the resolved compute node hostname substituted in.

#### B. Request schema (two formats)

For the **frontend audience**, both surfaces of `POST /v1/runs` are documented side-by-side:

1. **Pydantic model** (for backend devs / contract reference) — copied verbatim from `tribe_backend/api/schemas.py::SubmitForm` and `SubmitJsonBody`.
2. **Hand-written JSON example** (for frontend devs) showing every field, type, required/optional flag, and a realistic example value for a 30s clip.

Both transports must be covered:

- **`multipart/form-data`** — table with columns `field | type | required | example | notes`:

  | field      | type   | required | example                          | notes                                          |
  |------------|--------|----------|----------------------------------|------------------------------------------------|
  | `media`    | file   | yes      | `clip.mp4` (30s, H.264, ~12 MB)  | mp4 or webm, ≤200 MB, audio track preferred    |
  | `text`     | str    | yes      | `"The quick brown fox..."`       | UTF-8, no length cap in v1                     |
  | `audio_sr` | int    | no       | `16000`                          | only used if `media` has no audio track        |
  | `video_fps`| int    | no       | `25`                             | default 25                                     |
  | `t_start`  | float  | no       | `0.0`                            | seconds                                        |
  | `window_id`| str    | no       | `"550e8400-e29b-41d4-..."`       | server generates UUID if absent                |

- **`application/json`** with base64 — full example payload for a 30s clip:

  ```json
  {
    "video_b64": "<base64 of ~12MB mp4 — about 16MB encoded>",
    "audio_b64": "<base64 of 30s 16kHz mono wav — about 1.3MB encoded; omit if mp4 has audio>",
    "audio_sr": 16000,
    "video_fps": 25,
    "text": "The quick brown fox jumps over the lazy dog.",
    "t_start": 0.0,
    "window_id": "550e8400-e29b-41d4-a716-446655440000"
  }
  ```

#### C. Response schema

Full JSON shape of the `Report` returned by `GET /v1/runs/{job_id}` once `status="done"`. Every field typed, with a realistic example:

```json
{
  "job_id": "550e8400-...",
  "status": "done",
  "submitted_at": "2026-04-25T14:02:11Z",
  "started_at":   "2026-04-25T14:02:12Z",
  "finished_at":  "2026-04-25T14:07:13Z",
  "report": {
    "window_id": "550e8400-...",
    "t_start": 0.0,
    "t_end":  30.0,
    "top_regions": [
      {
        "name": "L_V1",
        "z_score": 3.42,
        "direction": "positive",
        "description": "Primary visual cortex (left hemisphere) — early visual processing."
      },
      { "name": "R_FFA", "z_score": 2.18, "direction": "positive",
        "description": "Right fusiform face area — face perception." }
    ]
  },
  "artifacts": {
    "mesh_meta":      "/v1/runs/{job_id}/mesh/meta",
    "mesh_colors":    "/v1/runs/{job_id}/mesh/colors",
    "animation_base": "/v1/runs/{job_id}/mesh/animation"
  }
}
```

Mesh artifact URLs are exactly the paths defined in §3 Mesh Artifacts (`mesh_meta`, `mesh_colors`, `animation/{t}`). Document content-types: `application/json` for `mesh_meta`, `application/octet-stream` for `mesh_colors` and per-frame `animation/{t}`.

#### D. Worked curl examples

Each example shows the command and a truncated expected output, so the frontend dev can copy-paste and verify. All URLs use the resolved `BASE_URL`.

1. **Submit a job** (multipart):
   ```bash
   curl -X POST $BASE_URL/v1/runs \
     -F "media=@clip.mp4" \
     -F "text=The quick brown fox..." \
     -F "video_fps=25"
   # → 202
   # { "job_id": "550e8400-...", "status": "queued",
   #   "submitted_at": "2026-04-25T14:02:11Z" }
   ```

2. **Poll status**:
   ```bash
   curl $BASE_URL/v1/runs/550e8400-...
   # → 200 { "status": "running", ... }
   # ... ~5 min later ...
   # → 200 { "status": "done", "report": {...}, "artifacts": {...} }
   ```

3. **Fetch the report** (same as poll once `status="done"`; section repeats the JSON shape from §9.2 C, truncated).

4. **Fetch mesh meta + colors**:
   ```bash
   curl $BASE_URL/v1/runs/550e8400-.../mesh/meta
   # → 200 application/json
   # { "vertices": 20484, "hemispheres": [...], ... }

   curl $BASE_URL/v1/runs/550e8400-.../mesh/colors -o brain_colors.bin
   # → 200 application/octet-stream
   # (binary, ~Nbytes)
   ```

#### E. CORS guidance

- Default dev allowlist: `http://localhost:5173`, `http://localhost:3000`.
- In `TRIBE_INFERENCE=gpu` (production) the allowlist is read from `settings.cors_origins` and is **empty by default** — frontend must request their origin be added.
- How to request a change: open a PR editing `tribe_backend/api/settings.py::Settings.cors_origins` (or set the `TRIBE_CORS_ORIGINS` env var on the Slurm job) — point of contact: backend on-call.
- Preflight: `OPTIONS` is handled by `fastapi.middleware.cors.CORSMiddleware`; the frontend should not need to do anything special beyond a normal browser fetch.

#### F. Latency expectations

- **Real GPU mode (`TRIBE_INFERENCE=gpu`)**: ~5 minutes per 30s clip on a single A6000. Empirically verified: **301s end-to-end** for a 30s clip at the time of report generation. Frontends must use the async path (`POST /v1/runs` → 202 → poll/SSE) — never `?wait=true` against GPU.
- **Dev fake mode (`TRIBE_INFERENCE=fake`)**: ~instant (sub-second). `?wait=true` is acceptable and convenient for frontend integration tests.
- Implication: any UI showing a progress indicator should plan for a 5-minute spinner with a cancellation affordance, and should hit the SSE stream (Phase D) once available rather than polling every second.

#### G. Error catalog

Exact HTTP status codes and JSON error shapes the frontend must handle. The shape is uniform: `{"error": {"code": "<machine-readable>", "message": "<human-readable, no traceback>"}}`.

| Status | `error.code`         | When                                                     | Frontend action                                  |
|--------|----------------------|----------------------------------------------------------|--------------------------------------------------|
| 400    | `decode_error`       | Uploaded media isn't a valid mp4/webm                    | Show "Couldn't read your video, try re-encoding" |
| 422    | `validation_error`   | Pydantic validation (missing `text`, bad `video_fps`)    | Show field-level message from `error.message`    |
| 404    | `job_not_found`      | Unknown `job_id` on GET/DELETE                           | Treat as expired; restart submission             |
| 409    | `job_not_cancelable` | DELETE on a running job                                  | Disable cancel button once status="running"      |
| 503    | `queue_full`         | Submission past `queue_depth`                            | Show "Server busy, retry in a few minutes"       |
| 200    | (job body `status="failed"`, `error.code="inference"`) | Inference raised `InferenceFailure` mid-run | Show "Processing failed: <message>", offer retry |
| 200    | (job body `status="failed"`, `error.code="internal"`)  | Unhandled exception in worker             | Show generic failure + a "report this" link      |

Tracebacks are **never** included; the frontend can rely on `error.message` being safe to display.

### §9.3 Generator script outline

`scripts/generate-usage-report.py` (referenced in Phase E):

1. Read `BASE_URL` from CLI arg or env, defaulting to `http://$(scontrol show node $(hostname) | awk -F= '/NodeAddr/ {print $2}'):${PORT:-8000}`.
2. `GET {BASE_URL}/v1/health` → embed JSON in §9.2 A; abort if non-200.
3. `GET {BASE_URL}/v1/openapi.json` → drive the schema tables in §9.2 B and C from the live spec, so the report cannot drift from the running server.
4. Render the markdown via a Jinja template (or f-string template) checked into `scripts/templates/backend-usage.md.j2`.
5. Write to `agent/reports/backend-usage-$(date +%Y-%m-%d).md`. If a file for today already exists, overwrite (deploys are idempotent).
6. Print the absolute path of the written report to stdout for the operator.

The generator must fail non-zero on any probe error, schema mismatch, or unresolved `{CLUSTER_HOST}` placeholder — a partially-populated report is worse than no report.

---

> **COMMIT PROTOCOL — When implementation is complete, launch a `commit-architect` sub-agent instance (via the Task tool with `subagent_type="commit-architect"`) to analyze your changes and produce clean, atomic Conventional Commits. Do not write commits manually.**
