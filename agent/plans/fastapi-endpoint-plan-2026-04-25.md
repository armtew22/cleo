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

> **COMMIT PROTOCOL — When implementation is complete, launch a `commit-architect` sub-agent instance (via the Task tool with `subagent_type="commit-architect"`) to analyze your changes and produce clean, atomic Conventional Commits. Do not write commits manually.**
