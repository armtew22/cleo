# TRIBE v2 Backend — Frontend Usage Report

> **Status:** Optimistic preview (Phase E live-smoke is still running). Schemas / curl / error catalog are derived from the implemented code on `main` at commit `8f4371a` (Phases A+B+C merged) and from the FastAPI plan §9. The endpoint URL has been resolved against the live Slurm host. If Phase E uncovers a wiring bug, the only thing that will change is the host:port — the schemas and flow are stable.

---

## 1. Endpoint

**Base URL:**

```
http://ellis-compute-02.cs.cornell.edu:8000
```

- **Host:** `ellis-compute-02.cs.cornell.edu` (`128.84.97.109`) — the Cornell CS Slurm compute node where the backend process runs.
- **Port:** `8000` (uvicorn default; overridable via `PORT` env var when starting the server).
- **Protocol:** HTTP (TLS termination is **not** in v1 — talk to the operator if a public-facing https URL is needed; typical pattern is an SSH tunnel: `ssh -L 8000:ellis-compute-02.cs.cornell.edu:8000 <login-node>`).
- **Health check:** `GET /v1/health` returns `{"status":"ok","inference":"gpu"|"fake","out_dir":"..."}` — confirm `inference == "gpu"` before relying on real Glasser results.

> **Reachability note for the frontend dev:** if your laptop can't reach `ellis-compute-02.cs.cornell.edu:8000` directly (likely — Cornell CS compute nodes are usually behind the campus firewall), use:
> ```
> ssh -L 8000:ellis-compute-02.cs.cornell.edu:8000 <your-cornell-cs-login>
> ```
> Then point the frontend at `http://localhost:8000`.

---

## 2. The user flow in one paragraph

POST a 30-second `.mp4` plus a one-sentence text caption to `/v1/runs`. You get back a `job_id` immediately (HTTP 202). Poll `GET /v1/runs/{job_id}` every few seconds until `status == "done"` (~5 minutes on the GPU). The done response carries the full **Glasser parcellation report** (ranked active brain regions with z-scores) and a `mesh` URL pointing at the 3D-mesh artifact bundle.

---

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET    | `/v1/health` | Liveness probe + which inference backend is loaded |
| POST   | `/v1/runs` | Submit a video → returns a `job_id` |
| GET    | `/v1/runs/{job_id}` | Poll job status; carries the report when done |
| DELETE | `/v1/runs/{job_id}` | Cancel a queued job (queued → cancelled; running can't be cancelled in v1) |
| GET    | `/v1/runs/{job_id}/mesh` | Manifest of all mesh artifact URLs for this job |
| GET    | `/v1/runs/{job_id}/mesh/meta` | `brain_meta.json` — vertex_count, face_count, format version |
| GET    | `/v1/runs/{job_id}/mesh/colors` | Per-vertex RGBA colors as raw bytes (`vertex_count * 4` = 81 936 bytes) |
| GET    | `/v1/runs/{job_id}/mesh/vertices` | Static fsaverage5 vertex coordinates as raw bytes (`vertex_count * 12` = 245 808 bytes) |
| GET    | `/v1/runs/{job_id}/mesh/faces` | Static fsaverage5 face indices as raw bytes (`face_count * 12` = 491 520 bytes) |

Live OpenAPI docs:

- Swagger UI: `http://ellis-compute-02.cs.cornell.edu:8000/docs`
- Raw spec: `http://ellis-compute-02.cs.cornell.edu:8000/openapi.json`

---

## 4. Submit a job

### Request — `POST /v1/runs`

**Content-Type:** `multipart/form-data` (only — JSON+base64 was deliberately dropped to keep the surface tight).

| Field   | Type   | Required | Description |
|---|---|---|---|
| `media` | file (mp4) | yes | The 30-second video. Audio is extracted from the file's audio track via ffmpeg server-side. Other container formats are not supported in v1. |
| `text`  | string | yes | A one-sentence description of what the clip shows. Fed to the model's text encoder. |
| `wait`  | bool   | no (default `false`) | If `true`, the request blocks until the report is ready and returns 200 + the Report inline. **Disabled when `TRIBE_INFERENCE=gpu`** — a 5-minute synchronous request would time out most clients. Use the async path. |
| `z_threshold` | float (≥ 0) | no (default 1.5) | Z-score cut-off for the ranked `top_regions[]` in the report. The default of 1.5 is calibrated for strong sustained activations; real stimuli often produce weaker per-parcel means and return an empty list. Lower this (e.g. `0.1`) to see the long tail. Higher values (e.g. `3.0`) only surface the most prominent regions. |

### Response — `202 Accepted`

```json
{
  "job_id": "57f289d727464e2cb2fa61f2b0922b87",
  "status": "queued"
}
```

### curl

```bash
curl -sS -X POST \
  -F "media=@traffic_30s.mp4" \
  -F "text=A busy traffic intersection at dusk with car horns and pedestrians." \
  -F "z_threshold=0.1" \
  http://ellis-compute-02.cs.cornell.edu:8000/v1/runs
```

---

## 5. Poll for the report

### Request — `GET /v1/runs/{job_id}`

No body. Returns the job's current state.

### Response — `200 OK`

While running:

```json
{
  "job_id": "57f289d727464e2cb2fa61f2b0922b87",
  "status": "running",
  "submitted_at": "2026-04-25T20:04:56.446297Z",
  "started_at":   "2026-04-25T20:04:56.446444Z",
  "finished_at":  null,
  "report": null,
  "error":  null,
  "mesh":   null
}
```

When `status == "done"`:

```json
{
  "job_id": "57f289d727464e2cb2fa61f2b0922b87",
  "status": "done",
  "submitted_at": "2026-04-25T20:04:56.446297Z",
  "started_at":   "2026-04-25T20:04:56.446444Z",
  "finished_at":  "2026-04-25T20:09:57.812031Z",
  "report": {
    "window_id":   "2026-04-25T20:04:56Z-303eff",
    "method":      "window_mean",
    "z_threshold": 1.5,
    "text":        "Window summary: strong activation in R_FFC_ROI (z=+3.42); moderate activation in L_V1_ROI (z=+2.10); ...",
    "top_regions": [
      { "name": "R_FFC_ROI",  "z_score":  3.42, "direction": "activation",   "description": "Right fusiform face complex" },
      { "name": "L_V1_ROI",   "z_score":  2.10, "direction": "activation",   "description": "Left primary visual cortex" },
      { "name": "L_STSva_ROI","z_score":  1.84, "direction": "activation",   "description": "..." }
    ]
  },
  "error": null,
  "mesh":  "/v1/runs/57f289d727464e2cb2fa61f2b0922b87/mesh"
}
```

When `status == "failed"`:

```json
{
  "job_id": "...",
  "status": "failed",
  "report": null,
  "error":  { "code": "INFERENCE_FAILURE", "message": "CUDA OOM on window ..." }
}
```

### curl

```bash
JOB=57f289d727464e2cb2fa61f2b0922b87
curl -sS http://ellis-compute-02.cs.cornell.edu:8000/v1/runs/$JOB | jq .
```

Recommended frontend polling cadence: every 5-10 seconds (a forward pass takes ~5 min; faster polling is wasted load).

---

## 6. Fetch the 3D mesh artifacts

### Manifest

```bash
curl -sS http://ellis-compute-02.cs.cornell.edu:8000/v1/runs/$JOB/mesh | jq .
```

```json
{
  "job_id": "57f289d727464e2cb2fa61f2b0922b87",
  "artifacts": {
    "meta":     "/v1/runs/57f289d727464e2cb2fa61f2b0922b87/mesh/meta",
    "colors":   "/v1/runs/57f289d727464e2cb2fa61f2b0922b87/mesh/colors",
    "vertices": "/v1/runs/57f289d727464e2cb2fa61f2b0922b87/mesh/vertices",
    "faces":    "/v1/runs/57f289d727464e2cb2fa61f2b0922b87/mesh/faces"
  }
}
```

### Binary artifacts

| File | Bytes | Layout |
|---|---|---|
| `mesh/colors`   | 81 936  | `uint8 RGBA` × 20 484 vertices — per-vertex color (per window) |
| `mesh/vertices` | 245 808 | `float32` × 3 × 20 484 vertices — fsaverage5 coords (static; cache forever) |
| `mesh/faces`    | 491 520 | `int32`   × 3 × 40 960 triangles — fsaverage5 face indices (static; cache forever) |
| `mesh/meta`     | small JSON | `{ "vertex_count": 20484, "face_count": 40960, "format": "tribe_brain_mesh_v1", ... }` |

The binary endpoints set `Cache-Control: public, max-age=31536000, immutable` (vertices/faces are pure functions of fsaverage5 and never change; colors are tied to a specific `window_id` and are also immutable once written). The JSON endpoints set `Cache-Control: no-store`.

### Loading colors in the browser

```js
const res = await fetch(`/v1/runs/${jobId}/mesh/colors`);
const buf = await res.arrayBuffer();
const colors = new Uint8Array(buf);     // length 81936 = 20484 * 4
// vertex i RGBA = colors.slice(i*4, i*4+4)
```

---

## 7. Cancel a queued job

```bash
curl -sS -X DELETE http://ellis-compute-02.cs.cornell.edu:8000/v1/runs/$JOB -i
```

- `204 No Content` if the job was queued and got cancelled.
- `409 Conflict` if the job is already running, done, failed, or cancelled.
- `404 Not Found` if `job_id` is unknown.

---

## 8. Error catalog

All errors return JSON with this envelope:

```json
{ "error": { "code": "ERROR_CODE", "message": "human-readable explanation" } }
```

| HTTP | code | When |
|---|---|---|
| 400 | `WAIT_DISABLED_GPU` | `wait=true` was sent while the server is in `TRIBE_INFERENCE=gpu` mode. Use the async path. |
| 404 | `JOB_NOT_FOUND` | Unknown `job_id` (or path-traversal attempt). |
| 409 | `JOB_NOT_READY` | You requested mesh artifacts on a job that is `queued` or `running`. |
| 409 | `JOB_NOT_CANCELLABLE` | DELETE on a job that is already running/done/failed/cancelled. |
| 410 | `JOB_FAILED` | You requested mesh artifacts on a job whose status is `failed` or `cancelled`. Artifacts will never exist. |
| 422 | (Pydantic) | Missing / wrong-shape fields on `POST /v1/runs`. |
| 500 | `INFERENCE_FAILURE` | The GPU adapter raised (OOM, NaN values, ffmpeg muxing failed, etc.). The job's `status` becomes `failed` and the same code appears in `error.code`. |
| 500 | `INFERENCE_ARTIFACTS_MISSING` | Job is `done` but the on-disk mesh files are gone (shouldn't happen; indicates an `OUT_DIR` race). |
| 503 | `QUEUE_FULL` | The job queue is at capacity (default `maxsize=4`). Response includes `Retry-After: <seconds>`. Wait and resubmit. |

---

## 9. CORS

Default allowlist: **`*`** (any origin, dev mode). Configurable via the `CORS_ALLOW_ORIGINS` env var on the server side (comma-separated list of origins).

Preflight `OPTIONS` requests are handled automatically; the frontend doesn't need to do anything special.

If you need a stricter allowlist for production (e.g. only `https://your-frontend.example.com`), email/ping the operator — it's a one-line server-side env-var change, no code redeploy needed.

---

## 10. Latency expectations

| Mode | Per-request latency | Use |
|---|---|---|
| `TRIBE_INFERENCE=fake` (dev) | ~50 ms | Local dev / contract tests. Returns a synthetic Report with empty `top_regions[]`. |
| `TRIBE_INFERENCE=gpu` (production) | **~5 minutes (301 s empirical on RTX A6000)** | Real Glasser report with real activations like `L_FFC_ROI`, `R_V1_ROI`. |

Model load on first server start: ~30-60 s (one-time per process). Subsequent requests skip this.

---

## 11. End-to-end frontend reference (JavaScript / fetch)

```js
async function runTribe(file, text) {
  const base = "http://ellis-compute-02.cs.cornell.edu:8000";

  // 1. Submit
  const fd = new FormData();
  fd.append("media", file);                 // <input type="file"> -> File object
  fd.append("text",  text);
  const submit = await fetch(`${base}/v1/runs`, { method: "POST", body: fd });
  if (submit.status !== 202) throw new Error(`submit failed: ${submit.status}`);
  const { job_id } = await submit.json();

  // 2. Poll
  let job;
  while (true) {
    await new Promise(r => setTimeout(r, 5000));
    job = await fetch(`${base}/v1/runs/${job_id}`).then(r => r.json());
    if (job.status === "done")   break;
    if (job.status === "failed") throw new Error(job.error.message);
    if (job.status === "cancelled") throw new Error("job cancelled");
  }

  // 3. Use the report (regions + narrative)
  console.log(job.report.text);
  for (const r of job.report.top_regions) {
    console.log(`${r.name}: z=${r.z_score.toFixed(2)} (${r.direction})`);
  }

  // 4. Optionally fetch mesh artifacts for 3D rendering
  const manifest = await fetch(`${base}${job.mesh}`).then(r => r.json());
  const [colors, vertices, faces] = await Promise.all([
    fetch(`${base}${manifest.artifacts.colors}`).then(r => r.arrayBuffer()),
    fetch(`${base}${manifest.artifacts.vertices}`).then(r => r.arrayBuffer()),
    fetch(`${base}${manifest.artifacts.faces}`).then(r => r.arrayBuffer()),
  ]);
  return { job, mesh: { colors, vertices, faces } };
}
```

---

## 12. Open caveats (read before integrating)

- **Live GPU smoke test is still running at the time of writing.** The schemas above are derived from code on `main` (Phases A+B+C merged) and from the plan; the host:port is real. If Phase E uncovers a wiring bug, only the URL or one error code may shift — the request/response JSON shapes are locked.
- **Single GPU = single worker.** The server has one in-flight job at a time; the rest queue. Backpressure is the `503 QUEUE_FULL` response — your client should retry with backoff.
- **One model in VRAM.** Don't try to spin up a second uvicorn process on the same GPU; it will OOM at model load.
- **No auth in v1.** Anyone with network access to `ellis-compute-02.cs.cornell.edu:8000` can submit jobs.
- **No persistence in v1.** Job state lives in memory; a server restart loses pending jobs (but on-disk mesh artifacts under `OUT_DIR/{window_id}/` survive).
- **Subcortical predictions are zero-filled.** The upstream `tribev2` model only exposes the cortical head in `predict()`; the API serves `(T, 8802)` zeros for subcortical until the upstream model wires up the subcortical head. Your frontend should ignore subcortical for now (and there is no separate subcortical endpoint anyway — only cortical in the report and mesh).
