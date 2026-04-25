"""FastAPI server for Cleo — hackathon demo build.

POST /jobs          multipart upload of a video file → {job_id}
GET  /jobs/{id}     poll for status/result
GET  /              demo UI (single-page, no build step)

Run:
    cleo-serve                          # production-ish (binds 0.0.0.0:8000)
    uvicorn cleo.api:app --reload       # dev with hot-reload

Env knobs:
    CLEO_HOST            bind host          (default 0.0.0.0)
    CLEO_PORT            bind port          (default 8000)
    CLEO_MAX_UPLOAD_MB   upload cap in MB   (default 100)
    CLEO_JOB_TTL_SEC     job retention      (default 3600)
    CLEO_CORS_ORIGINS    comma-sep origins  (default "*")

TRIBE is imported lazily inside the background worker; if the package is
absent or the GPU is unavailable the job falls back to a mock prediction so
the demo works on any machine.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

# ── config (env-driven so the GPU-box engineer can tune without code edits) ──
MAX_UPLOAD_BYTES = int(os.environ.get("CLEO_MAX_UPLOAD_MB", "100")) * 1024 * 1024
JOB_TTL_SECONDS = int(os.environ.get("CLEO_JOB_TTL_SEC", "3600"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("CLEO_CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Cleo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── in-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status": "pending"|"running"|"done"|"error", "result": ... } }
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _sweep_old_jobs() -> None:
    """Drop jobs older than ``JOB_TTL_SECONDS`` (lazy GC, runs on each new job)."""
    now = time.time()
    with _JOBS_LOCK:
        stale = [
            jid for jid, j in _JOBS.items()
            if (j.get("finished_at") or j.get("created_at", now)) < now - JOB_TTL_SECONDS
        ]
        for jid in stale:
            _JOBS.pop(jid, None)


# ── mock prediction (no GPU / no TRIBE) ──────────────────────────────────────

def _mock_forecast(video_path: Path) -> dict[str, Any]:
    """Return plausible-looking synthetic output when TRIBE is unavailable."""
    import random
    from cleo.captioning import caption_clip
    from cleo.feedback import generate_feedback

    rng = random.Random(str(video_path))
    caption = caption_clip(video_path)

    roi_stats = {
        "auditory": {
            "n_vertices": 512,
            "peak": rng.uniform(0.8, 2.5),
            "sustained": rng.uniform(0.5, 2.0),
            "z_sustained": rng.uniform(-0.5, 2.5),
            "z_peak": rng.uniform(0.0, 3.0),
        },
        "visual": {
            "n_vertices": 1024,
            "peak": rng.uniform(0.3, 1.8),
            "sustained": rng.uniform(0.2, 1.2),
            "z_sustained": rng.uniform(-1.0, 1.5),
            "z_peak": rng.uniform(-0.5, 2.0),
        },
        "limbic_adjacent": {
            "n_vertices": 384,
            "peak": rng.uniform(0.1, 1.2),
            "sustained": rng.uniform(0.0, 0.8),
            "z_sustained": rng.uniform(-1.5, 1.0),
            "z_peak": rng.uniform(-1.0, 1.5),
        },
    }
    subcortical_z = {
        "Amygdala":    rng.uniform(-1.0, 2.5),
        "Hippocampus": rng.uniform(-1.0, 2.0),
        "Thalamus":    rng.uniform(-0.5, 2.5),
    }
    ranking = sorted(roi_stats, key=lambda g: roi_stats[g]["z_sustained"], reverse=True)
    feedback = generate_feedback(caption, roi_stats, ranking, subcortical_z=subcortical_z)

    return {
        "video": str(video_path),
        "caption": caption,
        "mock": True,
        "preds_shape": {"cortical": [30, 20484], "subcortical": [30, 8802]},
        "n_segments": 30,
        "roi_stats": roi_stats,
        "subcortical_z": subcortical_z,
        "ranking_by_sustained_z": ranking,
        "feedback": feedback,
    }


# ── background worker ─────────────────────────────────────────────────────────

# Set to "1" to skip the live remote and force the mock path (handy for offline demos).
_FORCE_MOCK = os.environ.get("CLEO_FORCE_MOCK", "").strip() in ("1", "true", "yes")


def _run_job(job_id: str, video_path: Path, tmp_dir: str) -> None:
    def _update(**kw: Any) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id].update(kw)

    def _on_stage(stage: str, info: dict[str, Any]) -> None:
        # Forward pipeline stages so the frontend can show meaningful progress.
        _update(stage=stage, stage_info=info)

    _update(status="running", stage="starting", stage_info={})
    try:
        if _FORCE_MOCK:
            result = _mock_forecast(video_path)
        else:
            try:
                from cleo.pipeline import run_forecast

                result = run_forecast(
                    video_path,
                    include_feedback=True,
                    on_status=_on_stage,
                )
            except Exception as tribe_err:  # noqa: BLE001
                # Remote unreachable or failed → fall back to mock so the demo still works.
                print(
                    f"[cleo] remote TRIBE unavailable ({tribe_err.__class__.__name__}: "
                    f"{tribe_err}); using mock forecast",
                    flush=True,
                )
                _update(stage="mock_fallback", stage_info={"reason": str(tribe_err)})
                result = _mock_forecast(video_path)

        _update(status="done", result=result, finished_at=time.time(), stage="done")
    except Exception as exc:  # noqa: BLE001
        _update(status="error", error=f"{exc.__class__.__name__}: {exc}", stage="error")
    finally:
        try:
            video_path.unlink(missing_ok=True)
            Path(tmp_dir).rmdir()
        except Exception:  # noqa: BLE001
            pass


# ── routes ────────────────────────────────────────────────────────────────────

_VIDEO_EXT_ALLOWLIST = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


@app.post("/jobs")
async def create_job(video: UploadFile = File(...)) -> JSONResponse:
    """Accept a video upload and start processing in the background.

    Validates that the upload is plausibly a video (MIME type or extension)
    and caps it at ``CLEO_MAX_UPLOAD_MB``. Reads at most ``MAX_UPLOAD_BYTES+1``
    bytes so an oversize upload can't fill RAM/disk past the cap.
    """
    # ── validate type ─────────────────────────────────────────────────────────
    ctype = (video.content_type or "").lower()
    suffix = Path(video.filename or "clip.mp4").suffix.lower()
    if not ctype.startswith("video/") and suffix not in _VIDEO_EXT_ALLOWLIST:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported upload (content-type={ctype!r}, suffix={suffix!r}); expected a video",
        )

    # ── read with hard cap ────────────────────────────────────────────────────
    data = await video.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap",
        )
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    # ── persist past the request ──────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="cleo_")
    video_path = Path(tmp_dir) / f"clip{suffix or '.mp4'}"
    video_path.write_bytes(data)

    # ── enqueue ───────────────────────────────────────────────────────────────
    _sweep_old_jobs()
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "pending", "created_at": time.time()}

    thread = threading.Thread(target=_run_job, args=(job_id, video_path, tmp_dir), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(job)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    with _JOBS_LOCK:
        n = len(_JOBS)
    return {"ok": True, "jobs": n, "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    return _HTML


def serve() -> None:
    """`cleo-serve` entry point — runs uvicorn against this module's ``app``."""
    import uvicorn

    host = os.environ.get("CLEO_HOST", "0.0.0.0")
    port = int(os.environ.get("CLEO_PORT", "8000"))
    uvicorn.run("cleo.api:app", host=host, port=port, log_level="info")


# ── embedded demo UI ──────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Cleo — Sensory Load Scout</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f0f13;color:#e8e8f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 16px}
  h1{font-size:2rem;font-weight:700;letter-spacing:-.5px;margin-bottom:4px}
  .sub{color:#888;font-size:.95rem;margin-bottom:40px}
  .card{background:#1a1a24;border:1px solid #2a2a3a;border-radius:16px;padding:32px;width:100%;max-width:620px;margin-bottom:24px}
  .upload-area{border:2px dashed #3a3a5a;border-radius:12px;padding:48px 24px;text-align:center;cursor:pointer;transition:border-color .2s}
  .upload-area:hover,.upload-area.drag{border-color:#7c6af7}
  .upload-area input{display:none}
  .upload-icon{font-size:2.5rem;margin-bottom:12px}
  .upload-hint{color:#888;font-size:.85rem;margin-top:8px}
  .btn{display:inline-block;background:#7c6af7;color:#fff;border:none;border-radius:8px;padding:12px 28px;font-size:1rem;font-weight:600;cursor:pointer;transition:background .15s;width:100%;margin-top:16px}
  .btn:hover{background:#6a58e0}
  .btn:disabled{background:#444;cursor:not-allowed}
  .status{padding:16px;border-radius:10px;font-size:.9rem;margin-top:16px;display:none}
  .status.pending{background:#2a2a1a;border:1px solid #554}
  .status.running{background:#1a2a2a;border:1px solid #458}
  .status.done{background:#1a2a1a;border:1px solid #484}
  .status.error{background:#2a1a1a;border:1px solid #844}
  .spinner{display:inline-block;width:14px;height:14px;border:2px solid #fff3;border-top-color:#7c6af7;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:8px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .result{display:none}
  .score-badge{display:inline-flex;align-items:center;gap:10px;font-size:2.5rem;font-weight:800;margin-bottom:16px}
  .score-num{font-size:1.2rem;color:#aaa;font-weight:400}
  .caption-box{background:#111118;border-radius:8px;padding:16px;font-size:.85rem;color:#aab;line-height:1.6;margin-bottom:16px;font-style:italic}
  .roi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
  .roi-card{background:#111118;border-radius:10px;padding:14px;text-align:center}
  .roi-name{font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  .roi-z{font-size:1.4rem;font-weight:700}
  .roi-z.high{color:#f87}
  .roi-z.mid{color:#fb7}
  .roi-z.low{color:#7c6af7}
  .md-output{background:#111118;border-radius:8px;padding:20px;font-size:.88rem;line-height:1.7;white-space:pre-wrap;color:#ccd}
  .mock-badge{background:#2a2020;border:1px solid #644;color:#f98;font-size:.75rem;padding:4px 10px;border-radius:20px;display:inline-block;margin-bottom:16px}
  .filename{font-weight:600;color:#a9a;margin-top:8px;font-size:.9rem}
</style>
</head>
<body>
<h1>🧠 Cleo</h1>
<p class="sub">Sensory-load scout — powered by TRIBE v2 + Claude</p>

<div class="card">
  <div class="upload-area" id="dropzone">
    <div class="upload-icon">🎬</div>
    <div>Drop a 30-second clip here or <strong>click to browse</strong></div>
    <div class="upload-hint">MP4, MOV, AVI — any format ffmpeg handles</div>
    <input type="file" id="fileInput" accept="video/*"/>
    <div class="filename" id="filename"></div>
  </div>
  <button class="btn" id="submitBtn" disabled>Analyse clip</button>
  <div class="status pending" id="statusBox"></div>
</div>

<div class="card result" id="resultCard">
  <div id="mockBadge" class="mock-badge" style="display:none">⚠️ Mock prediction (no GPU)</div>
  <div class="score-badge" id="scoreBadge"></div>
  <div class="caption-box" id="captionBox"></div>
  <div class="roi-grid" id="roiGrid"></div>
  <div class="md-output" id="mdOutput"></div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const submitBtn = document.getElementById('submitBtn');
const statusBox = document.getElementById('statusBox');
const resultCard = document.getElementById('resultCard');
const filename  = document.getElementById('filename');

let selectedFile = null;
let pollTimer = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  filename.textContent = f.name;
  submitBtn.disabled = false;
}

function showStatus(cls, html) {
  statusBox.className = 'status ' + cls;
  statusBox.innerHTML = html;
  statusBox.style.display = 'block';
}

submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  submitBtn.disabled = true;
  resultCard.style.display = 'none';
  showStatus('pending', 'Uploading…');

  const fd = new FormData();
  fd.append('video', selectedFile);

  let jobId;
  try {
    const r = await fetch('/jobs', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    jobId = data.job_id;
  } catch(e) {
    showStatus('error', '❌ Upload failed: ' + e.message);
    submitBtn.disabled = false;
    return;
  }

  showStatus('running', '<span class="spinner"></span> Extracting audio, sampling frames, running caption…');
  poll(jobId, 0);
});

const MESSAGES = [
  'Extracting audio, sampling frames…',
  'Running Claude vision caption…',
  'Running TRIBE v2 brain-encoding inference…',
  'Aggregating Destrieux ROIs…',
  'Generating caregiver brief…',
];

function poll(jobId, tick) {
  pollTimer = setTimeout(async () => {
    let data;
    try {
      const r = await fetch('/jobs/' + jobId);
      data = await r.json();
    } catch(e) {
      showStatus('error', '❌ Poll error: ' + e.message);
      submitBtn.disabled = false;
      return;
    }

    if (data.status === 'done') {
      showStatus('done', '✅ Done!');
      renderResult(data.result);
      submitBtn.disabled = false;
    } else if (data.status === 'error') {
      showStatus('error', '❌ ' + (data.error || 'Unknown error'));
      submitBtn.disabled = false;
    } else {
      const msg = MESSAGES[Math.min(tick, MESSAGES.length - 1)];
      showStatus('running', '<span class="spinner"></span>' + msg);
      poll(jobId, tick + 1);
    }
  }, 2500);
}

function zColor(z) {
  if (z > 1.0) return 'high';
  if (z > 0.0) return 'mid';
  return 'low';
}

function renderResult(r) {
  resultCard.style.display = 'block';

  document.getElementById('mockBadge').style.display = r.mock ? 'inline-block' : 'none';

  const fb = r.feedback || {};
  const icon = fb.icon || '';
  const score = fb.score ?? '?';
  document.getElementById('scoreBadge').innerHTML =
    icon + ' <span>' + score + ' / 10</span> <span class="score-num">sensory load</span>';

  document.getElementById('captionBox').textContent = r.caption || '';

  const roi = r.roi_stats || {};
  const ranking = r.ranking_by_sustained_z || Object.keys(roi);
  const labels = { auditory:'Auditory', visual:'Visual', limbic_adjacent:'Limbic-adj.' };
  document.getElementById('roiGrid').innerHTML = ranking.map(g => {
    const z = (roi[g]?.z_sustained ?? 0).toFixed(2);
    const cls = zColor(roi[g]?.z_sustained ?? 0);
    return '<div class="roi-card"><div class="roi-name">' + (labels[g]||g) +
           '</div><div class="roi-z ' + cls + '">' + (z >= 0 ? '+' : '') + z + '</div></div>';
  }).join('');

  document.getElementById('mdOutput').textContent = fb.markdown || JSON.stringify(r, null, 2);
  resultCard.scrollIntoView({ behavior: 'smooth' });
}
</script>
</body>
</html>
"""
