# TDD Pipeline Plan — TRIBE v2 Backend Modules

**Scope:** Architect two modules — `GlasserParcellationUnit` (interpretation) and `BrainMeshExporter` (3D parser) — as modular backend components that consume the TRIBE v2 inference contract. Built test-first so that when the GPU inference unit comes online, real `(T, 20484)` + `(T, 8802)` tensors slot in with zero adapter code.

Source plans:
- `agent/glasser-parcellation-plan.md`
- `agent/brain-3d-parser.md`

---

## 0. Reference Fixture: real TRIBE v2 sample output

A verbatim sample of the model's output is committed at `agent/tribev2_sample_output.txt`:

- **Shape:** `(31, 20484)` float32 cortical predictions.
- **Window:** 30-second multimodal stimulus ("condition A", text+audio+video).
- **Segments:** 31 segments of `duration=1.0s` at 1Hz (`t=0..30` inclusive) — exactly matches `TribeOutput.EXPECTED_T_PER_30S_WINDOW`.
- **Format:** header lines + a verbatim NumPy `repr` of the array.

Both `parcellation/` and `mesh/` MUST verify they handle this real fixture end-to-end. A shared loader `tests/fixtures/load_sample.py::load_tribe_sample()` parses the txt file once into a `TribeOutput` (synthesizing zero-filled subcortical of shape `(31, 8802)` since the sample only carries the cortical head) and is used by both compartments. This guards against drift between synthetic fixtures and the real model's distribution (range, sign, sparsity).

---

## 1. Architectural Contract (the seam)

The whole pipeline hinges on one frozen data contract between TRIBE inference and everything downstream. Define it once, test against it everywhere.

```python
# tribe_backend/contracts.py
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class StimulusWindow:
    """A 30-second multimodal window emitted by the live feed poller."""
    window_id: str             # monotonic, e.g. "2026-04-25T14:00:00Z"
    t_start: float             # epoch seconds
    duration_s: float          # nominally 30.0
    video: np.ndarray          # (F, H, W, 3) uint8 — F frames at known fps
    video_fps: float
    audio: np.ndarray          # (S,) or (S, C) float32 — mono/stereo PCM
    audio_sr: int              # sample rate, Hz
    text: str                  # caption / description of what is shown

    def __post_init__(self):
        assert self.video.ndim == 4 and self.video.shape[-1] == 3
        assert self.audio.ndim in (1, 2)
        assert 25.0 <= self.duration_s <= 35.0   # tolerance around 30s

@dataclass(frozen=True)
class TribeOutput:
    cortical: np.ndarray       # (T, 20484) float32, z-scored BOLD — T==31 for a 30s window (t=0..30 inclusive at 1Hz)
    subcortical: np.ndarray    # (T,  8802) float32, z-scored BOLD
    fps: float = 1.0
    surface: str = "fsaverage5"
    subcortical_atlas: str = "harvard_oxford_2mm"
    window_id: str | None = None   # ties output back to the StimulusWindow

    EXPECTED_T_PER_30S_WINDOW = 31  # 1Hz inclusive of t=0 and t=30

    def __post_init__(self):
        assert self.cortical.ndim == 2 and self.cortical.shape[1] == 20484
        assert self.subcortical.ndim == 2 and self.subcortical.shape[1] == 8802
        assert self.cortical.shape[0] == self.subcortical.shape[0]

class TribeInference(Protocol):
    def __call__(self, window: StimulusWindow) -> TribeOutput: ...

class LiveFeedPoller(Protocol):
    """Yields 30s windows from a live source (camera, RTSP, file, mock)."""
    def __iter__(self) -> "LiveFeedPoller": ...
    def __next__(self) -> StimulusWindow: ...
```

Every module accepts/returns these types. **No module pokes at raw tensors with assumed shapes** — they go through the contract.

---

## 2. Module Layout — strict compartmentalization

Every module is a self-contained package: its own `__init__.py` with a narrow public API, its own tests, its own fixtures, no sibling imports. Cross-module communication happens **only** through `contracts.py` types or the `ControlUnit` orchestrator. Enforced by `import-linter` in CI.

```
tribe_backend/
├── contracts.py              # TribeOutput, StimulusWindow, InferenceRequest — the ONLY shared module
│
├── control/                  # ── COMPARTMENT: orchestrator (top of backend)
│   ├── __init__.py           #    public API: ControlUnit
│   ├── unit.py               #    poll loop, windowing, dispatch to inference, fan-out
│   ├── poller.py             #    LiveFeedPoller protocol + concrete adapters
│   ├── window_buffer.py      #    rolls 30s (video, audio, text) windows
│   └── dispatcher.py         #    async send to inference, receive TribeOutput, fan out
│
├── inference/                # ── COMPARTMENT: GPU model host
│   ├── __init__.py           #    public API: TribeInference protocol, GpuTribeInference
│   ├── protocol.py           #    TribeInference Protocol (the seam)
│   ├── stub.py               #    FakeTribeInference for tests
│   ├── gpu_runner.py         #    real GPU host (lands later) — loads tribev2 weights
│   └── preprocessing.py      #    StimulusWindow → model input tensors (per-modality)
│
├── parcellation/             # ── COMPARTMENT: ROI interpretation
│   ├── __init__.py           #    public API: GlasserParcellationUnit
│   ├── unit.py
│   ├── atlas.py              #    Glasser + Harvard-Oxford loaders (cached)
│   └── descriptions.json
│
├── mesh/                     # ── COMPARTMENT: 3D parser
│   ├── __init__.py           #    public API: BrainMeshExporter
│   ├── exporter.py
│   ├── colormap.py
│   └── geometry.py
│
└── (no top-level pipeline.py — control/ IS the pipeline)

tests/
├── conftest.py               # only the shared TribeOutput / StimulusWindow factories
├── test_contracts.py
├── control/        test_control_unit.py, test_poller.py, test_window_buffer.py
├── inference/      test_stub.py, test_preprocessing.py, test_gpu_runner.py (gpu mark)
├── parcellation/   test_parcellation_unit.py
├── mesh/           test_mesh_exporter.py
└── golden/                   # small fixture tensors + expected outputs
```

**Compartment rules (CI-enforced):**
- `parcellation/` and `mesh/` may import **only** `contracts` + stdlib + third-party. Not each other, not `control/`, not `inference/`.
- `inference/` imports **only** `contracts`. It does not know about parcellation or mesh.
- `control/` is the **only** module allowed to import all four (`inference`, `parcellation`, `mesh`, `contracts`). It is the composition root.
- `contracts.py` imports nothing from this package.

---

## 3. TDD Pipeline — Phases

Each phase is **red → green → refactor**. No code is written without a failing test.

### Phase 0 — Contract (½ day)
**Tests first** (`test_contracts.py`):
- `TribeOutput` rejects wrong shapes, mismatched T, non-2D arrays.
- Round-trips to/from a `.npz` on disk (so test fixtures are cheap).
- A `make_synthetic(T, seed)` factory deterministically produces a valid `TribeOutput` with planted "hot spots" at known vertex/voxel indices.

**Then implement** `contracts.py` + the synthetic factory in `conftest.py`.

This is the keystone — every later test reuses `make_synthetic`.

### Phase 1 — GlasserParcellationUnit (2–3 days) — **runs in parallel on worktree `wt-parcellation`**
Build it bottom-up so each layer is green before the next is written.

> **Parallelism contract:** Phase 1 executes on a dedicated git worktree (`../cleo-wt-parcellation`, branch `feat/parcellation-unit`). It touches **only** `tribe_backend/contracts.py` (read-only after Phase 0), `tribe_backend/parcellation/**`, and `tests/test_parcellation_unit.py` + parcellation fixtures. No edits to `mesh/` — guaranteed by import-linter and a CI path-filter check on the PR.

> **Window-level aggregation requirement:** the unit's primary report API operates on a full 30-second window — i.e. it averages across **all 31 timesteps** of a `(31, 20484)` cortical tensor (and `(31, 8802)` subcortical) before ranking and narrating. This is the default and what `ControlUnit` calls per window. Per-frame and peak-frame variants remain available but are secondary. The "31" comes from 1Hz fMRI sampling inclusive of `t=0` and `t=30` over a 30s window; see `TribeOutput.EXPECTED_T_PER_30S_WINDOW`.

1. **Atlas loader** (`atlas.py`)
   - Test: loading Glasser annot produces `(20484,) int` with values in `[0, 360]`, both hemispheres covered, no vertex unlabeled where it shouldn't be.
   - Test: Harvard-Oxford mask produces `(8802,) int` with values in `[0, 8]`.
   - Test: results are cached (second call doesn't re-read from disk — patch `nibabel.freesurfer.read_annot` with a counter).

2. **`parcellate_cortical` / `parcellate_subcortical`**
   - Test: plant a value of `5.0` at exactly the vertices belonging to parcel `FFC`, zero elsewhere → `result["FFC"]` time series equals `5.0`, all others `0.0`.
   - Test: shape is `(T,)` per region; dict has exactly 360 cortical + 8 subcortical keys.
   - Property test (hypothesis): mean-of-parcels equals weighted mean of vertices.

3. **`aggregate` / `rank_and_threshold`**
   - Test each method (`window_mean` [DEFAULT], `peak`, `peak_window`) on a hand-rolled time series with known answer.
   - **Window-mean specifically:** given a `(31, 20484)` input where vertex `v` has values `[0,1,2,...,30]`, `aggregate(method="window_mean")[v] == 15.0`. Same shape check for subcortical `(31, 8802)`.
   - Shape contract test: passing a tensor with `T != 31` triggers a `WindowSizeWarning` (not an error — partial windows from startup/shutdown are tolerated, but the default path expects 31).
   - Test ranking: top-k respects `z_threshold` and is sorted descending.

4. **`generate_report` — primary 30s-window API**
   - Signature: `generate_report(output: TribeOutput, *, method="window_mean", top_k=10, z_threshold=1.5) -> Report`. The default averages across all 31 timesteps before ranking; this is what `ControlUnit` calls.
   - Test: a `(31, 20484)` synthetic input where the Amygdala vertices alternate `[+4, 0, +4, 0, ...]` (mean ≈ +2) → report contains `"Amygdala"` with the `"moderate"` template, NOT `"strong"`. This proves the window mean (not the peak) is what drives the narrative.
   - Test: a `(31, 20484)` input where Amygdala spikes only at `t=15` to `+10` and is `0` elsewhere → window-mean report is below threshold (Amygdala does NOT appear), but `method="peak"` does surface it. This proves window averaging genuinely smooths transients.
   - Test: empty report when nothing crosses threshold (don't crash, return a "no significant activations" sentinel).
   - Test: `report.window_id == output.window_id` for traceability back to the `StimulusWindow`.
   - Snapshot test: full window-mean report against a stored golden string for one fixed seed of a `(31, 20484)` fixture.

5. **Real-sample acceptance test (`test_parcellation_real_sample.py`)** — required before Phase 1 PR merges.
   - Load `agent/tribev2_sample_output.txt` via `load_tribe_sample()` → `TribeOutput` with `cortical.shape == (31, 20484)`.
   - `unit.generate_report(output)` (default `method="window_mean"`) returns a non-empty `Report` with `report.window_id == output.window_id`.
   - The report's top region's window-mean z-score equals `output.cortical.mean(axis=0)[parcel_mask].mean()` for the parcel named in the report (proves correct vertex→parcel mapping on real data).
   - All 360 cortical parcel time series have shape `(31,)` — confirms no axis confusion when feeding the canonical 30s window.
   - Report serializes to JSON and round-trips without loss.

6. **Validation harness** (deferred but stubbed)
   - Skipped tests (`@pytest.mark.skip(reason="needs IBC localizers")`) for the paper's face/place/body contrasts. These light up once real inference is wired in — they're the acceptance gate, not blockers for module completion.

### Phase 2 — BrainMeshExporter (2–3 days) — **runs in parallel on worktree `wt-mesh`**
Same bottom-up shape.

> **Parallelism contract:** Phase 2 executes on a dedicated git worktree (`../cleo-wt-mesh`, branch `feat/mesh-exporter`). It touches **only** `tribe_backend/contracts.py` (read-only after Phase 0), `tribe_backend/mesh/**`, and `tests/test_mesh_exporter.py` + mesh fixtures. No edits to `parcellation/`.

1. **`geometry.py`**
   - Test: `load_fsaverage5(surface_type)` returns `coords (20484, 3) float32` and `faces (~40960, 3) int32`; combined LH+RH face indices are correctly offset.
   - Test: `_compute_vertex_normals` produces unit-length vectors and matches a known normal on a flat triangle.

2. **`colormap.py` — `activation_to_rgba`**
   - Test: clipping at `vmin/vmax` (values past range get the endpoint colors).
   - Test: zero activation maps to the colormap midpoint for `RdBu_r`.
   - Test: sulcal darkening reduces RGB but never alpha; output is `uint8` in `[0, 255]`.

3. **`aggregate_temporal`**
   - Test each method on a `(T, 20484)` synthetic input with a planted peak frame.

4. **Export formats** — one test per format, all consuming the same fixture:
   - **JSON**: round-trip (write → reload) preserves vertex count, face count, color bytes; schema has `format: "tribe_brain_mesh_v1"`.
   - **Binary**: byte-exact sizes match meta (`vertex_count * 12` for vertices, `* 4` for colors); reload as numpy from `.bin` recovers original arrays.
   - **GLB**: trimesh can re-load the file and vertex count matches. (Marked `slow`, optional dep.)
   - **Animation bundle**: T color files written, each is `vertex_count * 4` bytes, geometry written exactly once.

5. **Real-sample acceptance test (`test_mesh_real_sample.py`)** — required before Phase 2 PR merges.
   - Load `agent/tribev2_sample_output.txt` → `TribeOutput` with `cortical.shape == (31, 20484)`.
   - `exporter.export_binary(output, out_dir=tmp)` produces all required files; `brain_meta.json` reports `vertex_count == 20484` and `bytes_per_color == 4`.
   - `brain_colors.bin` size equals `20484 * 4` bytes — proves the fsaverage5 vertex count lines up with TRIBE's real cortical width.
   - `aggregate_temporal(output.cortical, method="mean")` returns shape `(20484,)` and its values match `output.cortical.mean(axis=0)` exactly (no off-by-one on the 31-axis).
   - `export_animation_bundle(output.cortical, out_dir=tmp)` writes exactly 31 per-frame color files (`brain_colors_t0000.bin` … `brain_colors_t0030.bin`), each `20484 * 4` bytes.
   - GLB export round-trips through trimesh and the reloaded mesh has 20484 vertices.

6. **Cross-module invariant test** (lives in `test_pipeline.py`):
   - For the same `TribeOutput`, the parcel that `GlasserParcellationUnit` ranks #1 must also be the spatial region with the highest mean color intensity in `BrainMeshExporter`'s output. This catches drift between the two modules' interpretation of the vertex ordering.

### Phase 3 — ControlUnit (top of backend) (1.5 days) — worktree `wt-control`

**User-facing contract:** the `ControlUnit` is the single entry point a user (or upstream service) calls with `(video, audio, text)` for a 30s window, and gets back a Glasser-parcellation report (plus the mesh artifacts). Everything below — windowing, GPU dispatch, parcellation, mesh export — is hidden behind one call.

```python
# The user-facing API
control = ControlUnit.default(weights_path=..., device="cuda:0")

# Option A — one-shot: user hands in a single 30s window, gets a report back
report: Report = control.process_window(
    video=np.ndarray,    # (F, H, W, 3) uint8
    audio=np.ndarray,    # (S,) or (S, C) float32
    audio_sr=16000,
    text="A busy traffic intersection at dusk with car horns and pedestrians.",
)
print(report.text)        # qualitative narrative from Glasser parser
report.top_regions        # ranked list with z-scores
report.mesh_dir           # path to 3D artifacts (optional)

# Option B — live: user attaches a poller, ControlUnit emits reports as windows arrive
for bundle in control.run_stream(poller=MyLiveFeedPoller()):
    print(bundle.report.text)
```

Internally, both paths funnel through the same loop — they only differ in whether the `StimulusWindow` is constructed once (A) or yielded by a poller (B):

```
LiveFeedPoller ──30s window──▶ ControlUnit ──StimulusWindow──▶ TribeInference (GPU)
                                    │                                │
                                    │◀──────── TribeOutput ◀─────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                 GlasserParcellationUnit   BrainMeshExporter
                          │                   │
                          ▼                   ▼
                     report.json         mesh artifacts
```

```python
# tribe_backend/control/unit.py
class ControlUnit:
    def __init__(self,
                 poller: LiveFeedPoller,
                 inference: TribeInference,
                 parcellation: GlasserParcellationUnit,
                 mesh: BrainMeshExporter,
                 sink: ArtifactSink):
        ...

    def run_forever(self): ...      # blocking poll loop
    def run_once(self) -> Bundle:   # consume one window, return artifacts
        window = next(self._poller)
        return self._process(window)

    def process_window(self, *, video, audio, audio_sr, text,
                       video_fps=30.0, t_start=None) -> Report:
        """One-shot user API: hand in raw media, get a Glasser report back."""
        window = StimulusWindow(
            window_id=_make_id(t_start),
            t_start=t_start or time.time(),
            duration_s=len(audio) / audio_sr,
            video=video, video_fps=video_fps,
            audio=audio, audio_sr=audio_sr,
            text=text,
        )
        return self._process(window).report

    def run_stream(self, poller: LiveFeedPoller) -> Iterator[Bundle]:
        for window in poller:
            yield self._process(window)

    def _process(self, window: StimulusWindow) -> Bundle:
        out = self._inference(window)                          # GPU call
        report = self._parcellation.generate_report(out)       # Glasser report
        mesh_dir = self._mesh.export_binary(out, out_dir=self._sink.path_for(window))
        return Bundle(window_id=window.window_id, report=report, mesh_dir=mesh_dir)
```

**Tests (`test_control_unit.py`):**
- `MockPoller` yielding 3 fixture windows + `FakeTribeInference` → `run_once` called 3 times produces 3 bundles, each with the correct `window_id`.
- **Backpressure:** if inference takes longer than 30s, the poller's queue must drop oldest, not block the camera. Test with a slow `FakeTribeInference` (sleeps 1s) and a fast poller (yields every 0.1s) — assert dropped count is logged and the loop never deadlocks.
- **Failure isolation:** if `parcellation.generate_report` raises, mesh export still runs (and vice versa). Each downstream module is wrapped in its own try/except; one failure doesn't kill the loop.
- **Graceful shutdown:** `stop()` drains in-flight work and closes the poller.
- **Window→Output traceability:** for every emitted `Bundle`, `bundle.window_id == window.window_id == output.window_id`.

**Tests (`test_window_buffer.py`):**
- A 30s buffer fed at variable rates always emits windows with `25 ≤ duration_s ≤ 35`.
- Audio/video/text are temporally aligned within the window (timestamps overlap).

**Tests (`test_poller.py`):**
- A `FilePoller` (reads a directory of pre-recorded clips) yields `StimulusWindow` objects matching the contract.
- A `MockLivePoller` (programmable) for use by other tests.
- The real RTSP/camera poller is `@pytest.mark.integration` and skipped in unit runs.

### Phase 4 — Inference compartment (1 day) — worktree `wt-inference`

Stub lands day 1; real GPU implementation can be developed in parallel with all other phases since it depends only on `contracts.py`.

```python
# tribe_backend/inference/stub.py
class FakeTribeInference:
    def __init__(self, fixture: TribeOutput): self._out = fixture
    def __call__(self, w: StimulusWindow) -> TribeOutput:
        return replace(self._out, window_id=w.window_id)

# tribe_backend/inference/gpu_runner.py  (lands later)
class GpuTribeInference:
    def __init__(self, weights_path, device="cuda:0"): ...
    def __call__(self, w: StimulusWindow) -> TribeOutput:
        x_text  = self._encode_text(w.text)
        x_audio = self._encode_audio(w.audio, w.audio_sr)
        x_video = self._encode_video(w.video, w.video_fps)
        cort, sub = self._tribe(x_text, x_audio, x_video)   # forward pass
        return TribeOutput(cortical=cort, subcortical=sub, window_id=w.window_id)
```

**Tests:**
- `test_stub.py`: stub returns the fixture, stamps the right `window_id`.
- `test_preprocessing.py`: each modality encoder produces the shape TRIBE expects, on synthetic windows. Pure CPU, no model.
- `test_gpu_runner.py` (`@pytest.mark.gpu`, skipped until weights are downloaded): one real forward pass on a tiny window, output shapes validate.

When the GPU module lands, **nothing in `control/`, `parcellation/`, or `mesh/` changes** — only `gpu_runner.py` is wired in via DI at the composition root.

---

### Phase 5 — Integration Head: TRIBE GPU engine attachment (1–1.5 days) — worktree `wt-integration`

The TRIBE v2 codebase is imported as a vendored dependency (e.g. `third_party/tribev2/` or installed from the upstream repo) — **no refactor of its internals**. Phase 5 is a thin "head" that bolts the imported engine onto the existing `TribeInference` seam by matching ports on both sides. Because every other compartment was built TDD against `FakeTribeInference`, integration collapses to:

1. **Adapter, not rewrite.** A single file `inference/tribev2_adapter.py` wraps the imported engine. It owns model loading and the per-window `forward()` call. Upstream tribev2 is not modified; we touch only the boundary.

2. **Port-matching checklist.** Each side of the seam has a known shape; the adapter's job is just to align them.

   | Direction | Our side | Adapter does | TRIBE engine side |
   |---|---|---|---|
   | In | `StimulusWindow.video` `(F, H, W, 3) uint8` | resample to V-JEPA-2 input fps, channel reorder, normalize | video tensor at 2 Hz, `D_video=1280` after V-JEPA-2 |
   | In | `StimulusWindow.audio` `(S,) float32 @ sr` | resample to Wav2Vec-Bert-2.0 sr, mono-mix | audio tensor at 2 Hz, `D_audio=1024` |
   | In | `StimulusWindow.text` `str` | tokenize via Llama-3.2-3B tokenizer, contextualize | text tensor at 2 Hz, `D_text=2048` |
   | Inside | engine forward pass on GPU | `torch.no_grad()`, AMP, "unseen subject" head | `(T, 20484)` cortical, `(T, 8802)` subcortical |
   | Out | engine raw tensors | move to CPU, cast to `float32`, validate shapes, stamp `window_id` | → `TribeOutput` (validated by the contract) |

3. **What we verify (tests, in `wt-integration`):**
   - `test_adapter_protocol.py` — `GpuTribeInference` satisfies the `TribeInference` protocol (`isinstance`-style structural check via `typing.get_type_hints`); a `StimulusWindow` fixture goes in, a contract-valid `TribeOutput` comes out.
   - `test_adapter_ports.py` — each preprocessing function (`_encode_video`, `_encode_audio`, `_encode_text`) produces tensors with the exact shape/dtype/sample-rate the engine expects. Pure CPU; no full forward pass.
   - `test_gpu_smoke.py` (`@pytest.mark.gpu`) — single forward pass on a tiny synthesized 30s window; asserts `cortical.shape == (31, 20484)` and `subcortical.shape == (31, 8802)` and that values are finite.
   - `test_sample_parity.py` (`@pytest.mark.gpu`) — run the engine on a fixed deterministic stimulus (seeded synthetic video/audio + canned caption); the resulting cortical tensor must be **structurally consistent** with `agent/tribev2_sample_output.txt` (same shape, comparable distribution: mean within ±0.1, std within ±0.3, no NaNs/Infs). This is the "did port-matching work" smoke check, not a numerical-equality test.
   - `test_e2e_user_endpoint.py` (`@pytest.mark.gpu`, `@pytest.mark.integration`) — **the real acceptance test the user asked for**:
     ```python
     control = ControlUnit.default(weights_path=..., device="cuda:0")
     report = control.process_window(
         video=load_real_clip("tests/media/traffic_30s.mp4").video,
         audio=load_real_clip("tests/media/traffic_30s.mp4").audio,
         audio_sr=16000,
         text="A busy traffic intersection at dusk with car horns and pedestrians.",
     )
     assert report.text != ""
     assert len(report.top_regions) >= 1
     assert all(r.z_score >= report.z_threshold for r in report.top_regions)
     assert report.window_id is not None
     ```
     Run with 2–3 short real media clips of distinct content (traffic, faces, speech). For each, the report must be non-empty and traceable. Optional sanity assertion (best-effort, not a hard gate): the face clip surfaces a face/social region (`FFC`, `STSva`, `PGi`) somewhere in the top-10.

4. **Wiring (composition root only):**
   ```python
   # tribe_backend/control/__init__.py — the ONLY edit needed in control/
   def default_control_unit(weights_path, device="cuda:0", out_dir=...):
       return ControlUnit(
           poller=NoopPoller(),  # one-shot users use process_window()
           inference=GpuTribeInference(weights_path, device=device),  # ← swap
           parcellation=GlasserParcellationUnit(),
           mesh=BrainMeshExporter(),
           sink=DiskArtifactSink(out_dir),
       )
   ```
   Swapping `FakeTribeInference` → `GpuTribeInference` is one line. **No edits to `parcellation/`, `mesh/`, or `control/unit.py`.** That is the whole point of Phase 5 being a head — the rest of the system was built test-first against the protocol, so attachment is mechanical.

5. **Failure modes the adapter must absorb (and tests cover):**
   - Weights missing or wrong path → `WeightsNotFoundError` raised at `__init__`, not at first `__call__`.
   - CUDA OOM mid-window → catch, log window_id, raise `InferenceFailure`; `ControlUnit`'s failure-isolation (Phase 3 tests) keeps the loop alive.
   - Engine returns non-finite values → adapter rejects with `InferenceFailure` rather than passing NaN downstream.
   - Window shorter than 30s (poller startup) → adapter pads or refuses per a configured policy; tested both ways.

**Worktree & branch:** `../cleo-wt-integration`, branch `feat/integration-head`. Depends on `feat/inference` (stub) being merged so the protocol exists; otherwise parallel with everything else.

**Merge gate:** PR merges only when `test_adapter_ports.py` and `test_e2e_user_endpoint.py` are green on a machine with a real GPU + weights. CPU-only CI runs the non-`gpu` tests; the `gpu`-marked suite runs nightly on a self-hosted runner.

---

## 4. Test Infrastructure

- **`pytest`** + **`hypothesis`** (property tests on parcellation math) + **`pytest-cases`** (parametrize across surface types / colormaps without explosion).
- **Golden fixtures** in `tests/golden/`:
  - **Real TRIBE sample** at `agent/tribev2_sample_output.txt` — a verbatim `(31, 20484)` float32 cortical prediction for a 30s text+audio+video stimulus ("condition A"), with 31 1-second segments. This is the canonical end-to-end fixture for both `parcellation/` and `mesh/`.
  - One tiny synthetic `(T=4, 20484)` `.npz` with planted hot spots in Amygdala, FFC, V1 for fast unit tests. Expected outputs (top-3 region names, color hashes) committed alongside.
- **Marks:**
  - `@pytest.mark.unit` (default, fast, runs on every save)
  - `@pytest.mark.integration` (loads real atlases — runs in CI)
  - `@pytest.mark.slow` (GLB export, full animation bundle)
  - `@pytest.mark.gpu` (skipped until inference unit exists)
- **CI gate:** unit + integration must pass; slow runs nightly.

---

## 5. Modularity Guarantees (how we keep them clean)

1. **No cross-module imports** between `parcellation/` and `mesh/`. Both depend only on `contracts.py`. Enforced by an import-linter rule in CI.
2. **Atlases and meshes are loaded once, cached on disk.** Constructors take a `cache_dir`. No global state.
3. **Pure functions where possible.** `activation_to_rgba`, `aggregate_temporal`, `rank_and_threshold` take arrays in, return arrays out — trivially testable, trivially parallelizable later.
4. **I/O at the edges only.** Export functions are the only place that writes files; everything else returns numpy arrays / dicts. Lets you swap on-disk for in-memory (e.g., HTTP streaming) without touching logic.
5. **Single source of truth for vertex ordering.** `contracts.py` documents `[0:10242] = LH, [10242:20484] = RH`. Any module that violates this fails the cross-module invariant test (Phase 2.5).

---

## 6. When GPU Inference Lands (Phase 5 — the Integration Head)

The TRIBE engine is **imported, not refactored**. The drop-in checklist:
1. Vendor or `pip install` the upstream tribev2 package; do not modify it.
2. Implement `inference/tribev2_adapter.py::GpuTribeInference` — a thin wrapper that satisfies the `TribeInference` protocol. Owns model loading, per-modality preprocessing, and casting back to `TribeOutput`.
3. Run the Phase-5 port-matching tests: shapes/dtypes/sample-rates align between `StimulusWindow` and what tribev2 expects, and between tribev2's raw output and `TribeOutput`.
4. Run the `test_e2e_user_endpoint.py` real-media test through `ControlUnit.process_window(video, audio, audio_sr, text)`; verify reports are non-empty and traceable.
5. Un-skip the IBC localizer validation tests (Phase 1.6) — they now have real predictions to score against.
6. **No edits to `parcellation/`, `mesh/`, or `control/unit.py`.** If you need to edit them, the contract was wrong; fix the contract, not the consumers. Integration is meant to be one line in `control/__init__.py` swapping the inference implementation, plus the adapter file itself.

---

## 7. Execution Order & Effort

| Phase | Worktree | Branch | Days | Blocking? |
|---|---|---|---|---|
| 0 — Contract + fixtures | `main` (root) | `feat/contracts` | 0.5 | Yes — everything depends on this |
| 1 — GlasserParcellationUnit | `../cleo-wt-parcellation` | `feat/parcellation-unit` | 2–3 | **Parallel with 2, 3-prep, 4** |
| 2 — BrainMeshExporter | `../cleo-wt-mesh` | `feat/mesh-exporter` | 2–3 | **Parallel with 1, 3-prep, 4** |
| 3 — ControlUnit | `../cleo-wt-control` | `feat/control-unit` | 1.5 | **Parallel** during stub phase; final wiring needs 1+2 merged |
| 4 — Inference (stub) | `../cleo-wt-inference` | `feat/inference` | 1 | **Parallel with all** — depends only on contracts |
| 5 — Integration Head (real GPU) | `../cleo-wt-integration` | `feat/integration-head` | 1–1.5 | Depends on 4 merged; can start while 1/2/3 finish |

### 7.1 Parallel execution — required, not optional

Phase 1 and Phase 2 **must** run concurrently on separate git worktrees. This is enforced by the directory-ownership contracts in §3 (each phase touches a disjoint set of paths), so there are no merge conflicts beyond `contracts.py` (frozen after Phase 0) and the test `conftest.py` (extended additively).

**Setup commands (run once, after Phase 0 lands on `main`):**

```bash
# from /home/md2292/cleo
git worktree add ../cleo-wt-parcellation -b feat/parcellation-unit main
git worktree add ../cleo-wt-mesh         -b feat/mesh-exporter    main
git worktree add ../cleo-wt-control      -b feat/control-unit     main
git worktree add ../cleo-wt-inference    -b feat/inference        main
```

All four worktrees run concurrently. Compartmentalization (§2) guarantees disjoint write sets:
- `wt-parcellation` writes only `tribe_backend/parcellation/**` + its tests
- `wt-mesh` writes only `tribe_backend/mesh/**` + its tests
- `wt-control` writes only `tribe_backend/control/**` + its tests (uses `FakeTribeInference` and stub modules from `wt-inference` via the protocol — no source dependency)
- `wt-inference` writes only `tribe_backend/inference/**` + its tests

**Recommended driver:** launch two `ralph-executor` agents in a single message — one per worktree — each given this plan plus its phase scope. Alternatively, the `/impl-race` workflow is appropriate if you want to evaluate competing implementations of the same module; for *different* modules in parallel, use plain parallel ralph-executors. After both branches are green, run `merge-orchestrator` to land them onto `main` in dependency order before starting Phase 3.

**Why this works without conflicts:**
- `contracts.py` is **read-only** for Phases 1 and 2. Any change to the contract triggers a Phase-0 amendment and a re-sync of both worktrees — but the contract is small and was the keystone of Phase 0 review precisely so this doesn't happen.
- Each worktree owns its own subdirectory under `tribe_backend/` and its own test file. `conftest.py` only gains additive fixtures; collisions are caught by import-linter and the path-filter CI check.
- Both worktrees install dependencies independently (`uv sync` or `pip install -e .` per worktree), so Python envs don't cross-contaminate.

**Daily sync:** each worktree rebases on `main` once per day to pull any Phase-4 stub changes. No long-lived divergence.

### 7.2 Merge gate

Merge order (via `merge-orchestrator`):
1. `feat/contracts` → `main` (already done at end of Phase 0).
2. `feat/inference` (stub) → `main` — unblocks `control` final wiring.
3. `feat/parcellation-unit` and `feat/mesh-exporter` → `main` (independent, either order).
4. `feat/control-unit` → `main` last — it imports the other three at the composition root.
5. The cross-module invariant test (§3, Phase 2.5) is added in the control-unit PR since that's the first place all modules coexist.

The real GPU `gpu_runner.py` lands in a follow-up PR on `feat/inference` once weights are downloaded; only `inference/__init__.py` swaps `FakeTribeInference` → `GpuTribeInference` at the DI site in `control/unit.py`'s default factory.

---

## 8. Dependencies

```
# core
numpy, scipy, nibabel, nilearn, matplotlib

# tests
pytest, pytest-cases, hypothesis, pytest-cov

# optional
trimesh           # GLB export
import-linter     # enforce module boundaries in CI
```

---

## 9. Acceptance Criteria

- `pytest -m "unit or integration"` is green.
- Coverage ≥ 90% on `parcellation/` and `mesh/`.
- **Real-sample tests pass on both compartments:** `test_parcellation_real_sample.py` and `test_mesh_real_sample.py` consume `agent/tribev2_sample_output.txt` end-to-end and produce valid artifacts. These are PR-merge gates for Phases 1 and 2 respectively.
- `ControlUnit.run_once()` driven by a `FakeTribeInference` returning the parsed real sample produces a non-empty report and a binary mesh bundle whose `brain_meta.json` validates.
- The cross-module invariant test passes on the real sample: top-ranked parcel ↔ brightest mesh region.
- `import-linter` passes (no cross-compartment imports outside `control/`).
- Replacing `FakeTribeInference` with `GpuTribeInference` requires zero changes to `parcellation/`, `mesh/`, or `control/` source code.
