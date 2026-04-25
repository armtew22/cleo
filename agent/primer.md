---
name: primer
description: Orientation for the wip-3d-map-parser branch — building the BrainMeshExporter (3D fsaverage5 mesh + activation coloring + Swift-consumable export) per agent/brain-3d-parser.md.
type: project
---

# cleo / tribe-backend — Primer

Generated: 2026-04-25 (focus: 3D Brain Mesh Parser → Swift UI; `agent/brain-3d-parser.md`)

**Branching note:** This work belongs on a separate `origin/wip-3d-map-parser` branch (per user). Do not initiate the build yet — this primer is orientation only.

**Scope correction (2026-04-25):** The deliverable on this branch is a **standalone Swift UI module** that renders the exporter's output. The Python exporter (`tribe_backend/mesh/`) already exists on `main` and is treated as fixed input — this branch consumes its binary/JSON/GLB outputs. The Swift module must be drop-in integratable into a larger frontend (no app-level coupling, no global state, no hard dependency on a specific navigation stack).

## Tech Stack
- **Language**: Python ≥3.10 (pure backend; no JS/Node)
- **Numerics**: numpy, scipy
- **Neuroimaging**: nibabel, nilearn (fsaverage5 surface), matplotlib (colormaps)
- **3D export**: trimesh (optional, for GLB)
- **Test**: pytest, pytest-cases, pytest-cov, hypothesis, import-linter
- **Build**: setuptools / pyproject.toml

## Architecture Pattern
**Compartmented monolith** with one shared frozen contract (`tribe_backend/contracts.py`). Sibling compartments — `inference/`, `mesh/`, `parcellation/`, `control/` — are isolated by `import-linter` and communicate only via `TribeOutput` / `StimulusWindow`. The 3D parser lives entirely under `tribe_backend/mesh/`.

## Mesh Compartment — Current State on `main`

Already scaffolded (these files exist and have tests):

```
tribe_backend/mesh/
├── __init__.py        # exports: BrainMeshExporter, activation_to_rgba, aggregate_temporal, load_fsaverage5
├── geometry.py  (107 LOC)  # load_fsaverage5(): coords/faces/sulc per hemi + combined
├── colormap.py   (62 LOC)  # activation_to_rgba(): scalar (20484,) → uint8 RGBA with sulcal shading
└── exporter.py  (214 LOC)  # BrainMeshExporter + aggregate_temporal; JSON / binary / GLB / animation outputs

tests/mesh/  # mirrors the layers 1:1
├── test_geometry.py            test_colormap.py        test_aggregate.py
├── test_export_json.py         test_export_binary.py   test_export_glb.py
├── test_export_animation.py    test_mesh_real_sample.py
```

Treat `agent/brain-3d-parser.md` as the **target spec**; treat the existing mesh module as the **starting baseline**. Before writing new code, diff the spec against the existing module to identify what's missing, divergent, or already done.

## Output Contract (per `agent/brain-3d-parser.md`)

Single entry point: `tribe_backend.mesh.BrainMeshExporter`. Pipeline (4 layers, bottom-up):

1. **`load_fsaverage5(surface_type, cache_dir) → meshes`** — load LH/RH `coords (10242,3)` + `faces (~20480,3)` + `sulc (10242,)` from FreeSurfer geometry via `nilearn.datasets.fetch_surf_fsaverage('fsaverage5')`. Combined mesh offsets RH face indices by `n_lh`. Surface types: `pial` (default), `inflated`, `white`.
2. **`aggregate_temporal(preds, method, time_idx)` → `(20484,)`** — collapse `(T, 20484)` to one frame. Methods: `mean`, `peak` (timestep with max mean), `rms`, `frame` (specific `time_idx`).
3. **`activation_to_rgba(activations, cmap, vmin, vmax, sulc_darken)` → `(20484, 4) uint8`** — matplotlib colormap (default `RdBu_r`, clim `[-3, 3]`); sulci darkened by `1 - sulc_darken * clip(sulc, 0, 1)`.
4. **`BrainMeshExporter.export_*`** — four formats:
   - `export_json(activations, path, cmap, vmin, vmax, hemispheres, separate)` — vertices/normals/faces/colors as flat lists in a JSON bundle (`tribe_brain_mesh_v1`); LH/RH split when `separate=True`.
   - `export_binary(activations, dir, ...)` — raw `.bin` files (`brain_vertices.bin`, `brain_normals.bin`, `brain_faces.bin`, `brain_colors.bin`) + `brain_meta.json`. Mesh files are static (ship-once); only `brain_colors.bin` updates per inference.
   - `export_glb(activations, path, ...)` — binary glTF via `trimesh`, vertex_colors baked in.
   - `export_animation_bundle(preds, dir, ...)` — static geometry once + per-frame `brain_colors_t{NNNN}.bin` at 1 Hz.

### Invariants worth knowing
- Cortical input must be `(T, 20484)` or `(20484,)`; layout is `[0:10242]=LH, [10242:20484]=RH` (matches `contracts.LH_VERTICES`/`RH_VERTICES`).
- Subcortical `(T, 8802)` is **out of scope** for this exporter — it has no surface mesh and is handled by the parcellation compartment instead.
- Per-vertex normals: area-weighted face-normal accumulation (see `_compute_vertex_normals`).
- Static mesh ≈ 960 KB; per-frame colors ≈ 80 KB; 60 s animation ≈ 5.7 MB.
- All exported numerics use plain Python types (or `np.float32`/`np.uint8` for binary) for stable, lossless serialization.

## Critical Files for Development (Swift module focus)

When build begins on `wip-3d-map-parser`:
1. **`swift/TribeBrainView/Package.swift`** — declares library + UI overlay products and test target.
2. **`swift/TribeBrainView/Sources/TribeBrainView/BrainMeshLoader.swift`** — buffer parsing + `SCNGeometry` assembly; the trust boundary with the Python exporter.
3. **`swift/TribeBrainView/Sources/TribeBrainView/BrainSceneView.swift`** — `UIViewRepresentable` + camera/lighting setup; the host integration surface.
4. **`swift/TribeBrainView/Sources/TribeBrainView/BrainAnimationController.swift`** — per-frame color-buffer swap; the perf-sensitive path.
5. **`swift/TribeBrainView/Tests/TribeBrainViewTests/`** — fixture-driven parsing & rendering tests.

## Critical Files for Development (Python exporter — input source)

1. **`tribe_backend/mesh/exporter.py`** — `BrainMeshExporter`, `aggregate_temporal`. Primary unit under test for any export-format change.
2. **`tribe_backend/mesh/geometry.py`** — `load_fsaverage5`. Touch when changing surface type, cache strategy, or combined-mesh layout.
3. **`tribe_backend/mesh/colormap.py`** — `activation_to_rgba`. Touch for colormap, clipping, or sulcal-shading tweaks.
4. **`tribe_backend/contracts.py`** — `TribeOutput`, `CORTICAL_VERTICES=20484`, `LH_VERTICES=RH_VERTICES=10242`, `EXPECTED_T_PER_30S_WINDOW=31`. Frozen contract — do not modify here.
5. **`tests/mesh/`** — eight files mirroring the four layers; extend here first, including `test_mesh_real_sample.py` for end-to-end with a real `TribeOutput`.

## Swift UI Module — Primary Deliverable for this Branch

A standalone, embeddable SwiftUI/SceneKit module that ingests `BrainMeshExporter` output and renders the brain interactively. Designed for drop-in integration into any host app.

### Module shape (proposed)
- **Distribution:** Swift Package (SPM) — `Package.swift` with one library product, e.g. `TribeBrainView`. Independent of the Python repo's build system; can live in `swift/TribeBrainView/` (or a sibling repo) and be added via `.package(path:)` or git URL.
- **Public API surface (kept minimal):**
  - `BrainMeshLoader` — loads from `URL` (binary dir, JSON file, or GLB). Returns `SCNGeometry`. Pure I/O, no UI.
  - `BrainSceneView: UIViewRepresentable` (and `NSViewRepresentable` for macOS) — renders the geometry. Inputs: mesh source, optional `@Binding` for current frame, optional camera/lighting config struct.
  - `BrainAnimationController` — owns frame index, drives color-buffer swaps for `(T, 20484)` animation bundles at configurable Hz.
  - `BrainViewConfiguration` — value type bundling colormap range echo, background, lighting model, camera defaults. Host app passes one in; no singletons.
- **Boundaries (non-negotiable for "standalone"):**
  - No `@EnvironmentObject` requirements; all state via `@Binding`/init args.
  - No hardcoded asset paths; consumers pass `URL`s.
  - No analytics, networking, or persistence inside the module.
  - Overlay UI (timeline scrubber, region report card) ships as **opt-in** views in a separate target (`TribeBrainViewUI`) so the core renderer has zero SwiftUI-component opinions beyond the scene host.

### Interop contract with the Python exporter
- **Preferred path:** binary buffers (smallest, fastest cold-load, animation-friendly). Layout per `agent/brain-3d-parser.md` §3.2:
  - `brain_vertices.bin` — `float32 × N×3`, stride 12, semantic `.vertex`
  - `brain_normals.bin`  — `float32 × N×3`, stride 12, semantic `.normal`
  - `brain_faces.bin`    — `uint32  × F×3`, `bytesPerIndex: 4`, `.triangles`
  - `brain_colors.bin`   — `uint8   × N×4`, stride 4, semantic `.color` (RGBA)
  - `brain_meta.json`    — `{vertex_count, face_count, lh_vertex_count, rh_vertex_count, ...}`
- **Animation:** static geometry loaded once; module memory-maps `brain_colors_t{NNNN}.bin` and rebuilds only the color `SCNGeometrySource` per tick (geometry, normals, faces, materials reused).
- **Fallbacks:** JSON bundle (`tribe_brain_mesh_v1`) for debugging / max introspection; GLB via ModelIO when the host already standardizes on glTF.

### Platform / dependency targets
- iOS 16+, macOS 13+, visionOS 1+ (RealityKit path optional; SceneKit is the common floor).
- Pure Apple frameworks: `SceneKit`, `SwiftUI`, `Foundation`, `ModelIO` (GLB only). No third-party Swift packages in the core target.

### Repo layout (proposed for this branch)
```
swift/
└── TribeBrainView/
    ├── Package.swift
    ├── Sources/
    │   ├── TribeBrainView/         # core: loader, scene view, animation controller
    │   └── TribeBrainViewUI/       # optional overlays: scrubber, report card
    ├── Tests/
    │   └── TribeBrainViewTests/    # XCTest: buffer parsing, geometry assembly, frame swap
    └── Examples/
        └── BrainViewerDemo/         # minimal SwiftUI host demonstrating embed
```
Fixtures for tests come from running the Python exporter against a stored `TribeOutput` sample (`tests/mesh/test_mesh_real_sample.py` already produces these); commit a small, deterministic fixture set under `Tests/Fixtures/`.

## Notable
- `pyproject.toml` markers: `unit` / `integration` / `slow` / `gpu`. Default `pytest` run hits `unit`.
- import-linter forbids `mesh` from importing `inference`, `parcellation`, or `control` (and vice versa).
- The parcellation compartment (`tribe_backend/parcellation/`) is the qualitative-text counterpart and ships alongside the mesh export per the data-flow diagram in §4 of the spec.
- Remote branches: `origin/wip-backend` exists; `origin/wip-3d-map-parser` is where this work will land.
- No `CLAUDE.md` in repo.
