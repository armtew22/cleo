# TribeBrainView — Build Status (Phase 8+9)

**Date:** 2026-04-25
**Branch:** wip-3d-map-parser
**Status:** Ready for macOS validation. v0.1.0 tag DEFERRED pending macOS CI.

---

## Shipped commits (Phases 1–7 + 6b + 6c)

| SHA | Description |
|---|---|
| `3870e0f` | feat(swift): Phase 1 — TribeBrainView SPM skeleton |
| `a97026a` | feat(fixtures): synthetic TRIBE fixtures for Swift loader tests |
| `513dfc2` | feat(api): mesh static + inference routes (/v1/mesh/static, /v1/inference/colors\|full) |
| `dc20ae8` | feat(swift): Phase 3+4 — binary loader + color swap |
| `11e637f` | feat(swift): Phase 5+6 — scene view, camera presets, animation controller |
| `a436755` | feat(swift): Phase 7 — overlay UI (scrubber, region card, colorbar) |
| `34a2aec` | feat(swift): Phase 6c — BrainMeshClient + polling + remote source |

---

## Linux-runnable validation

### pytest tests/api/test_mesh_routes.py

```
21 passed, 2 warnings in 7.14s
```

All 21 tests green:
- `TestMeshStatic` (7 tests): 200 response, tar contents, ETag, Cache-Control, 304
- `TestInferenceColors` (8 tests): binary response, exact 81936-byte body, X-* headers, 422 on missing image
- `TestInferenceFull` (6 tests): JSON envelope shape, colors_url follow-up, 404 on expired window, 422

### Smoke script (scripts/smoke_mesh_endpoints.py)

```
Total checks : 26
Passed       : 26
Failed       : 0
Result       : ALL PASS
```

Checks performed against a live `uvicorn` process on port 18000:
1. `GET /v1/mesh/static` → 200 + tar with 4 expected files + ETag + Cache-Control: immutable
2. `POST /v1/inference/colors` with 3-byte stub image → 200 + exactly 81936-byte body + all 5 X-* headers
3. `POST /v1/inference/full` → JSON envelope shape + `colors_url` GET → 81936 bytes
4. `GET /v1/mesh/static` with `If-None-Match` → 304

Full output: `agent/reports/swift-brain-smoke-output.txt`

---

## Deferred to macOS validation (explicit checklist)

- [ ] `swift test` — TribeBrainViewTests target (XCTest; requires macOS/Swift toolchain)
- [ ] `swift test` — TribeBrainViewUITests target (snapshot tests)
- [ ] `xcodebuild test` — iOS Simulator (iPhone 15 scheme)
- [ ] `swift package diagnose-api-breaking-changes` against tag `0.0.0` to freeze public API surface
- [ ] Demo app (BrainViewerDemo): static brain loads, 5-frame animation at 1 Hz, camera presets cycle
- [ ] Demo app "Live" mode: `AVCaptureSession` frames → `POST /v1/inference/colors` → brain updates
- [ ] Instruments profiling: target 60 fps idle, color-swap < 4 ms on iPhone 15

---

## Open follow-ups (§8 of swift-brain-view-plan.md)

1. **Subcortical extension** — port `tribev2/plotting/subcortical.py` marching-cubes pipeline into `BrainMeshExporter`; Swift module gains a second `BrainMeshSource.Kind`.
2. **`"half"` inflate mode** — blend pial/inflated 50/50 in the Python exporter (upstream `base.py:127` reference).
3. **kNN resampler** — clean reimplementation of `get_stat_map` (`base.py:178`) in `tribe_backend/mesh/resample.py` for fsaverage5↔6↔7 support.
4. **RealityKit renderer** for visionOS — share `BrainMeshLoader`, swap the scene view.
5. **Live IPC / streaming colors** — WebSocket → ring buffer → `updateColors`. Defer until demand signal.

---

## Definition of Done (§9) — checklist

| Item | Status |
|---|---|
| `swift test` green on macOS-14 + iOS Simulator (matrix CI) | 🟡 Deferred to macOS runner |
| `pytest tests/api/test_mesh_routes.py` green; `uvicorn` boots cleanly | ✅ 21/21 passed |
| Demo app: static brain, 5-frame animation at 1 Hz, camera presets cycle | 🟡 Deferred to macOS runner |
| Demo app "Live" mode polls local FastAPI and updates brain from posted images | 🟡 Deferred to macOS runner |
| README copy-paste snippets work for both local-files and remote embedding | ✅ Verified against public API; `BrainMeshSource.Kind`, `BrainSceneView`, `BrainMeshClient`, `AggregationMethod` all match committed code |
| Public API frozen under `swift package diagnose-api-breaking-changes` | 🟡 Deferred to macOS runner |
| License header on every source file (Apache-2.0) | ✅ All 19 source files + 7 test files verified |
| Tag `v0.1.0`; PR `wip-3d-map-parser` → `main` open and reviewed | ❌ Not tagged — macOS validation required first |
| No dependency on tribev2 source (reference-only citations OK) | ✅ Camera presets and kNN reimplemented independently; no CC BY-NC code copied |

**v0.1.0 is NOT tagged.** Tag and PR once macOS CI confirms `swift test` green on both targets.
