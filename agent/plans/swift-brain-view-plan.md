# Plan: `TribeBrainView` — Pluggable SwiftUI 3D Brain Renderer

**Branch:** `wip-3d-map-parser` (off `main`)
**Generated:** 2026-04-25
**Target:** Standalone Swift Package, drop-in for any host SwiftUI app, renders TRIBE v2 cortical activations on the fsaverage5 mesh — fed by a polling FastAPI loop where the **device renders, the server emits data**.

---

## 0. Question Asked: Does this work directly with raw TRIBE model output?

**No — and that's the design.** The interop boundary is the Python `BrainMeshExporter` (already on `main` at `tribe_backend/mesh/`), not the raw `(T, 20484)` tensor.

```
TRIBE v2 inference
   └─ TribeOutput.cortical: (T, 20484) float32
       └─ BrainMeshExporter.export_binary()  ← Python boundary
            ├─ brain_vertices.bin   (static, ~240 KB)
            ├─ brain_normals.bin    (static, ~240 KB)
            ├─ brain_faces.bin      (static, ~480 KB)
            ├─ brain_colors.bin     (per-frame, ~80 KB each)
            └─ brain_meta.json
                 └─ TribeBrainView (Swift) ← consumes
```

**Why not direct:**
- Pushing raw tensors into Swift would mean re-implementing fsaverage5 mesh loading (`nibabel`/`nilearn`), kNN resampling, sulcal shading, and matplotlib colormaps in Swift. Wrong language, wrong libraries, brittle.
- The exporter is already tested (8 test files in `tests/mesh/`), bundled with the Glasser atlas, and produces deterministic byte-for-byte output.
- Binary buffers are the smallest, fastest, animation-friendliest path: SceneKit memory-maps them straight into `SCNGeometrySource`.
- Host apps that don't run Python can ship the static mesh files (`vertices/normals/faces`, ~960 KB) once with the app bundle and only stream `brain_colors.bin` per inference.

**What "raw output" path *would* look like (rejected):** stream `(T, 20484) float32` over the wire to Swift, perform sulcal shading and colormap evaluation in a Metal compute shader, load fsaverage5 geometry from a baked `.bin` shipped in the package. Faster cold-start for live inference, but couples Swift to neuroimaging conventions and forks the rendering logic from Python's well-tested path. Defer until a real perf requirement demands it.

**Conclusion:** Swift module ingests `BrainMeshExporter` output. Period.

---

## 0b. End-to-End Loop (frontend polling → server → on-device render)

The device drives the loop. The server stays stateless and CPU-only — no offscreen GL.

```
┌── Device (SwiftUI host app) ─────────────────────────────────────┐
│  Camera frames @ N Hz  ──────────┐                                │
│                                  ▼                                │
│  POST /v1/inference/colors  (multipart: image / audio / text)     │
│                                  │                                │
│                                  │   (one-time on first launch)   │
│  GET /v1/mesh/static  ──────────►│   cached forever via ETag      │
└──────────────────────────────────┼────────────────────────────────┘
                                   ▼
┌── Server (FastAPI, tribe_backend) ───────────────────────────────┐
│  TRIBE v2 inference → TribeOutput(T, 20484)                       │
│  ├─ BrainMeshExporter.aggregate_temporal + activation_to_rgba     │
│  │   → brain_colors.bin  (uint8 RGBA × 20484 = 80 KB / frame)     │
│  └─ GlasserParcellationUnit.generate_report  (top regions JSON)   │
└──────────────────────────────────┬────────────────────────────────┘
                                   ▼
┌── Device (TribeBrainView) ───────────────────────────────────────┐
│  BrainMeshLoader.updateColors(...) — replace only color source    │
│  RegionReportCard renders the qualitative report alongside        │
│  Free orbit / zoom / camera presets — no new request needed       │
└───────────────────────────────────────────────────────────────────┘
```

**Why on-device render (not server-rendered images):**

| | Server render (PNG/MP4/GLB w/ baked colors) | On-device (this plan) |
|---|---|---|
| Per-poll payload | 200 KB – 2 MB image, or 5+ MB animation | **~80 KB binary** after one-time ~960 KB mesh bootstrap |
| Interactivity | Camera frozen; rotate/zoom = new request | Free orbit / zoom / preset cycling, zero round-trips |
| Server stack | Needs offscreen VTK + xvfb + GPU — fragile | Pure CPU/numpy; reuses existing exporter |
| Latency | inference + render + encode + transfer | inference + transfer (render ≈ 4 ms on-device) |
| Animation | Pre-baked video, fixed fps | Scrubbable, host-controlled fps |

The server-render path only wins for non-Apple clients or hardware too weak for SceneKit — neither applies. **Server emits data; device renders pixels.**

---

## 1. Scope & Non-Goals

### In scope
- Load the binary-buffer export format produced by `BrainMeshExporter.export_binary()` from **either local files or HTTP**.
- A pair of FastAPI endpoints (`GET /v1/mesh/static`, `POST /v1/inference/colors`, optional `POST /v1/inference/full`) that wrap the existing exporter — no new render code on the server.
- A Swift networking layer (`BrainMeshClient`) that fetches the static mesh once, caches it via ETag, and pulls per-poll color buffers.
- Render the cortical mesh with per-vertex RGBA colors in SceneKit (iOS 16+, macOS 13+, visionOS 1+).
- Animate by swapping color buffers from `export_animation_bundle()` (file path) or streamed multi-frame responses (network path) at the configured Hz.
- Camera presets (lateral, medial, dorsal, ventral, anterior, posterior — adapted from upstream `VIEW_DICT` in `cortical_pv.py:27` via clean reimplementation, CC BY-NC clearance).
- Optional overlay UI (timeline scrubber, region report card) in a **separate, opt-in** target.
- One example host app demonstrating drop-in embedding **and** the polling loop against a local FastAPI dev server.

### Out of scope (this branch)
- Subcortical rendering (the 8802-voxel output) — Python exporter doesn't yet support it. Tracked as a future extension informed by `tribev2/plotting/subcortical.py` (marching-cubes blueprint).
- Server-side rendering (PNG/MP4/GLB-with-baked-colors). Explicitly rejected — see §0b.
- WebSocket / SSE streaming. v0.1.0 is request/response polling; streaming is a future minor.
- Auth, multi-tenant session state, persistent storage of inference results. Endpoints are stateless.
- WebGL / Android / cross-platform port. Apple platforms only.
- GLB/JSON loader paths beyond a thin compatibility shim — binary is the primary format.
- Server-side rendering / video export. SceneKit on-device only.
- Bundling the static mesh `.bin` files inside the package — host apps ship them.

### Non-negotiable boundaries (what makes it "pluggable")
- **No `@EnvironmentObject` requirements** — all state via `@Binding` / init args.
- **No singletons, no global state, no app-level navigation assumptions.**
- **No third-party Swift packages** in the core target. SceneKit + SwiftUI + Foundation only. ModelIO only if/when GLB path is enabled.
- **No hardcoded asset paths** — consumers pass `URL`s.
- **No analytics, logging frameworks, networking, or persistence.**
- **Overlay UI lives in a separate target** so the core renderer stays headless-friendly.

---

## 1b. FastAPI Contract — server side

Lives under `tribe_backend/api/mesh_routes.py` (new file). Wraps existing `BrainMeshExporter` + `GlasserParcellationUnit`. No new render code.

### `GET /v1/mesh/static?surface=pial`
One-time bootstrap, cacheable forever (geometry never changes).
```
200 application/octet-stream
Headers:
  ETag: "fsaverage5-pial-v1"
  Cache-Control: public, max-age=31536000, immutable
Body: tar archive containing
  ├─ brain_vertices.bin   (~240 KB)
  ├─ brain_normals.bin    (~240 KB)
  ├─ brain_faces.bin      (~480 KB)
  └─ brain_meta.json
```
Implementation: load fsaverage5 once on startup (`load_fsaverage5(surface_type)`), serialize via `BrainMeshExporter` to an in-memory `BytesIO`, tar it, return `StreamingResponse`. Honor `If-None-Match` → `304`.

### `POST /v1/inference/colors`
Hot path. Multipart input → 80 KB binary out.
```
Request: multipart/form-data
  image: image/jpeg | video/mp4   (required)
  audio: audio/wav                 (optional)
  text:  text/plain                (optional)
  method: window_mean | peak | peak_window  (default window_mean)
  vmin / vmax / cmap               (optional overrides)

Response: 200 application/octet-stream
Headers:
  X-Window-Id: <uuid echoed back>
  X-Tribe-Method: window_mean
  X-Vmin: -3
  X-Vmax: 3
  X-Frame-Count: 1                  (>1 only if animation requested)
  Content-Encoding: gzip            (middleware)
Body: brain_colors.bin (uint8 RGBA × 20484 = 80 KB; concatenated for animation)
```

### `POST /v1/inference/full` (optional)
Same input; envelope response combines colors URL + qualitative report.
```json
{
  "window_id": "abc123",
  "colors_url": "/v1/inference/colors/abc123",
  "report": { ...GlasserParcellationUnit.Report... },
  "vmin": -3, "vmax": 3, "method": "window_mean"
}
```
Swift fetches the report inline and the binary on a parallel `URLSession` task.

### Server invariants
- Stateless: `colors_url` is a short-TTL signed URL or in-memory cache key, not a session.
- Deterministic: same input + seed → same bytes (already guaranteed by exporter).
- Hot mesh load: `load_fsaverage5()` runs once at startup, held in app state.
- Compression: `gzip` middleware (RGBA buffers compress 30–40% from sulcal-shading redundancy).
- No GPU, no offscreen GL anywhere on the server.

---

## 2. Public API Surface

Keep it tiny. Five public types in the core target.

```swift
// MARK: - Core target: TribeBrainView

public struct BrainMeshSource {
    public enum Kind {
        case binaryDirectory(URL)                              // local files
        case remote(BrainMeshClient)                           // FastAPI: static + per-poll colors
        case json(URL)
        case glb(URL)
    }
    public let kind: Kind
}

/// Talks to the FastAPI server. Caches static mesh by ETag.
public actor BrainMeshClient {
    public init(baseURL: URL,
                session: URLSession = .shared,
                cacheDirectory: URL? = nil)            // defaults to Application Support
    public func bootstrapStaticMesh() async throws -> URL // local cached dir
    public func fetchColors(image: Data,
                            audio: Data? = nil,
                            text: String? = nil,
                            method: AggregationMethod = .windowMean) async throws -> ColorsResponse
    public func fetchFull(image: Data, ...) async throws -> InferenceEnvelope  // colors + report
}

public struct ColorsResponse {
    public let windowId: String
    public let frameCount: Int
    public let vmin: Double; public let vmax: Double
    public let buffer: Data           // raw uint8 RGBA × 20484 × frameCount
}

public struct InferenceEnvelope {
    public let windowId: String
    public let colors: ColorsResponse
    public let report: BrainReport    // Codable mirror of GlasserParcellationUnit.Report
}

public struct BrainMeshAssets {
    public let geometry: SCNGeometry
    public let vertexCount: Int
    public let lhVertexCount: Int
    public let rhVertexCount: Int
    public let faceCount: Int
    public let colormapMeta: ColormapMeta   // echoed from brain_meta.json
}

public enum BrainMeshLoader {
    public static func load(_ source: BrainMeshSource) throws -> BrainMeshAssets
    public static func updateColors(on geometry: SCNGeometry,
                                    from url: URL,
                                    vertexCount: Int) throws
}

public struct BrainViewConfiguration {
    public var cameraPreset: CameraPreset = .lateralLeft
    public var background: Color = .clear
    public var lighting: LightingMode = .physicallyBased
    public var allowsCameraControl: Bool = true
    public var smoothShading: Bool = true
    public init() {}
}

public enum CameraPreset: String, CaseIterable {
    case lateralLeft, lateralRight, medialLeft, medialRight,
         dorsal, ventral, anterior, posterior, frontalThreeQuarter, custom
}

public struct BrainSceneView: View {
    public init(source: BrainMeshSource,
                configuration: BrainViewConfiguration = .init(),
                currentFrame: Binding<Int>? = nil,    // nil → static
                animation: BrainAnimationBundle? = nil)
    public var body: some View { ... }
}

public final class BrainAnimationController: ObservableObject {
    @Published public var currentFrame: Int
    public init(bundle: BrainAnimationBundle, fps: Double = 1.0)
    public func play(); public func pause(); public func seek(to frame: Int)
}

public struct BrainAnimationBundle {
    public let directory: URL
    public let frameCount: Int
    public let fps: Double
    public init(directory: URL) throws    // reads brain_anim_meta.json
}
```

### Optional overlay target: `TribeBrainViewUI`
```swift
public struct BrainTimelineScrubber: View { ... }     // bound to BrainAnimationController
public struct RegionReportCard: View { ... }          // consumes a Codable Report from parcellation
public struct ColorbarLegend: View { ... }            // reads ColormapMeta
```
Host apps that want zero opinions skip this target entirely.

---

## 3. File-by-File Breakdown

```
swift/TribeBrainView/
├── Package.swift
├── README.md                                # 3-screen quickstart, embed snippet
├── Sources/
│   ├── TribeBrainView/
│   │   ├── BrainMeshSource.swift            # value types: source kind, configuration
│   │   ├── BrainMeshAssets.swift            # loaded result struct
│   │   ├── BrainMeshLoader.swift            # binary + JSON parsers; SCNGeometry assembly
│   │   ├── BrainMeshLoader+GLB.swift        # ModelIO path, gated behind canImport
│   │   ├── BrainSceneView.swift             # UIViewRepresentable / NSViewRepresentable
│   │   ├── BrainSceneCoordinator.swift      # SCNView lifecycle, color-swap on update
│   │   ├── BrainAnimationController.swift   # CADisplayLink-driven frame ticker
│   │   ├── BrainAnimationBundle.swift       # parse brain_anim_meta.json + URL pattern
│   │   ├── BrainMeshClient.swift            # actor: ETag cache, multipart upload, gzip decode
│   │   ├── BrainMeshClient+Polling.swift    # AsyncSequence helper for N-Hz polling loops
│   │   ├── BrainReport.swift                # Codable mirror of parcellation Report
│   │   ├── CameraPresets.swift              # (view_vector, up_vector) tuples; lifted-clean
│   │   ├── ColormapMeta.swift               # Codable mirror of brain_meta.json
│   │   └── Internal/
│   │       ├── BinaryBufferReader.swift     # Data → typed pointer helpers
│   │       └── SCNGeometry+VertexColor.swift
│   └── TribeBrainViewUI/
│       ├── BrainTimelineScrubber.swift
│       ├── RegionReportCard.swift
│       ├── ColorbarLegend.swift
│       └── Report.swift                     # Codable mirror of GlasserParcellationUnit.Report
├── Tests/
│   ├── TribeBrainViewTests/
│   │   ├── BinaryLoaderTests.swift          # parses fixtures, asserts counts/strides
│   │   ├── GeometryAssemblyTests.swift      # SCNGeometrySource shape correctness
│   │   ├── ColorSwapTests.swift             # only color source changes; geom/normals reused
│   │   ├── AnimationBundleTests.swift       # meta parse, frame URL resolution
│   │   ├── MeshClientTests.swift            # mocked URLProtocol: ETag cache, gzip decode, error paths
│   │   └── CameraPresetTests.swift          # vector orthogonality, no NaNs
│   ├── TribeBrainViewUITests/
│   │   └── ScrubberSnapshotTests.swift
│   └── Fixtures/
│       ├── README.md                        # how fixtures were generated
│       ├── static/                          # one frame: 4 .bin + meta.json
│       └── animation/                       # 5 frames + anim_meta.json
└── Examples/
    └── BrainViewerDemo/                     # SwiftUI iOS app target
        ├── BrainViewerApp.swift
        ├── ContentView.swift                # embeds BrainSceneView + scrubber
        └── BundledFixtures/                 # symlink to ../../Tests/Fixtures
```

---

## 4. Build Plan — Phased

Each phase ends with a green test suite and a working `swift test` invocation.

### Phase 1 — Package skeleton (½ day)
1. `mkdir -p swift/TribeBrainView` and run `swift package init --type library --name TribeBrainView`.
2. Edit `Package.swift`: declare two products (`TribeBrainView`, `TribeBrainViewUI`), two test targets, set `platforms: [.iOS(.v16), .macOS(.v13), .visionOS(.v1)]`.
3. Drop placeholder source files matching §3 layout.
4. CI smoke: `swift build` and `swift test` pass with a single `XCTAssertTrue(true)` test.

**Gate:** `swift test` returns 0 on macOS host.

### Phase 2 — Fixtures (½ day)
1. Add a Python script `scripts/generate_swift_fixtures.py` that:
   - Loads a deterministic synthetic `TribeOutput` (`np.random.default_rng(42)` over `(31, 20484)`).
   - Calls `BrainMeshExporter.export_binary()` → `Tests/Fixtures/static/`.
   - Calls `export_animation_bundle()` for 5 frames → `Tests/Fixtures/animation/`.
   - Writes a `MANIFEST.json` with byte-counts and SHA-256 of each file for test assertions.
2. Run once; commit the resulting fixtures (~5 MB total — acceptable for a renderer test pack).
3. Do **not** make tests regenerate fixtures; cross-language test deps are pain.

**Gate:** `Tests/Fixtures/static/brain_meta.json` parses as `{vertex_count: 20484, ...}`.

### Phase 3 — Binary loader (1 day)
1. `BinaryBufferReader` — wraps `Data` with stride/count validation, throws `BrainMeshError.malformed`.
2. `BrainMeshLoader.loadBinary(URL)`:
   - Parse `brain_meta.json` (Codable).
   - Memory-map (`Data(contentsOf:options: .mappedIfSafe)`) the four `.bin` files.
   - Build `SCNGeometrySource`s with the exact stride/component layout from `agent/brain-3d-parser.md` §3.2 (vertex/normal stride 12, color stride 4, bytesPerIndex 4).
   - Build a single `SCNGeometryElement` (triangles).
   - Return `BrainMeshAssets`.
3. Unit tests assert: vertex count, face count, LH/RH split (`lh_vertex_count + rh_vertex_count == 20484`), `SCNGeometrySource.semantic` for each entry.

**Gate:** `BinaryLoaderTests` green.

### Phase 4 — Color swap path (½ day)
1. `BrainMeshLoader.updateColors(on:from:vertexCount:)`:
   - Parse only `brain_colors.bin`.
   - Rebuild `SCNGeometry` reusing the existing vertex/normal sources and element; replace only the `.color` source.
2. `ColorSwapTests` asserts: `geometry.sources(for: .vertex)` is **identity-equal** before/after (no realloc); only `.color` source changes.

**Gate:** `ColorSwapTests` green; instrument with `os_signpost` for later perf work.

### Phase 5 — Scene view + camera (1 day)
1. `BrainSceneCoordinator`: owns the `SCNView`, sets up an `SCNCamera`, applies `BrainViewConfiguration`.
2. `CameraPresets.swift`: 9 presets as `(SCNVector3 viewVector, SCNVector3 up)` tuples — derived independently from anatomy textbooks and the existing `pyvista` defaults documented in `tribev2/plotting/cortical_pv.py:27`. **Reimplement; do not copy verbatim** (CC BY-NC).
3. `BrainSceneView`: `UIViewRepresentable` + `NSViewRepresentable` (use `#if canImport(UIKit)` split). `updateUIView` triggers color swap when `currentFrame` binding changes.
4. `BrainViewerDemo` example app boots, shows the brain, camera presets cycle via a button.

**Gate:** demo app builds and runs in iOS simulator; no warnings.

### Phase 6 — Animation controller (½ day)
1. `BrainAnimationBundle.init(directory:)` parses `brain_anim_meta.json`.
2. `BrainAnimationController`: `CADisplayLink`-throttled tick that increments `currentFrame` at `fps`, with `play/pause/seek/loop` controls.
3. `BrainSceneView` reacts to `currentFrame` changes: resolve URL from bundle pattern (`brain_colors_t%04d.bin`), call `updateColors`.
4. `AnimationBundleTests` covers: meta parse, URL resolution, out-of-range seek behavior.

**Gate:** demo's scrubber slides smoothly across 5 frames.

### Phase 6b — FastAPI mesh endpoints (1 day, server-side, parallel-safe)
1. New module `tribe_backend/api/mesh_routes.py`:
   - `GET /v1/mesh/static?surface=pial` — bootstrap. Cache the four `.bin` + `meta.json` blobs in app state at startup; return as a tar `StreamingResponse` with `ETag` + `Cache-Control: immutable`. Honor `If-None-Match`.
   - `POST /v1/inference/colors` — multipart in (image/audio/text + method), runs TRIBE inference (or a stub for tests), pipes `TribeOutput` through `BrainMeshExporter.aggregate_temporal` → `activation_to_rgba` → returns `brain_colors.bin` bytes with `X-*` metadata headers.
   - `POST /v1/inference/full` — same but envelope JSON with embedded `report` from `GlasserParcellationUnit.generate_report` and a short-TTL `colors_url`.
2. Wire into the existing FastAPI app (the plan in `agent/plans/` for the API already references the parcellation/mesh compartments — extend it, do not fork).
3. Tests under `tests/api/test_mesh_routes.py`:
   - ETag round-trip → `304`.
   - Synthetic `(31, 20484)` input → response body is exactly 81936 bytes.
   - Headers carry `X-Vmin`, `X-Vmax`, `X-Frame-Count`.
   - Multipart with no image → `422`.
4. No GPU, no offscreen GL, no PyVista anywhere in the request path.

**Gate:** `pytest tests/api/test_mesh_routes.py` green; `curl` against a local `uvicorn` returns 80 KB binary.

### Phase 6c — Swift `BrainMeshClient` + polling (1 day)
1. `BrainMeshClient` actor:
   - `bootstrapStaticMesh()` — `GET /v1/mesh/static`, store ETag in `UserDefaults`, untar to `Application Support/TribeBrainView/static/`, return that directory URL.
   - `fetchColors(image:audio:text:method:)` — multipart `URLRequest`, decode response bytes + `X-*` headers into `ColorsResponse`.
   - `fetchFull(...)` — same but parses the envelope and triggers a parallel colors fetch.
2. `BrainMeshClient+Polling.swift` — an `AsyncSequence` helper:
   ```swift
   for try await colors in client.poll(images: imageStream, every: .milliseconds(1000)) { ... }
   ```
   Backpressure: drop in-flight requests if a newer frame arrives (latest-wins).
3. `BrainSceneView` accepts `.remote(client)` source: bootstrap on first appear, then color updates flow through `BrainMeshLoader.updateColors` exactly as in the local-file path.
4. `MeshClientTests` use a custom `URLProtocol` to mock the server: ETag hit/miss, gzip decode, error mapping (4xx/5xx), large-payload streaming.
5. Demo app gains a "Live" toggle: when on, frames from `AVCaptureSession` (or a stubbed image) get POSTed at the configured Hz and the brain updates.

**Gate:** Demo app, pointed at `http://localhost:8000`, animates the brain in response to polled images.

### Phase 7 — Overlay UI target (½ day)
1. `BrainTimelineScrubber` — SwiftUI `Slider` + play/pause button bound to `BrainAnimationController`.
2. `RegionReportCard` — renders a `Report` (Codable struct mirroring `tribe_backend/parcellation/unit.py::Report`); shows top regions with strength label + z-score chip.
3. `ColorbarLegend` — gradient strip annotated with `vmin`/`vmax` from `ColormapMeta`.
4. Snapshot tests on macOS (`pointfreeco/swift-snapshot-testing` is the only dev-dependency; **not** part of the production target).

**Gate:** overlays render correctly in demo's preview canvas.

### Phase 8 — Documentation & embed snippet (½ day)
1. `README.md` with:
   - 3-line install (`.package(path:)` and SPM URL).
   - Minimal embed snippet (`BrainSceneView(source: .binaryDirectory(url))`).
   - "How to generate the input files from Python" pointing at `BrainMeshExporter.export_binary()`.
   - Static vs animation modes.
   - License (Apache-2.0 — **our** repo, not tribev2's CC BY-NC).
2. DocC docs on the public API (Xcode build setting `DOCC_EXTRACT_SPECIAL_DOCS=YES`).

**Gate:** copy-paste snippet from README into a fresh app, it works.

### Phase 9 — Polish & integration smoke (½ day)
1. Run the demo on a real device; profile with Instruments (Time Profiler + Core Animation FPS). Target: 60 fps idle, color-swap < 4 ms on iPhone 15.
2. `swift package diagnose-api-breaking-changes` against tag `0.0.0` to lock public API.
3. Tag `v0.1.0`. Open PR from `wip-3d-map-parser` → `main`.

**Gate:** PR opened, CI green.

**Total estimate:** ~8 working days (Phases 1–9 including server endpoints and networking client). Phase 6b can run in parallel with Phases 5–6 if a second contributor handles the Python side.

---

## 5. Testing Strategy

| Layer | Test type | What's covered |
|---|---|---|
| `BinaryBufferReader` | XCTest unit | malformed inputs, stride mismatches, oversize files |
| `BrainMeshLoader` (binary) | Fixture-driven | counts, semantics, hemisphere split |
| `BrainMeshLoader.updateColors` | Fixture-driven | source identity invariants, throws on size mismatch |
| `BrainAnimationBundle` | Fixture-driven | meta parse, URL pattern resolution |
| `CameraPresets` | Property-style | unit-length up vectors, view ⊥ up where required |
| `BrainSceneView` | Snapshot (UI target) | static render at known camera; pixel-diff tolerance 0.5% |
| `RegionReportCard` | Snapshot | top-3 regions sample, empty state |
| Cross-language | Manual / CI matrix | regenerate fixtures via Python, hash-check against `MANIFEST.json` |
| `BrainMeshClient` | URLProtocol mocks | ETag hit/miss, gzip decode, multipart upload shape, header parsing, latest-wins backpressure |
| FastAPI mesh routes | pytest + httpx | bootstrap tar contents, `304` on `If-None-Match`, exact response byte count, `X-*` headers, multipart 422s |
| End-to-end | One CI job: `uvicorn` + `swift test` | Swift hits a real server on `localhost:8000`, asserts brain updates from a fixture image |

CI: GitHub Actions matrix on macOS-14 runner — `swift test` for both targets, `xcodebuild test` for the demo on iOS Simulator.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SceneKit per-vertex color path is slow on older devices | M | Benchmark Phase 9; if >4 ms, switch the color buffer to a 1D texture lookup driven by activation scalars (would need an `(20484,) float32` companion buffer — Python exporter change). |
| `Data(contentsOf:options: .mappedIfSafe)` falls back to read on iOS sandbox | L | Detected at runtime; fall back to in-memory `Data(contentsOf:)`. Document the perf impact in README. |
| Vertex color rendering looks "flat" without lighting tuning | M | Default `lighting = .physicallyBased` with `roughness = 0.6, metalness = 0.0`. Expose via `BrainViewConfiguration.lighting`. |
| Host app on visionOS wants RealityKit, not SceneKit | M | Phase 9 ships SceneKit only. RealityKit path goes behind a `BrainSceneRealityView` wrapper in a future minor version, sharing `BrainMeshLoader`. |
| GLB ingest via ModelIO produces inverted normals or swapped winding | L | Skip GLB in v0.1.0; binary is canonical. Re-add post-launch with a dedicated test fixture. |
| License entanglement with tribev2 (CC BY-NC) | L | Camera presets and any kNN code are reimplemented from scratch, not copied. README explicitly states upstream provides only Python *reference* — no Swift, no portable mesh export. |
| Static mesh files balloon app bundle | L | 960 KB total; if a host objects, document a remote-fetch pattern in README. |
| Polling at 1+ Hz floods the server | M | Latest-wins cancellation in `BrainMeshClient.poll`; FastAPI side rejects with 429 if a per-client semaphore is held. |
| Inference latency spikes block the UI | M | `BrainMeshClient` is an `actor`; calls are `async`. The scene view never blocks on a request — it just shows the last delivered colors until the next response lands. |
| Host network is unreliable / offline | M | `.binaryDirectory(URL)` fallback always works. Demo app shows graceful degradation: last-known colors stay visible, error banner surfaces from the overlay target. |
| Server-side TRIBE inference is slow on CPU | H (real cost) | Out of scope for this plan, but the contract is shaped so the inference half can be replaced (GPU worker, queue) without touching the Swift client. |
| Multipart upload payload is large (raw frames) | M | Endpoint accepts `image/jpeg` (compressed) by default; document a downscale step on the device before upload. |

---

## 7. What This Plan Explicitly *Does Not* Touch

- `tribe_backend/mesh/exporter.py` and friends. The Python boundary is treated as fixed for this branch. Any exporter changes (e.g., emitting a raw `(N,) float32` activation buffer for shader-side colormap evaluation) would be a separate PR with its own tests.
- `tribe_backend/parcellation/`. The `Report` Codable mirror in the overlay target is a passive consumer of the JSON shape only.
- Live inference / model serving. The Swift module is a renderer, not a client.
- Cross-platform UI (Android, web, desktop Linux). Host the renderer there with a different module if needed.

---

## 8. Out-of-Branch Follow-ups (capture, don't build now)

1. **Subcortical extension.** Port `tribev2/plotting/subcortical.py`'s marching-cubes pipeline into `BrainMeshExporter` to emit per-nucleus closed meshes; Swift module gains a second `BrainMeshSource` kind.
2. **`"half"` inflate mode.** Upstream blends pial/inflated 50/50 (`base.py:127`) — a nice anatomical default. Add as `surface_type="half"` to the Python exporter.
3. **kNN resampler.** Reimplement upstream `get_stat_map` (`base.py:178`) cleanly in `tribe_backend/mesh/resample.py` to support fsaverage5↔6↔7. Swift module is unchanged.
4. **RealityKit renderer** for visionOS — share `BrainMeshLoader`, swap the scene view.
5. **Live IPC / streaming colors.** WebSocket → ring buffer → `updateColors`. Only after a real demand signal.

---

## 9. Definition of Done (Phase 9 exit)

- [ ] `swift test` green on macOS-14 + iOS Simulator (matrix CI).
- [ ] `pytest tests/api/test_mesh_routes.py` green; `uvicorn` boots cleanly with the new routes.
- [ ] Demo app launches, shows static brain (local), animates 5 frames at 1 Hz, camera presets cycle.
- [ ] Demo app "Live" mode polls a local FastAPI dev server and updates the brain from posted images.
- [ ] README copy-paste snippets work for both local-files and remote (FastAPI) embedding.
- [ ] Public API frozen under `swift package diagnose-api-breaking-changes`.
- [ ] License header on every source file (Apache-2.0).
- [ ] Tag `v0.1.0`; PR `wip-3d-map-parser` → `main` open and reviewed.
- [ ] No dependency on tribev2 source. Reference-only documentation citations are fine.

---

## 10. Quick Reference: How a Host App Embeds It

### Local-files mode (offline / pre-baked)

```swift
import SwiftUI
import TribeBrainView
import TribeBrainViewUI   // optional

struct MyBrainScreen: View {
    @StateObject private var anim = try! BrainAnimationController(
        bundle: BrainAnimationBundle(directory: bundleURL)
    )
    var body: some View {
        VStack {
            BrainSceneView(
                source: .binaryDirectory(bundleURL),
                configuration: { var c = BrainViewConfiguration();
                                 c.cameraPreset = .lateralLeft; return c }(),
                currentFrame: $anim.currentFrame,
                animation: anim.bundle
            )
            BrainTimelineScrubber(controller: anim)
        }
    }
}
```

### Live polling mode (FastAPI loop — primary use case)

```swift
import SwiftUI
import TribeBrainView
import TribeBrainViewUI

struct LiveBrainScreen: View {
    let client = BrainMeshClient(baseURL: URL(string: "https://api.cleo.app")!)
    @State private var report: BrainReport?

    var body: some View {
        ZStack(alignment: .bottom) {
            BrainSceneView(source: .remote(client))     // bootstrap + live colors

            if let report { RegionReportCard(report: report) }
        }
        .task {
            // Drive the loop: capture → POST → update colors → render.
            for try await env in client.pollFull(images: cameraStream(),
                                                 every: .milliseconds(1000)) {
                report = env.report                       // colors arrive automatically
            }                                             // via the client → scene view
        }
    }
}
```

Two source kinds, one renderer, no global state. The Swift module doesn't care whether bytes came from disk or HTTP — both paths flow through `BrainMeshLoader`.

That's the whole integration. No global config, no init lifecycle dance, no environment values. Drop it in.
