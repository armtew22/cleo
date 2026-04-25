# TribeBrainView

A pluggable SwiftUI 3D brain renderer for TRIBE v2 cortical activation maps.
Renders the fsaverage5 mesh with per-vertex RGBA colors sourced from local files
or a live FastAPI polling loop.

**Platforms:** iOS 16+, macOS 13+, visionOS 1+. Apple platforms only.
**License:** Apache-2.0 (this repository). The upstream `tribev2` reference code
is CC BY-NC; no code was copied from it.

---

## Install

### Swift Package Manager — local path (monorepo)

```swift
// Package.swift
dependencies: [
    .package(path: "../../swift/TribeBrainView"),   // adjust relative path
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "TribeBrainView",   package: "TribeBrainView"),
        .product(name: "TribeBrainViewUI", package: "TribeBrainView"),  // optional
    ]),
]
```

### Swift Package Manager — remote URL (once published)

```swift
dependencies: [
    .package(url: "https://github.com/your-org/TribeBrainView", from: "0.1.0"),
],
```

---

## Quick start

### Local-files mode (offline / pre-baked)

Drop the four files produced by `BrainMeshExporter.export_binary()` into your
app bundle (or any local directory) and point `BrainSceneView` at them:

```swift
import SwiftUI
import TribeBrainView
import TribeBrainViewUI   // optional overlay target

struct MyBrainScreen: View {
    // bundleURL must contain brain_vertices.bin, brain_normals.bin,
    // brain_faces.bin, brain_colors.bin, brain_meta.json
    let bundleURL: URL

    @StateObject private var anim = try! BrainAnimationController(
        bundle: BrainAnimationBundle(directory: bundleURL)
    )

    var body: some View {
        VStack {
            BrainSceneView(
                source: BrainMeshSource(kind: .binaryDirectory(bundleURL)),
                configuration: {
                    var c = BrainViewConfiguration()
                    c.cameraPreset = .lateralLeft
                    return c
                }(),
                currentFrame: $anim.currentFrame,
                animation: anim.bundle
            )
            BrainTimelineScrubber(controller: anim)  // TribeBrainViewUI
        }
        .onAppear { anim.play() }
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
            // .remote(_) bootstraps the static mesh once (ETag-cached),
            // then color updates flow through BrainMeshLoader.updateColors.
            BrainSceneView(source: BrainMeshSource(kind: .remote(client)))

            if let report { RegionReportCard(report: report) }
        }
        .task {
            // Drive the loop: capture → POST /v1/inference/full → update.
            // Backpressure: latest-wins (older in-flight requests are cancelled).
            for try await env in client.pollFull(
                images: cameraStream(),
                every: .milliseconds(1000),
                method: .windowMean
            ) {
                report = env.report   // colors arrive automatically via the client
            }
        }
    }

    /// Replace with your AVCaptureSession or stubbed image stream.
    func cameraStream() -> AsyncStream<Data> { AsyncStream { _ in } }
}
```

Two source kinds, one renderer, no global state. The module doesn't care whether
bytes came from disk or HTTP — both paths flow through `BrainMeshLoader`.

---

## How to generate input files from Python

The four `.bin` files + `brain_meta.json` are produced by
`tribe_backend/mesh/exporter.py::BrainMeshExporter.export_binary(output, directory)`.

A ready-made script that generates deterministic test fixtures from a synthetic
`TribeOutput` is at `scripts/generate_swift_fixtures.py`. Run it once:

```sh
python scripts/generate_swift_fixtures.py
# Writes to swift/TribeBrainView/Tests/Fixtures/static/ and .../animation/
```

For animation bundles (multi-frame color sequences), use
`BrainMeshExporter.export_animation_bundle(output, directory, fps)`.
The result is consumed by `BrainAnimationBundle(directory: URL)` on the Swift side.

---

## Static vs animation modes

| Mode | Source kind | Controller | Notes |
|---|---|---|---|
| Static single frame | `.binaryDirectory(url)` | none | `brain_colors.bin` in the directory |
| Local animation | `.binaryDirectory(url)` | `BrainAnimationController` | `animation_meta.json` + per-frame `brain_colors_t%04d.bin` |
| Live polling | `.remote(client)` | none needed | `BrainMeshClient.pollFull` drives updates |

`BrainAnimationController` is driven by `CADisplayLink` on iOS/visionOS and a
`Timer` on macOS. Pass explicit `fps` to override the bundle default.

---

## Server contract

The `BrainMeshClient` speaks to three endpoints in
`tribe_backend/api/mesh_routes.py`:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/mesh/static?surface=pial` | One-time bootstrap: returns tar archive with `brain_vertices.bin`, `brain_normals.bin`, `brain_faces.bin`, `brain_meta.json`. ETag-cached (`immutable`). |
| `POST` | `/v1/inference/colors` | Hot path: multipart image/audio/text → `brain_colors.bin` (81 936 bytes, uint8 RGBA × 20484) with `X-Window-Id`, `X-Tribe-Method`, `X-Vmin`, `X-Vmax`, `X-Frame-Count` headers. |
| `POST` | `/v1/inference/full` | Same multipart input; JSON envelope: `window_id`, `colors_url`, `report` (Glasser parcellation), `vmin`, `vmax`, `method`. |

The server is stateless, CPU-only, and requires no GPU or offscreen rendering.
Auth is not implemented in v0.1.0 — add an `Authorization` header via a custom
`URLSession` configuration if needed.

---

## Public API summary

```
TribeBrainView (core target — SceneKit + SwiftUI + Foundation only)

BrainMeshSource / BrainMeshSource.Kind   — where to load from
BrainMeshLoader                          — load() + updateColors(on:from:vertexCount:)
BrainMeshAssets                          — SCNGeometry + counts + meta
BrainMeshClient  (actor)                 — bootstrapStaticMesh() + fetchColors() + fetchFull()
  + poll(images:every:method:)           — AsyncThrowingStream<ColorsResponse>
  + pollFull(images:every:method:)       — AsyncThrowingStream<InferenceEnvelope>
BrainSceneView                           — UIViewRepresentable / NSViewRepresentable
BrainAnimationController                 — CADisplayLink-driven frame ticker
BrainAnimationBundle                     — parses animation_meta.json
BrainViewConfiguration                   — camera preset, lighting, background
CameraPreset / cameraDescriptor(for:)    — 9 anatomical presets (MNI/RAS convention)
AggregationMethod                        — windowMean | peak | peakWindow
ColorsResponse / InferenceEnvelope       — server response types
BrainReport / RegionActivation           — Codable mirror of GlasserParcellationUnit.Report

TribeBrainViewUI (optional overlay target — add only if you want the UI widgets)

BrainTimelineScrubber   — slider + play/pause bound to BrainAnimationController
RegionReportCard        — top-N regions from BrainReport
ColorbarLegend          — gradient strip annotated with vmin/vmax
```

---

## Server connection guide

### Base URL configuration

Construct `BrainMeshClient` with the root URL of your FastAPI server — no path
suffix, no trailing slash:

```swift
// Production
let client = BrainMeshClient(baseURL: URL(string: "https://api.example.com")!)

// Development (uvicorn on localhost)
let client = BrainMeshClient(baseURL: URL(string: "http://localhost:8000")!)
```

The client appends path components (`v1/mesh/static`, `v1/inference/colors`,
etc.) automatically.

**HTTPS in production.** App Transport Security (ATS) requires HTTPS for all
connections in production iOS and macOS apps. For local development against
`http://localhost` you must add an ATS exception in your app's `Info.plist`:

```xml
<!-- Info.plist — development only; remove before App Store submission -->
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
</dict>
```

`NSAllowsLocalNetworking` (iOS 10+) covers `localhost`, `127.0.0.1`, and
`.local` mDNS names without opening up all HTTP traffic. Do **not** use
`NSAllowsArbitraryLoads` unless you have a reviewed justification.

**Cache directory.** The static mesh tar is untarred into Application Support
by default (`~/Library/Application Support/TribeBrainView/static/<surface>/`).
Pass a custom `cacheDirectory` URL to override — useful in app extensions or
unit tests:

```swift
let client = BrainMeshClient(
    baseURL: URL(string: "http://localhost:8000")!,
    cacheDirectory: URL(fileURLWithPath: "/tmp/brain-cache")
)
```

---

### Endpoint reference

All endpoints live under the same FastAPI app. The CORS middleware
(`tribe_backend/api/app.py`) allows all origins, methods, and headers — no
preflight special-casing is needed from Swift. Web/Mac sandbox note: if you
embed a WKWebView calling these endpoints, the browser's CORS enforcement kicks
in independently; the server's `Access-Control-Allow-Origin: *` header satisfies
it.

There is a separate async-job namespace `/v1/runs/{job_id}/mesh/*` for
artifacts produced by the Phase B async job queue (see `tribe_backend/api/routes.py`).
That namespace is distinct from the synchronous mesh routes documented below.

#### GET /v1/mesh/static

One-time bootstrap. The tar archive is built once at server startup and held in
app state; subsequent requests are served from memory.

| Field | Value |
|---|---|
| Method | `GET` |
| Path | `/v1/mesh/static` |
| Query param | `surface` (string, default `"pial"`) |
| Request headers | `If-None-Match: <etag>` (optional; triggers 304) |
| Response status | `200` (full tar), `304` (ETag hit, no body) |
| Response Content-Type | `application/octet-stream` |
| Response body | POSIX tar archive containing `brain_vertices.bin`, `brain_normals.bin`, `brain_faces.bin`, `brain_meta.json` |

Response headers on 200:

| Header | Type | Meaning |
|---|---|---|
| `ETag` | `string` | `"fsaverage5-<surface>-v1"` — version-pinned, changes only on wire format bump |
| `Cache-Control` | `string` | `public, max-age=31536000, immutable` |
| `Content-Disposition` | `string` | `attachment; filename="brain_static_<surface>.tar"` |

#### POST /v1/inference/colors

Hot path. Returns raw `brain_colors.bin` bytes for one frame.

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/v1/inference/colors` |
| Request Content-Type | `multipart/form-data` |
| Response status | `200` (binary), `422` (missing/invalid `image` field) |
| Response Content-Type | `application/octet-stream` |
| Response body | `uint8 RGBA × 20484 = 81 936 bytes` |

Multipart fields:

| Field name | Content-Type | Required | Notes |
|---|---|---|---|
| `image` | `image/jpeg` or `video/mp4` | Yes | The field name must be exactly `image` |
| `audio` | `audio/wav` | No | Optional audio modality |
| `text` | `text/plain` (form field) | No | Optional text modality |
| `method` | form field (string) | No | Aggregation method; default `window_mean` |
| `vmin` | form field (float) | No | Colormap lower bound; default `-3.0` |
| `vmax` | form field (float) | No | Colormap upper bound; default `3.0` |
| `cmap` | form field (string) | No | matplotlib colormap name; default `RdBu_r` |

Response headers on 200:

| Header | Type | Meaning |
|---|---|---|
| `X-Window-Id` | `string` | Opaque ID for this inference window (echoed from `TribeOutput.window_id`) |
| `X-Tribe-Method` | `string` | Aggregation method actually applied (`window_mean`, `peak`, or `peak_window`) |
| `X-Vmin` | `string` (parseable as `Double`) | Lower colormap bound used |
| `X-Vmax` | `string` (parseable as `Double`) | Upper colormap bound used |
| `X-Frame-Count` | `string` (parseable as `Int`) | Number of frames in the body (`1` for the current single-frame path) |

#### POST /v1/inference/full

Same multipart input as `/v1/inference/colors`. Returns a JSON envelope that
combines the colors URL with a qualitative Glasser parcellation report. The
Swift client fetches `colors_url` in a parallel `async let` task immediately
after decoding the envelope.

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/v1/inference/full` |
| Request | identical multipart fields as `/v1/inference/colors` |
| Response status | `200` (JSON), `422` |
| Response Content-Type | `application/json` |

Response body shape:

```json
{
  "window_id": "abc123def456",
  "colors_url": "/v1/inference/colors/abc123def456",
  "report": {
    "top_regions": [
      { "name": "V1", "z_score": 3.2, "direction": "activation", "description": "..." }
    ],
    "text": "Strongest activation in V1 ...",
    "method": "window_mean",
    "z_threshold": 2.0,
    "window_id": "abc123def456"
  },
  "vmin": -3.0,
  "vmax": 3.0,
  "method": "window_mean"
}
```

#### GET /v1/inference/colors/{window_id}

Serve cached colors bytes for a window ID produced by a prior
`POST /v1/inference/full`. The cache has a 5-minute TTL.

| Field | Value |
|---|---|
| Method | `GET` |
| Path | `/v1/inference/colors/{window_id}` |
| Response status | `200` (binary), `404` (not found or expired) |
| Response Content-Type | `application/octet-stream` |
| Response body | Same 81 936 bytes as the hot-path endpoint |

Response headers on 200:

| Header | Type | Meaning |
|---|---|---|
| `X-Window-Id` | `string` | Echoed window ID |
| `Content-Length` | `string` | Byte count of the response body |

---

### ETag protocol

The static mesh ETag has the fixed format `"fsaverage5-<surface>-v1"` (e.g.
`"fsaverage5-pial-v1"`), including the double-quote characters. It is
version-pinned: it changes only when the wire format version is bumped (see
Versioning section).

The Swift client persists the ETag in `UserDefaults` under the key
`TribeBrainView.staticMesh.etag.<surface>` (e.g.
`TribeBrainView.staticMesh.etag.pial`). On subsequent calls to
`bootstrapStaticMesh()`, if this key is set and the local cache directory
contains the required files (`brain_vertices.bin`, `brain_faces.bin`,
`brain_meta.json`), the client sends `If-None-Match: <stored_etag>`. The server
returns `304` and no body; the client returns the cached directory URL
immediately.

If the server returns `304` but the local files are missing (e.g. the user
cleared Application Support), the client deletes the stored ETag key and
retries unconditionally.

---

### AggregationMethod mapping

`AggregationMethod` (Swift enum) maps to the `method` form field string:

| Swift case | Wire string | Server behavior |
|---|---|---|
| `.windowMean` | `window_mean` | Mean over all T timesteps (`np.mean(axis=0)`) |
| `.peak` | `peak` | Value at the timestep with the largest absolute activation, per vertex |
| `.peakWindow` | `peak_window` | Alias for `peak` in the current server implementation |

The method string is echoed back in the `X-Tribe-Method` response header.

---

### vmin / vmax / cmap defaults

If the caller omits `vmin`, `vmax`, or `cmap` from `fetchColors` / `fetchFull`
(passing `nil`), the Swift client does not include those fields in the multipart
body. The server defaults are:

| Parameter | Default |
|---|---|
| `vmin` | `-3.0` |
| `vmax` | `3.0` |
| `cmap` | `RdBu_r` (red-blue diverging; white at center = 0 activation) |

The colormap is baked into the RGBA bytes on the server. The client receives
the `vmin`/`vmax` actually used in the `X-Vmin`/`X-Vmax` headers; use them to
annotate a `ColorbarLegend`.

---

### Error semantics

| Server status | Condition | Swift error |
|---|---|---|
| `304` | ETag hit (not an error) | — client returns cached directory |
| `404` | `window_id` not found or expired in `/v1/inference/colors/{id}` | `BrainMeshClientError.server(status: 404, body: ...)` |
| `422` | Multipart `image` field missing or unreadable | `BrainMeshClientError.server(status: 422, body: ...)` |
| `5xx` | Server-side inference or I/O failure | `BrainMeshClientError.server(status: 5xx, body: ...)` |

`BrainMeshClientError` cases:

| Case | When thrown |
|---|---|
| `.server(status:body:)` | Any non-2xx/304 HTTP response |
| `.invalidResponse(String)` | Response is not `HTTPURLResponse`, or JSON/header decoding fails |
| `.missingHeader(String)` | A required `X-*` header is absent from `/v1/inference/colors` response |
| `.tarMalformed(String)` | The bootstrap tar archive is truncated or structurally invalid |

---

### Polling backpressure model

`BrainMeshClient.poll` and `pollFull` implement a latest-wins model: if a new
image frame arrives while a prior `POST /v1/inference/colors` request is still
in flight, the in-flight task is cancelled and a fresh request is sent for the
newer frame. This prevents unbounded queue growth at slow inference Hz.

Use `poll` when you need only the color buffer (lighter, no JSON parse):

```swift
for try await colors in client.poll(images: imageStream, every: .milliseconds(500)) {
    // colors.buffer: Data (81936 bytes)
    // colors.windowId, .vmin, .vmax, .frameCount
}
```

Use `pollFull` when you also need the `BrainReport` (Glasser parcellation):

```swift
for try await env in client.pollFull(images: imageStream, every: .milliseconds(1000)) {
    // env.colors, env.report
}
```

The `every` interval is the minimum time between successive POSTs, not the
exact cadence — network latency is additive. At 1 Hz with a 200 ms inference
round-trip the brain updates roughly every 1.2 s.

---

### CORS

The server mounts `CORSMiddleware` with `allow_origins=settings.cors_allow_origins`,
`allow_methods=["*"]`, `allow_headers=["*"]`, and `allow_credentials=False`. All
preflight `OPTIONS` requests are handled automatically. For a default development
server the allowed origins come from `Settings.cors_allow_origins`; check
`tribe_backend/api/settings.py` for the default list (typically `["*"]` in dev).

For Mac Catalyst or macOS App Sandbox builds, outbound HTTP to `localhost` is
permitted without entitlements. For connections to a remote server, add
`com.apple.security.network.client` to your `.entitlements` file.

---

### End-to-end live-mode example

The following snippet is copy-pastable. It bootstraps the static mesh once on
appear, then drives the polling loop from `AVCaptureSession` frames.

```swift
import SwiftUI
import AVFoundation
import TribeBrainView
import TribeBrainViewUI

// MARK: - Live Brain Screen

struct LiveBrainScreen: View {
    // Point at your FastAPI server.
    private let client = BrainMeshClient(
        baseURL: URL(string: "http://localhost:8000")!
    )

    @State private var report: BrainReport?
    @State private var errorMessage: String?

    var body: some View {
        ZStack(alignment: .bottom) {
            // BrainSceneView bootstraps the static mesh automatically on first
            // render and applies incoming color buffers as they arrive.
            BrainSceneView(
                source: BrainMeshSource(kind: .remote(client)),
                configuration: {
                    var cfg = BrainViewConfiguration()
                    cfg.cameraPreset = .lateralLeft
                    return cfg
                }()
            )

            VStack(spacing: 8) {
                if let errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding(8)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                }
                if let report {
                    RegionReportCard(report: report)   // TribeBrainViewUI
                }
            }
            .padding()
        }
        .task {
            await runPollingLoop()
        }
    }

    private func runPollingLoop() async {
        do {
            for try await env in client.pollFull(
                images: captureStream(),
                every: .milliseconds(1000),
                method: .windowMean
            ) {
                await MainActor.run {
                    report = env.report
                    errorMessage = nil
                }
                // env.colors is applied to the scene view automatically by the
                // .remote source integration inside BrainSceneCoordinator.
            }
        } catch {
            await MainActor.run {
                errorMessage = error.localizedDescription
            }
        }
    }
}

// MARK: - Camera capture helper

private func captureStream() -> AsyncStream<Data> {
    // Replace the stub below with a real AVCaptureSession integration.
    // The stream emits JPEG-encoded Data for each camera frame you want to POST.
    AsyncStream { continuation in
        // Example: emit a stubbed JPEG every second.
        Task {
            let stubImage = UIImage(systemName: "brain")!
            let jpegData = stubImage.jpegData(compressionQuality: 0.8)!
            while !Task.isCancelled {
                continuation.yield(jpegData)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            continuation.finish()
        }
    }
}
```

For a real AVCaptureSession, set up an `AVCaptureVideoDataOutput` delegate that
JPEG-compresses `CMSampleBuffer` frames and feeds them into the continuation.
Keep the compression quality at 0.7–0.8 to limit upload size; the server accepts
any valid JPEG.

---

## Local artifact parsing and rendering

### Artifact directory layout

`BrainMeshLoader.loadBinary(_:)` expects a directory containing these files, read
in the following order:

| File | Size | Required | Description |
|---|---|---|---|
| `brain_meta.json` | variable | Yes | JSON metadata (see field reference below) |
| `brain_vertices.bin` | `vertex_count × 12` bytes | Yes | 3 × `float32` XYZ per vertex (240 KB for 20484 vertices) |
| `brain_faces.bin` | `face_count × 12` bytes | Yes | 3 × `int32` indices per triangle (480 KB for 40960 faces) |
| `brain_colors.bin` | `vertex_count × 4` bytes | Yes | RGBA `uint8` per vertex (80 KB for 20484 vertices) |
| `brain_normals.bin` | `vertex_count × 12` bytes | No | 3 × `float32` per vertex; optional — SceneKit auto-shades if absent |

Exact byte counts for fsaverage5 `pial` surface (20484 vertices, 40960 faces):

| File | Exact bytes |
|---|---|
| `brain_vertices.bin` | 245 808 (20484 × 12) |
| `brain_faces.bin` | 491 520 (40960 × 12) |
| `brain_colors.bin` | 81 936 (20484 × 4) |
| `brain_normals.bin` | 245 808 (20484 × 12) — not currently emitted by exporter |

---

### brain_meta.json field reference

Produced by `BrainMeshExporter.export_binary()`. The Swift `ColormapMeta`
Codable struct maps every field via `CodingKeys` aliases.

| JSON key | Swift property | Type | Required by loader | Emitted by exporter today | Notes |
|---|---|---|---|---|---|
| `format` | `format` | `String` | No (informational) | Yes | Always `"tribe_brain_mesh_v1"` |
| `surface` | `surface` | `String?` | No | Yes (`"fsaverage5"`) | Absent from server bootstrap tar |
| `surface_type` | `surfaceType` | `String` | Yes (loader reads it) | Yes (`"pial"`) | |
| `vertex_count` | `vertexCount` | `Int` | Yes | Yes (`20484`) | |
| `face_count` | `faceCount` | `Int` | Yes | Yes (`40960`) | |
| `bytes_per_vertex` | `bytesPerVertex` | `Int` | Yes (validated) | Yes (`12`) | Loader throws `metaMismatch` if != 12 |
| `bytes_per_face` | `bytesPerFace` | `Int` | Yes (validated) | Yes (`12`) | Loader throws `metaMismatch` if != 12 |
| `bytes_per_color` | `bytesPerColor` | `Int` | Yes (validated) | Yes (`4`) | Loader throws `metaMismatch` if != 4 |
| `files` | `files` | `Files` | Yes | Yes | Object with `vertices`, `faces`, `colors` string keys |
| `files.vertices` | `files.vertices` | `String` | Yes | Yes (`"brain_vertices.bin"`) | |
| `files.faces` | `files.faces` | `String` | Yes | Yes (`"brain_faces.bin"`) | |
| `files.colors` | `files.colors` | `String?` | No | Yes (`"brain_colors.bin"`) | Absent from server bootstrap tar |
| `files.normals` | `files.normals` | `String?` | No | No (not yet emitted) | Reserved |
| `window_id` | `windowId` | `String?` | No | Yes (`"fixture-seed42"` from script) | |
| `vmin` | `vmin` | `Double?` | No | No | Forward-compatible; colormap baked into RGBA today |
| `vmax` | `vmax` | `Double?` | No | No | Forward-compatible |
| `colormap_name` | `colormapName` | `String?` | No | No | Forward-compatible |

---

### SCNGeometrySource construction recipe

`BrainMeshLoader.loadBinary` assembles sources as follows (line numbers refer to
`BrainMeshLoader.swift`):

**Vertices** (line ~160):

```swift
SCNGeometrySource(
    data: verticesData,          // Data(contentsOf:, options: .mappedIfSafe)
    semantic: .vertex,
    vectorCount: vertexCount,    // meta.vertexCount
    usesFloatComponents: true,
    componentsPerVector: 3,      // X, Y, Z
    bytesPerComponent: 4,        // float32
    dataOffset: 0,
    dataStride: 12               // bytesPerVertex = 3 × 4
)
```

**Normals** (line ~182, only if `brain_normals.bin` exists):

```swift
SCNGeometrySource(
    data: normalsData,
    semantic: .normal,
    vectorCount: vertexCount,
    usesFloatComponents: true,
    componentsPerVector: 3,
    bytesPerComponent: 4,
    dataOffset: 0,
    dataStride: 12               // bytesPerNormal = 3 × 4
)
```

**Colors** (line ~194):

```swift
SCNGeometrySource(
    data: colorsData,
    semantic: .color,
    vectorCount: vertexCount,
    usesFloatComponents: false,  // uint8
    componentsPerVector: 4,      // R, G, B, A
    bytesPerComponent: 1,
    dataOffset: 0,
    dataStride: 4                // bytesPerColor = 4 × 1
)
```

**Element** (line ~207):

```swift
SCNGeometryElement(
    data: facesData,
    primitiveType: .triangles,
    primitiveCount: faceCount,   // meta.faceCount
    bytesPerIndex: 4             // int32
)
```

`SCNGeometry.sources` and `.elements` are immutable after construction. The
`updateColors(on:from:vertexCount:)` method creates a **new** `SCNGeometry`
object that reuses the existing vertex and normal `SCNGeometrySource` instances
by identity (reference equality) and replaces only the `.color` source. The
`BrainSceneCoordinator` then swaps the geometry on the `SCNNode`. This means
vertex and normal memory is never reallocated on a color update.

---

### Generating artifacts offline

```sh
# From the repo root. Requires the repo's Python environment (numpy, nibabel,
# nilearn, matplotlib — all listed in pyproject.toml).
python scripts/generate_swift_fixtures.py

# Optional: write to a custom directory
python scripts/generate_swift_fixtures.py --out-dir /path/to/output
```

The script uses seed `42` and `TribeOutput` shape `(31, 20484) float32`. Output:

```
<out>/static/           — brain_vertices.bin, brain_faces.bin, brain_colors.bin, brain_meta.json
<out>/animation/        — brain_colors_t0000.bin … brain_colors_t0030.bin + animation_meta.json
<out>/MANIFEST.json     — SHA-256 + byte size for every file
```

To bundle the generated files into a host app's Resources:

**SPM resource target:**

```swift
// In your host Package.swift target
.target(
    name: "MyApp",
    dependencies: [ ... ],
    resources: [
        .copy("Resources/BrainFixtures")   // folder containing static/ and animation/
    ]
)
```

**Xcode app target:** Add the generated directory to the app target's "Copy
Bundle Resources" build phase. Access at runtime via:

```swift
let staticDir = Bundle.main.url(forResource: "static", withExtension: nil,
                                subdirectory: "BrainFixtures")!
```

---

### Animation bundle

`BrainAnimationBundle(directory:)` reads `animation_meta.json` (falling back to
`brain_anim_meta.json` if the primary name is absent).

`animation_meta.json` schema:

| Key | Type | Required | Description |
|---|---|---|---|
| `format` | `String?` | No | `"tribe_brain_mesh_v1"` if present |
| `kind` | `String?` | No | `"animation_bundle"` if present |
| `vertex_count` | `Int` | Yes | Must match the static mesh vertex count |
| `frame_count` | `Int` | Yes | Must equal `frames.count` |
| `bytes_per_color` | `Int` | Yes | Always `4` |
| `frames` | `[String]` | Yes | Ordered list of per-frame `.bin` filenames |
| `fps` | `Double?` | No | Not emitted today; `BrainAnimationBundle.defaultFPS = 10.0` used if absent |

`frameURL(at:)` returns the URL of the `.bin` file for a given frame index. Out-of-range
indices are **clamped** to `[0, frameCount-1]` — no crash, no throw. Callers that
prefer an error for out-of-range access should range-check before calling.

---

### Color-swap sequence

```
1. Caller invokes BrainMeshLoader.updateColors(on: geometry, from: colorURL, vertexCount: N)
2. Data(contentsOf: colorURL, options: .mappedIfSafe) → colorsData  [81936 bytes]
3. Validate: colorsData.count == N × 4  (throws sizeMismatch if not)
4. Build new SCNGeometrySource(data: colorsData, semantic: .color, ...)
5. Iterate geometry.sources:
      .color source  → replace with new source
      all other sources → pass through unchanged (vertex/normal identity preserved)
6. If no .color source found → append the new one
7. SCNGeometry(sources: rebuiltSources, elements: geometry.elements)  [elements reused]
8. Copy materials and name from original geometry
9. Return new SCNGeometry
10. BrainSceneCoordinator.swapColors(new geometry) → set on SCNNode.geometry
```

Memory note: `Data(contentsOf:options: .mappedIfSafe)` memory-maps the file on
iOS/macOS where permitted. In the iOS app sandbox, mapping sometimes falls back to
an in-memory read; this is transparent to the caller but means peak RSS can spike
by ~82 KB during the swap.

---

### Camera preset reference

Coordinate convention: MNI/RAS — `+X` = subject's right, `+Y` = anterior,
`+Z` = superior. `viewVector` points from the origin toward the camera; the
camera sits at `viewVector × brainCameraRadius` (320 mm) looking back at
the origin. All presets use `fov = 30.0°`.

| Preset | `viewVector` | `upVector` | Anatomical view |
|---|---|---|---|
| `.lateralLeft` | `(-1, 0, 0)` | `(0, 0, 1)` | Left hemisphere lateral surface |
| `.lateralRight` | `(1, 0, 0)` | `(0, 0, 1)` | Right hemisphere lateral surface |
| `.medialLeft` | `(1, 0, 0)` | `(0, 0, 1)` | Left hemisphere medial surface (from right) |
| `.medialRight` | `(-1, 0, 0)` | `(0, 0, 1)` | Right hemisphere medial surface (from left) |
| `.dorsal` | `(0, 0, 1)` | `(0, 1, 0)` | Top-down; anterior toward top of frame |
| `.ventral` | `(0, 0, -1)` | `(0, -1, 0)` | Bottom-up; anterior toward top of frame |
| `.anterior` | `(0, 1, 0)` | `(0, 0, 1)` | Frontal (face-on) |
| `.posterior` | `(0, -1, 0)` | `(0, 0, 1)` | Occipital |
| `.frontalThreeQuarter` | `(s, s, s)` where `s=1/√3` | `(0, 0, 1)` | Oblique 45° from anterior-right-superior |
| `.custom` | `(0, 0, 0)` (identity placeholder) | `(0, 0, 1)` | Host sets camera transform directly |

Use `cameraDescriptor(for:)` to retrieve the `CameraPresetDescriptor`
(viewVector, upVector, fov) for a preset and apply it in `BrainSceneCoordinator`.

---

### BrainViewConfiguration fields

| Field | Type | Default | Effect |
|---|---|---|---|
| `cameraPreset` | `CameraPreset` | `.lateralLeft` | Initial camera orientation |
| `background` | `Color` | `.clear` | SceneKit background; `.clear` inherits the SwiftUI background |
| `lighting` | `LightingMode` | `.physicallyBased` | Scene lighting mode |
| `allowsCameraControl` | `Bool` | `true` | Enables pinch-to-zoom and drag-to-orbit in SCNView |
| `smoothShading` | `Bool` | `true` | Per-vertex smooth shading vs flat shading |

---

### Bundling fixtures in a host app

**SPM monorepo pattern** — symlink `Tests/Fixtures` into your demo target:

```swift
// Package.swift
.executableTarget(
    name: "BrainViewerDemo",
    dependencies: [
        .product(name: "TribeBrainView", package: "TribeBrainView"),
        .product(name: "TribeBrainViewUI", package: "TribeBrainView"),
    ],
    resources: [
        .copy("BundledFixtures")   // symlink → ../../Tests/Fixtures
    ]
)
```

Then in code:

```swift
let fixturesDir = Bundle.main.url(forResource: "BundledFixtures",
                                  withExtension: nil)!
let staticDir   = fixturesDir.appendingPathComponent("static")
let animDir     = fixturesDir.appendingPathComponent("animation")
```

**Xcode app target** — add the directory in the "Copy Bundle Resources" build
phase and use `Bundle.main.url(forResource:withExtension:subdirectory:)`.

---

### Offline-mode end-to-end example

```swift
import SwiftUI
import TribeBrainView
import TribeBrainViewUI

struct OfflineBrainScreen: View {
    // Directory produced by scripts/generate_swift_fixtures.py --out-dir ...
    // and bundled into the app's Resources.
    private let staticDir: URL = {
        Bundle.main.url(forResource: "static", withExtension: nil,
                        subdirectory: "BrainFixtures")!
    }()
    private let animDir: URL = {
        Bundle.main.url(forResource: "animation", withExtension: nil,
                        subdirectory: "BrainFixtures")!
    }()

    @StateObject private var controller: BrainAnimationController = {
        let bundle = try! BrainAnimationBundle(directory:
            Bundle.main.url(forResource: "animation", withExtension: nil,
                            subdirectory: "BrainFixtures")!
        )
        return BrainAnimationController(bundle: bundle, fps: 4.0)
    }()

    var body: some View {
        VStack(spacing: 0) {
            BrainSceneView(
                source: BrainMeshSource(kind: .binaryDirectory(staticDir)),
                configuration: {
                    var cfg = BrainViewConfiguration()
                    cfg.cameraPreset = .dorsal
                    cfg.allowsCameraControl = true
                    return cfg
                }(),
                currentFrame: $controller.currentFrame,
                animation: controller.bundle
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Optional: timeline scrubber from TribeBrainViewUI
            BrainTimelineScrubber(controller: controller)
                .padding()
        }
        .onAppear { controller.play() }
    }
}
```

---

## Troubleshooting

**"vertex_count mismatch in meta"**
The fixtures directory was produced by a different version of the exporter
(`BrainMeshExporter`) than the Swift loader expects. Re-run
`python scripts/generate_swift_fixtures.py` to regenerate consistent artifacts.

**304 storm without bytes / cache appears empty**
The local cache directory under Application Support is missing the required
`.bin` files but UserDefaults still holds a stale ETag. The client detects this
and retries automatically (`bootstrapStaticMesh` calls itself recursively after
clearing the ETag key). If the loop persists, delete the stored key manually:

```swift
UserDefaults.standard.removeObject(forKey: "TribeBrainView.staticMesh.etag.pial")
```

**Multipart 422 Unprocessable Entity**
The `image` field name in your multipart body is wrong. The server requires
exactly the field name `image` (lowercase). `BrainMeshClient.fetchColors` and
`fetchFull` set this correctly; if you are building a custom `URLRequest`,
double-check the `Content-Disposition` part header.

**Colors look gray / brain shows uniform color**
The first `fetchColors` response has not been applied yet. `BrainSceneView` in
`.remote` mode renders the static mesh geometry immediately after bootstrap but
with placeholder colors (the first `brain_colors.bin` from the cached static
directory) until the first inference response lands. This is expected behavior;
the placeholder is replaced within one polling interval.

**Normals missing — lighting looks flat**
`brain_normals.bin` is not emitted by the current `BrainMeshExporter.export_binary()`
implementation. The loader falls back to SceneKit's automatic normal computation
from triangle geometry. The visual result is correct but may differ subtly from
a pre-computed smooth-normal shading. This is a known open item; when the
exporter begins emitting normals, the loader will use them automatically without
requiring any Swift-side changes.

---

## Versioning and compatibility

### Wire format version

The binary format version constant is `"tribe_brain_mesh_v1"` (defined as
`MESH_FORMAT_VERSION` in `tribe_backend/mesh/exporter.py`). It appears in:
- `brain_meta.json` → `format` field
- The static mesh ETag: `"fsaverage5-<surface>-v1"`
- `animation_meta.json` → `format` field (when present)

**Bump policy:** increment the version string (e.g. to `tribe_brain_mesh_v2`) and
update the ETag template whenever the binary layout, field semantics, or required
file set changes in a backward-incompatible way. The Swift loader validates
`bytes_per_vertex`, `bytes_per_face`, and `bytes_per_color` against hard-coded
constants and throws `BrainMeshError.metaMismatch` on any mismatch.

### Platform minimums

| Platform | Minimum |
|---|---|
| iOS | 16.0 |
| macOS | 13.0 |
| visionOS | 1.0 |

### Render backend

v0.1.0 uses SceneKit exclusively (`SCNGeometry`, `SCNView`). A RealityKit
renderer for visionOS (sharing `BrainMeshLoader` geometry construction) is
tracked as a follow-up in §8 of `agent/plans/swift-brain-view-plan.md` and is
not part of this release.
