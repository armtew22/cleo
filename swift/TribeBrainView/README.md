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
