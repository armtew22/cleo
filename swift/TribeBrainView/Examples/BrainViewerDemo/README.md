# BrainViewerDemo

A SwiftUI iOS example app that demonstrates all three embedding scenarios for
`TribeBrainView`. Full integration documentation is in
[`../../README.md`](../../README.md).

## Scenarios

1. **Local fixture (offline).** Loads `brain_vertices.bin`, `brain_faces.bin`,
   `brain_colors.bin`, and `brain_meta.json` from `BundledFixtures/static/`
   (a copy of `Tests/Fixtures/static/` bundled via SPM `.copy("BundledFixtures")`).
   No network required. Demonstrates `BrainMeshSource(kind: .binaryDirectory(url))`.

2. **Remote bootstrap.** Calls `GET /v1/mesh/static` against a local
   `uvicorn` server on `http://localhost:8000`, caches the tar archive,
   and renders the static mesh via `BrainMeshSource(kind: .remote(client))`.
   Demonstrates `BrainMeshClient.bootstrapStaticMesh()` and the ETag cache.

3. **Live polling toggle.** A toolbar toggle switches between offline (scenario 1)
   and live mode. In live mode, a stubbed `AsyncStream<Data>` (replaceable with a
   real `AVCaptureSession`) drives `client.pollFull(images:every:)` at 1 Hz,
   updating the mesh color and showing a `RegionReportCard` overlay from
   `TribeBrainViewUI`.

## Running

```sh
# Start the FastAPI server (scenarios 2 and 3)
uvicorn tribe_backend.api.app:get_app --factory --port 8000

# Build the demo (requires macOS with Xcode 15+ and Swift 5.9+)
cd swift/TribeBrainView
swift build
# or open the package in Xcode and run on an iOS 16+ simulator
```

For generating fresh fixtures used by scenario 1:

```sh
python scripts/generate_swift_fixtures.py
```

See the main [`README.md`](../../README.md) for complete field references,
endpoint documentation, and copy-pastable snippets.
