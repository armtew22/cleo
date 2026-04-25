// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import TribeBrainView

/// Minimal demo screen. Points at a fixture directory shipped alongside the
/// Examples target — not expected to run as-is on Linux; the path layout is
/// designed for the Xcode/macOS demo target where the bundle is structured.
///
/// "Live" toggle (Phase 6c): switches between a local `.binaryDirectory`
/// fixture source and a `.remote(client)` source pointed at a local FastAPI
/// dev server (http://localhost:8000). When live, a "Send synthetic frame"
/// button POSTs a tiny stub JPEG via `BrainMeshClient.fetchColors` and pushes
/// the resulting buffer through the coordinator. AVCaptureSession wiring is
/// intentionally left as a TODO for a real device — this demo proves the loop
/// without device permissions or camera plumbing.
struct ContentView: View {
    @State private var presetIndex: Int = 0
    @State private var configuration: BrainViewConfiguration = {
        var c = BrainViewConfiguration()
        c.cameraPreset = .lateralLeft
        return c
    }()
    @State private var currentFrame: Int = 0
    @State private var liveMode: Bool = false
    @State private var liveStatus: String = "idle"

    private let liveClient = BrainMeshClient(
        baseURL: URL(string: "http://localhost:8000")!
    )

    private static let presets: [CameraPreset] = [
        .lateralLeft, .lateralRight, .medialLeft, .medialRight,
        .dorsal, .ventral, .anterior, .posterior, .frontalThreeQuarter,
    ]

    private var fixtureDirectory: URL {
        Bundle.main.resourceURL?
            .appendingPathComponent("BundledFixtures/static")
        ?? URL(fileURLWithPath: "/tmp/missing-brain-fixtures")
    }

    private var animationDirectory: URL {
        Bundle.main.resourceURL?
            .appendingPathComponent("BundledFixtures/animation")
        ?? URL(fileURLWithPath: "/tmp/missing-brain-fixtures")
    }

    private var source: BrainMeshSource {
        if liveMode {
            return BrainMeshSource(kind: .remote(liveClient))
        } else {
            return BrainMeshSource(kind: .binaryDirectory(fixtureDirectory))
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            BrainSceneView(
                source: source,
                configuration: configuration,
                currentFrame: liveMode ? nil : $currentFrame,
                animation: liveMode
                    ? nil
                    : (try? BrainAnimationBundle(directory: animationDirectory))
            )

            HStack {
                Toggle("Live", isOn: $liveMode)
                    .toggleStyle(.switch)
                Spacer()
                if liveMode {
                    Button("Send synthetic frame") {
                        Task { await sendSyntheticFrame() }
                    }
                    Text(liveStatus).font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)
            .padding(.top, 8)

            HStack {
                Button("Prev preset") { cyclePreset(by: -1) }
                Spacer()
                Text(Self.presets[presetIndex].rawValue).font(.caption)
                Spacer()
                Button("Next preset") { cyclePreset(by: 1) }
            }
            .padding()
        }
    }

    private func cyclePreset(by delta: Int) {
        let n = Self.presets.count
        presetIndex = ((presetIndex + delta) % n + n) % n
        configuration.cameraPreset = Self.presets[presetIndex]
    }

    /// Posts a 3-byte JPEG stub (0xFF 0xD8 0xFF — JPEG magic prefix; server
    /// treats the bytes as opaque and only hashes them in stub-inference mode).
    /// Real device capture would replace this with an AVCaptureSession sample
    /// buffer encoded as JPEG via VTCompressionSession; intentionally stubbed.
    private func sendSyntheticFrame() async {
        liveStatus = "posting…"
        do {
            let stub = Data([0xFF, 0xD8, 0xFF])
            let r = try await liveClient.fetchColors(image: stub)
            liveStatus = "got \(r.buffer.count) bytes (\(r.windowId.prefix(6)))"
        } catch {
            liveStatus = "error: \(error)"
        }
    }
}
