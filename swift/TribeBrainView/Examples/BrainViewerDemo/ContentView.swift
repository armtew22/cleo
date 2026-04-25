// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import TribeBrainView

/// Minimal demo screen. Points at a fixture directory shipped alongside the
/// Examples target — not expected to run as-is on Linux; the path layout is
/// designed for the Xcode/macOS demo target where the bundle is structured.
struct ContentView: View {
    @State private var presetIndex: Int = 0
    @State private var configuration: BrainViewConfiguration = {
        var c = BrainViewConfiguration()
        c.cameraPreset = .lateralLeft
        return c
    }()
    @State private var currentFrame: Int = 0

    private static let presets: [CameraPreset] = [
        .lateralLeft, .lateralRight, .medialLeft, .medialRight,
        .dorsal, .ventral, .anterior, .posterior, .frontalThreeQuarter,
    ]

    private var fixtureDirectory: URL {
        // Demo assumes the fixture dir is bundled relative to the app's main
        // bundle. Swap this for a real path when wiring into a host app.
        Bundle.main.resourceURL?
            .appendingPathComponent("BundledFixtures/static")
        ?? URL(fileURLWithPath: "/tmp/missing-brain-fixtures")
    }

    private var animationDirectory: URL {
        Bundle.main.resourceURL?
            .appendingPathComponent("BundledFixtures/animation")
        ?? URL(fileURLWithPath: "/tmp/missing-brain-fixtures")
    }

    var body: some View {
        VStack(spacing: 0) {
            BrainSceneView(
                source: .init(kind: .binaryDirectory(fixtureDirectory)),
                configuration: configuration,
                currentFrame: $currentFrame,
                animation: try? BrainAnimationBundle(directory: animationDirectory)
            )

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
}
