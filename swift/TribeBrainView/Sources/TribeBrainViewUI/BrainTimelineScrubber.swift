// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import TribeBrainView

// MARK: - BrainTimelineScrubber

/// A compact horizontal scrubber bound to a `BrainAnimationController`.
///
/// Displays a slider spanning all frames, a play/pause toggle, and a
/// "current / total" frame counter.
public struct BrainTimelineScrubber: View {
    @ObservedObject public var controller: BrainAnimationController
    @State private var isPlaying: Bool = false

    public init(controller: BrainAnimationController) {
        self.controller = controller
    }

    private var frameCount: Int {
        controller.bundle.frameCount
    }

    /// Slider range upper bound — clamped to at least 1 to avoid a
    /// zero-length range when frameCount == 1.
    private var sliderMax: Double {
        Double(max(frameCount - 1, 1))
    }

    private var sliderBinding: Binding<Double> {
        Binding(
            get: { Double(controller.currentFrame) },
            set: { controller.seek(to: Int($0.rounded())) }
        )
    }

    public var body: some View {
        HStack(spacing: 8) {
            Button {
                isPlaying.toggle()
                if isPlaying {
                    controller.play()
                } else {
                    controller.pause()
                }
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 20, height: 20)
            }
            .buttonStyle(.borderless)

            Slider(
                value: sliderBinding,
                in: 0...sliderMax,
                step: 1
            )

            Text("\(controller.currentFrame) / \(max(frameCount - 1, 0))")
                .monospacedDigit()
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize()
        }
        .padding(.horizontal)
    }
}
