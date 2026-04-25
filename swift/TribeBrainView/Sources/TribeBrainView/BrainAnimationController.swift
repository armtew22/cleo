// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import Combine
import QuartzCore

#if canImport(UIKit)
import UIKit
#endif

// MARK: - BrainAnimationController

/// Drives `currentFrame` over time at `fps`. UI hosts bind to `currentFrame`
/// and pass it into `BrainSceneView`.
///
/// Constructor `fps` is **authoritative**: if the caller passes an explicit
/// non-NaN value, that wins, regardless of whether `bundle.fps` differs.
/// Pass `Double.nan` to fall back to `bundle.fps` (or the bundle default).
public final class BrainAnimationController: ObservableObject {
    @Published public var currentFrame: Int = 0

    public let bundle: BrainAnimationBundle
    /// Effective playback rate (frames per second). > 0.
    public let fps: Double
    public var loops: Bool = true

    public private(set) var isPlaying: Bool = false

    // Tick driver
    #if canImport(UIKit)
    private var displayLink: CADisplayLink?
    #else
    private var timer: Timer?
    #endif
    private var lastTickTime: CFTimeInterval = 0
    private var accumulator: Double = 0

    public init(bundle: BrainAnimationBundle, fps: Double = 1.0) {
        self.bundle = bundle
        // Resolution: explicit param wins unless caller passed `.nan`.
        if fps.isNaN || fps <= 0 {
            self.fps = bundle.fps > 0 ? bundle.fps : BrainAnimationBundle.defaultFPS
        } else {
            self.fps = fps
        }
    }

    deinit {
        invalidateTicker()
    }

    // MARK: - Controls

    public func play() {
        guard !isPlaying else { return }
        isPlaying = true
        lastTickTime = CACurrentMediaTime()
        accumulator = 0
        installTicker()
    }

    public func pause() {
        guard isPlaying else { return }
        isPlaying = false
        invalidateTicker()
    }

    public func seek(to frame: Int) {
        let clamped = max(0, min(frame, bundle.frameCount - 1))
        currentFrame = clamped
        accumulator = 0
    }

    // MARK: - Ticker

    private func installTicker() {
        #if canImport(UIKit)
        let link = CADisplayLink(target: self, selector: #selector(tick))
        link.add(to: .main, forMode: .common)
        self.displayLink = link
        #else
        // Macs without UIKit: drive at ~60 Hz via Timer; the time-accumulator
        // logic below decouples display refresh from animation fps.
        let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        timer.tolerance = 0.005
        self.timer = timer
        #endif
    }

    private func invalidateTicker() {
        #if canImport(UIKit)
        displayLink?.invalidate()
        displayLink = nil
        #else
        timer?.invalidate()
        timer = nil
        #endif
    }

    @objc private func tick() {
        let now = CACurrentMediaTime()
        let dt = now - lastTickTime
        lastTickTime = now
        accumulator += dt

        let secondsPerFrame = 1.0 / fps
        while accumulator >= secondsPerFrame {
            accumulator -= secondsPerFrame
            advanceOneFrame()
        }
    }

    private func advanceOneFrame() {
        let next = currentFrame + 1
        if next >= bundle.frameCount {
            if loops {
                currentFrame = 0
            } else {
                currentFrame = bundle.frameCount - 1
                pause()
            }
        } else {
            currentFrame = next
        }
    }
}
