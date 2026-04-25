// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainAnimationController

public final class BrainAnimationController: ObservableObject {
    @Published public var currentFrame: Int = 0

    public let bundle: BrainAnimationBundle
    public let fps: Double

    public init(bundle: BrainAnimationBundle, fps: Double = 1.0) {
        self.bundle = bundle
        self.fps = fps
    }

    public func play() {
        fatalError("unimplemented")
    }

    public func pause() {
        fatalError("unimplemented")
    }

    public func seek(to frame: Int) {
        fatalError("unimplemented")
    }
}
