// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - BrainSceneCoordinator

/// Owns the SCNView lifecycle: scene setup, lighting, camera, and color-swap on frame changes.
public final class BrainSceneCoordinator: NSObject {
    public let source: BrainMeshSource
    public let configuration: BrainViewConfiguration

    private(set) public var assets: BrainMeshAssets?
    private(set) public var scene: SCNScene?

    public init(source: BrainMeshSource, configuration: BrainViewConfiguration) {
        self.source = source
        self.configuration = configuration
    }

    /// Set up the scene on the SCNView after it is created.
    public func configure(_ scnView: SCNView) {
        fatalError("unimplemented")
    }

    /// Swap the color source to reflect a new frame index.
    public func applyFrame(_ frame: Int, bundle: BrainAnimationBundle) {
        fatalError("unimplemented")
    }
}
