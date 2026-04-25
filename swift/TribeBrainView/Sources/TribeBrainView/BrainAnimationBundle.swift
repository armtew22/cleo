// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainAnimationBundle

public struct BrainAnimationBundle {
    public let directory: URL
    public let frameCount: Int
    public let fps: Double

    /// Reads brain_anim_meta.json from `directory` to determine frame count and fps.
    public init(directory: URL) throws {
        self.directory = directory
        // Phase 6 will parse brain_anim_meta.json here.
        self.frameCount = 0
        self.fps = 1.0
    }

    /// Returns the URL for the color buffer at the given frame index.
    public func colorURL(for frame: Int) -> URL {
        fatalError("unimplemented")
    }
}

// MARK: - BrainAnimMeta (Codable mirror of brain_anim_meta.json)

public struct BrainAnimMeta: Codable {
    public let frameCount: Int
    public let fps: Double
    public let framePattern: String   // e.g. "brain_colors_t%04d.bin"
    public let vertexCount: Int

    private enum CodingKeys: String, CodingKey {
        case frameCount = "frame_count"
        case fps
        case framePattern = "frame_pattern"
        case vertexCount = "vertex_count"
    }
}
