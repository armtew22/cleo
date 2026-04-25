// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - BrainMeshError

public enum BrainMeshError: Error {
    case malformed(String)
    case missingFile(String)
    case sizeMismatch(expected: Int, actual: Int)
    case unsupportedFormat
}

// MARK: - BrainMeshLoader

public enum BrainMeshLoader {
    /// Load a complete mesh from a binary-buffer directory or other source.
    public static func load(_ source: BrainMeshSource) throws -> BrainMeshAssets {
        fatalError("unimplemented")
    }

    /// Replace only the per-vertex color source on an existing geometry.
    /// Vertex, normal, and face sources are reused without reallocation.
    public static func updateColors(
        on geometry: SCNGeometry,
        from url: URL,
        vertexCount: Int
    ) throws {
        fatalError("unimplemented")
    }
}
