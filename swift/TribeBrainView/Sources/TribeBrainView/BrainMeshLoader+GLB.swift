// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

#if canImport(ModelIO)
import ModelIO
import SceneKit

// MARK: - GLB loading path (gated behind ModelIO availability)

extension BrainMeshLoader {
    /// Load a mesh from a GLB/GLTF file using ModelIO.
    /// This is a thin compatibility shim; binary is the canonical format.
    public static func loadGLB(_ url: URL) throws -> BrainMeshAssets {
        fatalError("unimplemented")
    }
}
#endif
