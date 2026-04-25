// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - SCNGeometry vertex-color helpers (internal)

extension SCNGeometry {
    /// Build a color SCNGeometrySource from raw RGBA uint8 data.
    /// Stride: 4 bytes (R, G, B, A). Component count: 4. Bytes per component: 1.
    static func colorSource(from data: Data, vertexCount: Int) -> SCNGeometrySource {
        fatalError("unimplemented")
    }

    /// Replace the existing .color source, reusing all other sources and the element.
    func replacingColorSource(with newSource: SCNGeometrySource) -> SCNGeometry {
        fatalError("unimplemented")
    }
}
