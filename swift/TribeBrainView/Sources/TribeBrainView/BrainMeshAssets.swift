// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - BrainMeshAssets

public struct BrainMeshAssets {
    public let geometry: SCNGeometry
    public let vertexCount: Int
    public let lhVertexCount: Int
    public let rhVertexCount: Int
    public let faceCount: Int
    public let colormapMeta: ColormapMeta

    public init(
        geometry: SCNGeometry,
        vertexCount: Int,
        lhVertexCount: Int,
        rhVertexCount: Int,
        faceCount: Int,
        colormapMeta: ColormapMeta
    ) {
        self.geometry = geometry
        self.vertexCount = vertexCount
        self.lhVertexCount = lhVertexCount
        self.rhVertexCount = rhVertexCount
        self.faceCount = faceCount
        self.colormapMeta = colormapMeta
    }
}
