// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - BrainMeshAssets

/// The result of loading a brain mesh: assembled `SCNGeometry` plus metadata.
public struct BrainMeshAssets {
    /// Fully assembled SceneKit geometry ready to attach to an `SCNNode`.
    public let geometry: SCNGeometry
    /// Total number of vertices (20484 for fsaverage5).
    public let vertexCount: Int
    /// Number of left-hemisphere vertices (derived: `vertexCount / 2`).
    public let lhVertexCount: Int
    /// Number of right-hemisphere vertices (`vertexCount - lhVertexCount`).
    public let rhVertexCount: Int
    /// Number of triangle faces.
    public let faceCount: Int
    /// Parsed colormap / surface metadata from `brain_meta.json`.
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
