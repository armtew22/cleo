// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - ColormapMeta

/// Codable mirror of brain_meta.json produced by BrainMeshExporter.
public struct ColormapMeta: Codable {
    public let vertexCount: Int
    public let lhVertexCount: Int
    public let rhVertexCount: Int
    public let faceCount: Int
    public let colormapName: String
    public let vmin: Double
    public let vmax: Double
    public let surfaceType: String

    public init(
        vertexCount: Int,
        lhVertexCount: Int,
        rhVertexCount: Int,
        faceCount: Int,
        colormapName: String,
        vmin: Double,
        vmax: Double,
        surfaceType: String
    ) {
        self.vertexCount = vertexCount
        self.lhVertexCount = lhVertexCount
        self.rhVertexCount = rhVertexCount
        self.faceCount = faceCount
        self.colormapName = colormapName
        self.vmin = vmin
        self.vmax = vmax
        self.surfaceType = surfaceType
    }

    private enum CodingKeys: String, CodingKey {
        case vertexCount = "vertex_count"
        case lhVertexCount = "lh_vertex_count"
        case rhVertexCount = "rh_vertex_count"
        case faceCount = "face_count"
        case colormapName = "colormap_name"
        case vmin
        case vmax
        case surfaceType = "surface_type"
    }
}
