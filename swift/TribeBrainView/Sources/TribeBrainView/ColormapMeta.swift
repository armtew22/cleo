// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - ColormapMeta

/// Codable mirror of `brain_meta.json` produced by `BrainMeshExporter.export_binary()`.
///
/// The Python exporter (see `tribe_backend/mesh/exporter.py`) writes:
/// ```json
/// {
///   "format": "tribe_brain_mesh_v1",
///   "surface": "fsaverage5",
///   "surface_type": "pial",
///   "vertex_count": 20484,
///   "face_count": 40960,
///   "bytes_per_vertex": 12,
///   "bytes_per_face": 12,
///   "bytes_per_color": 4,
///   "files": { "vertices": "...", "faces": "...", "colors": "..." },
///   "window_id": "..."
/// }
/// ```
///
/// NOTE: `vmin`, `vmax`, and `colormap_name` are NOT emitted by the current exporter
/// (the colormap is baked into the RGBA buffer at export time). They are exposed here
/// as optional fields so a future exporter revision can populate them without breaking
/// this Codable surface. TODO: revisit if/when exporter starts emitting them.
public struct ColormapMeta: Codable, Equatable {
    public let format: String
    /// Optional: the local-fixture exporter writes "fsaverage5"; the FastAPI
    /// server's bootstrap tar (Phase 6b) omits this field. Treat as informational.
    public let surface: String?
    public let surfaceType: String
    public let vertexCount: Int
    public let faceCount: Int
    public let bytesPerVertex: Int
    public let bytesPerFace: Int
    public let bytesPerColor: Int
    public let files: Files
    public let windowId: String?

    // Optional / forward-compatible fields (not currently emitted).
    public let vmin: Double?
    public let vmax: Double?
    public let colormapName: String?

    public struct Files: Codable, Equatable {
        public let vertices: String
        public let faces: String
        /// Optional: the server bootstrap tar (Phase 6b) ships only static
        /// geometry files; colors arrive on the hot path. The local-fixture
        /// exporter emits this field. Treat as informational.
        public let colors: String?
        public let normals: String?

        public init(vertices: String, faces: String, colors: String? = nil, normals: String? = nil) {
            self.vertices = vertices
            self.faces = faces
            self.colors = colors
            self.normals = normals
        }
    }

    public init(
        format: String,
        surface: String? = nil,
        surfaceType: String,
        vertexCount: Int,
        faceCount: Int,
        bytesPerVertex: Int,
        bytesPerFace: Int,
        bytesPerColor: Int,
        files: Files,
        windowId: String? = nil,
        vmin: Double? = nil,
        vmax: Double? = nil,
        colormapName: String? = nil
    ) {
        self.format = format
        self.surface = surface
        self.surfaceType = surfaceType
        self.vertexCount = vertexCount
        self.faceCount = faceCount
        self.bytesPerVertex = bytesPerVertex
        self.bytesPerFace = bytesPerFace
        self.bytesPerColor = bytesPerColor
        self.files = files
        self.windowId = windowId
        self.vmin = vmin
        self.vmax = vmax
        self.colormapName = colormapName
    }

    private enum CodingKeys: String, CodingKey {
        case format
        case surface
        case surfaceType = "surface_type"
        case vertexCount = "vertex_count"
        case faceCount = "face_count"
        case bytesPerVertex = "bytes_per_vertex"
        case bytesPerFace = "bytes_per_face"
        case bytesPerColor = "bytes_per_color"
        case files
        case windowId = "window_id"
        case vmin
        case vmax
        case colormapName = "colormap_name"
    }
}
