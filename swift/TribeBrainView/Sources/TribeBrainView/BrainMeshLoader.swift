// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - BrainMeshError

public enum BrainMeshError: Error {
    case malformed(String)
    case missingFile(URL)
    case metaMismatch(String)
    case sizeMismatch(expected: Int, actual: Int)
    case unsupportedFormat
}

// MARK: - BrainMeshLoader

public enum BrainMeshLoader {
    // Layout constants — must match `tribe_backend/mesh/exporter.py::export_binary`.
    fileprivate static let bytesPerVertex = 12      // 3 × Float32
    fileprivate static let bytesPerFaceTriangle = 12 // 3 × Int32 (uint32 indices)
    fileprivate static let bytesPerColor = 4         // RGBA × UInt8
    fileprivate static let bytesPerNormal = 12       // 3 × Float32

    /// Load a complete mesh from a binary-buffer directory or other source.
    public static func load(_ source: BrainMeshSource) throws -> BrainMeshAssets {
        switch source.kind {
        case .binaryDirectory(let url):
            return try loadBinary(url)
        case .remote, .json, .glb:
            // Other source kinds wired in later phases.
            fatalError("unimplemented")
        }
    }

    /// Replace only the per-vertex color source on an existing geometry.
    /// Vertex, normal, and face sources are reused without reallocation —
    /// the returned `SCNGeometry` carries the *same* `SCNGeometrySource` instances
    /// for `.vertex` and `.normal` semantics (object-identity preserved).
    @discardableResult
    public static func updateColors(
        on geometry: SCNGeometry,
        from url: URL,
        vertexCount: Int
    ) throws -> SCNGeometry {
        // Read only the colors file. URL points at a single .bin (NOT a directory).
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw BrainMeshError.missingFile(url)
        }
        let expected = vertexCount * bytesPerColor
        guard data.count == expected else {
            throw BrainMeshError.sizeMismatch(expected: expected, actual: data.count)
        }

        let newColorSource = SCNGeometrySource(
            data: data,
            semantic: .color,
            vectorCount: vertexCount,
            usesFloatComponents: false,
            componentsPerVector: 4,
            bytesPerComponent: 1,
            dataOffset: 0,
            dataStride: bytesPerColor
        )

        // Preserve identity for vertex/normal sources; rebuild geometry with the
        // existing element(s) intact. SCNGeometry.sources/.elements are immutable.
        var rebuiltSources: [SCNGeometrySource] = []
        for source in geometry.sources {
            if source.semantic == .color {
                rebuiltSources.append(newColorSource)
            } else {
                rebuiltSources.append(source)
            }
        }
        // If the original geometry happened to have no color source, append one.
        if !rebuiltSources.contains(where: { $0.semantic == .color }) {
            rebuiltSources.append(newColorSource)
        }

        let rebuilt = SCNGeometry(sources: rebuiltSources, elements: geometry.elements)
        rebuilt.materials = geometry.materials
        rebuilt.name = geometry.name
        return rebuilt
    }

    // MARK: - Internals

    /// Load the four binary files + meta JSON from a directory and assemble an SCNGeometry.
    static func loadBinary(_ directory: URL) throws -> BrainMeshAssets {
        let metaURL = directory.appendingPathComponent("brain_meta.json")
        let vertsURL = directory.appendingPathComponent("brain_vertices.bin")
        let facesURL = directory.appendingPathComponent("brain_faces.bin")
        let colorsURL = directory.appendingPathComponent("brain_colors.bin")
        let normalsURL = directory.appendingPathComponent("brain_normals.bin")

        // Required files.
        for required in [metaURL, vertsURL, facesURL, colorsURL] {
            guard FileManager.default.fileExists(atPath: required.path) else {
                throw BrainMeshError.missingFile(required)
            }
        }

        // Decode meta.
        let metaData: Data
        do {
            metaData = try Data(contentsOf: metaURL)
        } catch {
            throw BrainMeshError.missingFile(metaURL)
        }
        let meta: ColormapMeta
        do {
            meta = try JSONDecoder().decode(ColormapMeta.self, from: metaData)
        } catch {
            throw BrainMeshError.malformed("brain_meta.json: \(error)")
        }

        // Sanity-check meta against expected layout constants.
        guard meta.bytesPerVertex == bytesPerVertex else {
            throw BrainMeshError.metaMismatch(
                "bytes_per_vertex=\(meta.bytesPerVertex), expected \(bytesPerVertex)"
            )
        }
        guard meta.bytesPerFace == bytesPerFaceTriangle else {
            throw BrainMeshError.metaMismatch(
                "bytes_per_face=\(meta.bytesPerFace), expected \(bytesPerFaceTriangle)"
            )
        }
        guard meta.bytesPerColor == bytesPerColor else {
            throw BrainMeshError.metaMismatch(
                "bytes_per_color=\(meta.bytesPerColor), expected \(bytesPerColor)"
            )
        }

        let vertexCount = meta.vertexCount
        let faceCount = meta.faceCount

        // Memory-map .bin files. Falls back to in-memory copy if mapping is unsafe (sandbox).
        let verticesData = try mappedData(at: vertsURL)
        let facesData = try mappedData(at: facesURL)
        let colorsData = try mappedData(at: colorsURL)

        // Validate sizes.
        try BinaryBufferReader(data: verticesData, label: "vertices")
            .validate(expectedCount: vertexCount, stride: bytesPerVertex)
        try BinaryBufferReader(data: facesData, label: "faces")
            .validate(expectedCount: faceCount, stride: bytesPerFaceTriangle)
        try BinaryBufferReader(data: colorsData, label: "colors")
            .validate(expectedCount: vertexCount, stride: bytesPerColor)

        // Build SCNGeometrySources.
        var sources: [SCNGeometrySource] = []

        let vertexSource = SCNGeometrySource(
            data: verticesData,
            semantic: .vertex,
            vectorCount: vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: 4,
            dataOffset: 0,
            dataStride: bytesPerVertex
        )
        sources.append(vertexSource)

        // Normals are OPTIONAL: the current Python exporter does not emit
        // brain_normals.bin (see tribe_backend/mesh/exporter.py::export_binary).
        // If present, use them; otherwise let SceneKit auto-shade from triangle
        // geometry. TODO: when the exporter starts emitting normals, this block
        // becomes the authoritative source and the no-normals branch can drop.
        if FileManager.default.fileExists(atPath: normalsURL.path) {
            let normalsData = try mappedData(at: normalsURL)
            try BinaryBufferReader(data: normalsData, label: "normals")
                .validate(expectedCount: vertexCount, stride: bytesPerNormal)
            let normalSource = SCNGeometrySource(
                data: normalsData,
                semantic: .normal,
                vectorCount: vertexCount,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: 4,
                dataOffset: 0,
                dataStride: bytesPerNormal
            )
            sources.append(normalSource)
        }

        let colorSource = SCNGeometrySource(
            data: colorsData,
            semantic: .color,
            vectorCount: vertexCount,
            usesFloatComponents: false,
            componentsPerVector: 4,
            bytesPerComponent: 1,
            dataOffset: 0,
            dataStride: bytesPerColor
        )
        sources.append(colorSource)

        // Build a single triangle element. Faces are 3 × uint32 per triangle.
        let element = SCNGeometryElement(
            data: facesData,
            primitiveType: .triangles,
            primitiveCount: faceCount,
            bytesPerIndex: 4
        )

        let geometry = SCNGeometry(sources: sources, elements: [element])

        // fsaverage5 splits 50/50 across hemispheres (10242 + 10242). The current
        // exporter does not emit lh/rh counts in meta, so we derive from total.
        // TODO: read explicit lh_vertex_count / rh_vertex_count when the exporter
        // begins emitting them.
        let lh = vertexCount / 2
        let rh = vertexCount - lh

        return BrainMeshAssets(
            geometry: geometry,
            vertexCount: vertexCount,
            lhVertexCount: lh,
            rhVertexCount: rh,
            faceCount: faceCount,
            colormapMeta: meta
        )
    }

    private static func mappedData(at url: URL) throws -> Data {
        do {
            return try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw BrainMeshError.missingFile(url)
        }
    }
}
