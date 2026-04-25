// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainAnimationBundle

/// Loaded animation bundle: directory of per-frame color `.bin` files plus the
/// `animation_meta.json` that lists them.
///
/// The Python exporter (`tribe_backend/mesh/exporter.py::export_animation_bundle`)
/// emits a meta JSON with the schema:
/// ```json
/// {
///   "format": "tribe_brain_mesh_v1",
///   "kind": "animation_bundle",
///   "vertex_count": 20484,
///   "frame_count": 31,
///   "bytes_per_color": 4,
///   "frames": ["brain_colors_t0000.bin", ...]
/// }
/// ```
/// NOTE: `fps` is NOT present in the meta — playback rate is decided by the
/// caller (see `BrainAnimationController.init(bundle:fps:)`). We default
/// `bundle.fps` to `defaultFPS` so legacy callers keep working.
public struct BrainAnimationBundle {
    public let directory: URL
    public let frameCount: Int
    public let fps: Double
    public let vertexCount: Int
    public let frameFilenames: [String]

    /// Default playback rate when the meta does not specify one.
    public static let defaultFPS: Double = 10.0

    /// Reads `animation_meta.json` (with a fallback to `brain_anim_meta.json`)
    /// from `directory`.
    public init(directory: URL) throws {
        self.directory = directory

        let candidates = [
            directory.appendingPathComponent("animation_meta.json"),
            directory.appendingPathComponent("brain_anim_meta.json"),
        ]
        guard let metaURL = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) else {
            throw BrainMeshError.missingFile(candidates[0])
        }

        let data: Data
        do {
            data = try Data(contentsOf: metaURL)
        } catch {
            throw BrainMeshError.missingFile(metaURL)
        }
        let meta: BrainAnimMeta
        do {
            meta = try JSONDecoder().decode(BrainAnimMeta.self, from: data)
        } catch {
            throw BrainMeshError.malformed("animation_meta.json: \(error)")
        }

        guard meta.frameCount == meta.frames.count else {
            throw BrainMeshError.metaMismatch(
                "frame_count=\(meta.frameCount), but frames array has \(meta.frames.count) entries"
            )
        }
        guard meta.frameCount > 0 else {
            throw BrainMeshError.malformed("animation_meta.json: frame_count must be > 0")
        }

        self.frameCount = meta.frameCount
        self.fps = meta.fps ?? Self.defaultFPS
        self.vertexCount = meta.vertexCount
        self.frameFilenames = meta.frames
    }

    /// Resolve the URL of the per-frame `.bin` for a given frame index.
    /// Out-of-range indices are clamped to `[0, frameCount-1]` (documented
    /// behavior; callers that prefer explicit errors should range-check first).
    public func frameURL(at index: Int) -> URL {
        let clamped = max(0, min(index, frameCount - 1))
        return directory.appendingPathComponent(frameFilenames[clamped])
    }
}

// MARK: - BrainAnimMeta (Codable mirror of animation_meta.json)

/// Codable mirror of `animation_meta.json` produced by `BrainMeshExporter.export_animation_bundle()`.
public struct BrainAnimMeta: Codable {
    public let format: String?
    public let kind: String?
    public let vertexCount: Int
    public let frameCount: Int
    public let bytesPerColor: Int
    public let frames: [String]
    /// Optional — not currently emitted by the Python exporter.
    public let fps: Double?

    private enum CodingKeys: String, CodingKey {
        case format
        case kind
        case vertexCount = "vertex_count"
        case frameCount = "frame_count"
        case bytesPerColor = "bytes_per_color"
        case frames
        case fps
    }
}
