// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import XCTest
import SceneKit
@testable import TribeBrainView

final class BinaryLoaderTests: XCTestCase {
    // MARK: - Fixture access

    /// Locate the `Fixtures/` directory inside the test bundle.
    static func fixturesRoot() throws -> URL {
        guard let url = Bundle.module.url(forResource: "Fixtures", withExtension: nil) else {
            throw XCTSkip("Fixtures directory not present in test bundle")
        }
        return url
    }

    static func staticFixtureDirectory() throws -> URL {
        try fixturesRoot().appendingPathComponent("static")
    }

    // MARK: - Tests

    func testLoadStaticFixtureProducesCorrectCounts() throws {
        let dir = try Self.staticFixtureDirectory()
        let assets = try BrainMeshLoader.load(.init(kind: .binaryDirectory(dir)))

        XCTAssertEqual(assets.vertexCount, 20484)
        XCTAssertEqual(assets.lhVertexCount + assets.rhVertexCount, 20484)
        XCTAssertGreaterThan(assets.faceCount, 0)
        XCTAssertEqual(assets.faceCount, 40960)
        XCTAssertEqual(assets.colormapMeta.surface, "fsaverage5")
        XCTAssertEqual(assets.colormapMeta.surfaceType, "pial")
    }

    func testMissingColorsFileThrowsMissingFile() throws {
        let staticDir = try Self.staticFixtureDirectory()
        let tmp = try makeTempCopy(of: staticDir, excluding: ["brain_colors.bin"])
        defer { try? FileManager.default.removeItem(at: tmp) }

        XCTAssertThrowsError(
            try BrainMeshLoader.load(.init(kind: .binaryDirectory(tmp)))
        ) { error in
            guard case BrainMeshError.missingFile(let url) = error else {
                XCTFail("Expected missingFile, got \(error)")
                return
            }
            XCTAssertEqual(url.lastPathComponent, "brain_colors.bin")
        }
    }

    func testWrongVertexFileSizeThrowsSizeMismatch() throws {
        let staticDir = try Self.staticFixtureDirectory()
        let tmp = try makeTempCopy(of: staticDir, excluding: [])
        defer { try? FileManager.default.removeItem(at: tmp) }

        // Truncate brain_vertices.bin to half the expected size.
        let vertsURL = tmp.appendingPathComponent("brain_vertices.bin")
        let original = try Data(contentsOf: vertsURL)
        let truncated = original.prefix(original.count / 2)
        try truncated.write(to: vertsURL)

        XCTAssertThrowsError(
            try BrainMeshLoader.load(.init(kind: .binaryDirectory(tmp)))
        ) { error in
            guard case BrainMeshError.sizeMismatch(let expected, let actual) = error else {
                XCTFail("Expected sizeMismatch, got \(error)")
                return
            }
            XCTAssertEqual(expected, 20484 * 12)
            XCTAssertEqual(actual, truncated.count)
        }
    }

    func testManifestSanityCheckMatchesVerticesFileSize() throws {
        // Sanity check against MANIFEST.json so a drifting fixture surfaces here,
        // not as a confusing failure deeper in the loader.
        let manifestURL = try Self.fixturesRoot().appendingPathComponent("MANIFEST.json")
        let entries = try JSONDecoder().decode(
            [ManifestEntry].self,
            from: Data(contentsOf: manifestURL)
        )
        guard let verts = entries.first(where: { $0.relpath == "static/brain_vertices.bin" }) else {
            XCTFail("MANIFEST missing static/brain_vertices.bin entry")
            return
        }
        let vertsURL = try Self.staticFixtureDirectory()
            .appendingPathComponent("brain_vertices.bin")
        let actualBytes = try FileManager.default
            .attributesOfItem(atPath: vertsURL.path)[.size] as? Int
        XCTAssertEqual(actualBytes, verts.bytes)
        XCTAssertEqual(verts.bytes, 20484 * 12)
    }

    // MARK: - Helpers

    private struct ManifestEntry: Decodable {
        let relpath: String
        let sha256: String
        let bytes: Int
    }

    private func makeTempCopy(of source: URL, excluding: [String]) throws -> URL {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("brainmesh-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let contents = try FileManager.default.contentsOfDirectory(atPath: source.path)
        for name in contents where !excluding.contains(name) {
            try FileManager.default.copyItem(
                at: source.appendingPathComponent(name),
                to: tmp.appendingPathComponent(name)
            )
        }
        return tmp
    }
}
