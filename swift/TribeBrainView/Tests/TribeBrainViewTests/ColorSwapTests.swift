// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import XCTest
import SceneKit
@testable import TribeBrainView

final class ColorSwapTests: XCTestCase {
    func testUpdateColorsReusesVertexAndNormalSources() throws {
        let staticDir = try BinaryLoaderTests.staticFixtureDirectory()
        let assets = try BrainMeshLoader.load(.init(kind: .binaryDirectory(staticDir)))
        let original = assets.geometry

        let originalVertex = original.sources(for: .vertex).first
        let originalNormal = original.sources(for: .normal).first
        let originalColor = original.sources(for: .color).first
        XCTAssertNotNil(originalVertex)
        XCTAssertNotNil(originalColor)

        // Swap to frame 10 of the animation bundle.
        let frameURL = try BinaryLoaderTests.fixturesRoot()
            .appendingPathComponent("animation/brain_colors_t0010.bin")
        let updated = try BrainMeshLoader.updateColors(
            on: original,
            from: frameURL,
            vertexCount: assets.vertexCount
        )

        let newVertex = updated.sources(for: .vertex).first
        let newNormal = updated.sources(for: .normal).first
        let newColor = updated.sources(for: .color).first

        // Vertex source identity is preserved (no realloc of the heavy buffer).
        XCTAssertNotNil(newVertex)
        XCTAssertTrue(originalVertex === newVertex,
                      "vertex SCNGeometrySource must be the same instance after color swap")

        // Normal source identity preserved when present.
        if let originalNormal, let newNormal {
            XCTAssertTrue(originalNormal === newNormal,
                          "normal SCNGeometrySource must be the same instance after color swap")
        }

        // Color source must have changed.
        XCTAssertNotNil(newColor)
        XCTAssertFalse(originalColor === newColor,
                       "color SCNGeometrySource must be a new instance after color swap")
    }

    func testUpdateColorsSizeMismatchThrows() throws {
        let staticDir = try BinaryLoaderTests.staticFixtureDirectory()
        let assets = try BrainMeshLoader.load(.init(kind: .binaryDirectory(staticDir)))

        // Write a 1-byte file to a temporary location.
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bad-colors-\(UUID().uuidString).bin")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        XCTAssertThrowsError(
            try BrainMeshLoader.updateColors(
                on: assets.geometry,
                from: tmp,
                vertexCount: assets.vertexCount
            )
        ) { error in
            guard case BrainMeshError.sizeMismatch(let expected, let actual) = error else {
                XCTFail("Expected sizeMismatch, got \(error)")
                return
            }
            XCTAssertEqual(expected, assets.vertexCount * 4)
            XCTAssertEqual(actual, 1)
        }
    }
}
