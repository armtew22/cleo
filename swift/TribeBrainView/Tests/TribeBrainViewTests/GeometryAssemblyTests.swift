// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import XCTest
import SceneKit
@testable import TribeBrainView

final class GeometryAssemblyTests: XCTestCase {
    func testGeometryHasExpectedSourcesAndElement() throws {
        let dir = try BinaryLoaderTests.staticFixtureDirectory()
        let assets = try BrainMeshLoader.load(.init(kind: .binaryDirectory(dir)))
        let geometry = assets.geometry

        let vertexSources = geometry.sources(for: .vertex)
        let normalSources = geometry.sources(for: .normal)
        let colorSources = geometry.sources(for: .color)

        XCTAssertEqual(vertexSources.count, 1, "expect exactly one .vertex source")
        XCTAssertEqual(colorSources.count, 1, "expect exactly one .color source")
        // Normals are optional — current Python exporter does not emit them.
        XCTAssertLessThanOrEqual(normalSources.count, 1)

        XCTAssertEqual(vertexSources.first?.vectorCount, assets.vertexCount)
        XCTAssertEqual(vertexSources.first?.componentsPerVector, 3)
        XCTAssertEqual(vertexSources.first?.bytesPerComponent, 4)
        XCTAssertEqual(vertexSources.first?.dataStride, 12)

        XCTAssertEqual(colorSources.first?.vectorCount, assets.vertexCount)
        XCTAssertEqual(colorSources.first?.componentsPerVector, 4)
        XCTAssertEqual(colorSources.first?.bytesPerComponent, 1)
        XCTAssertEqual(colorSources.first?.dataStride, 4)
        XCTAssertEqual(colorSources.first?.usesFloatComponents, false)

        XCTAssertEqual(geometry.elements.count, 1)
        let element = geometry.elements[0]
        XCTAssertEqual(element.primitiveType, .triangles)
        XCTAssertEqual(element.primitiveCount, assets.faceCount)
        XCTAssertEqual(element.bytesPerIndex, 4)
    }
}
