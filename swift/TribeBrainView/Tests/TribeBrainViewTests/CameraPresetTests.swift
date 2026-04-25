// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import XCTest
import SceneKit
@testable import TribeBrainView

final class CameraPresetTests: XCTestCase {

    private func length(_ v: SCNVector3) -> Float {
        Float((Double(v.x) * Double(v.x) + Double(v.y) * Double(v.y) + Double(v.z) * Double(v.z)).squareRoot())
    }

    private func dot(_ a: SCNVector3, _ b: SCNVector3) -> Float {
        Float(Double(a.x) * Double(b.x) + Double(a.y) * Double(b.y) + Double(a.z) * Double(b.z))
    }

    private func hasNaN(_ v: SCNVector3) -> Bool {
        Float(v.x).isNaN || Float(v.y).isNaN || Float(v.z).isNaN
    }

    func testAllPresetsReachableViaCaseIterable() {
        let names = Set(CameraPreset.allCases.map { $0.rawValue })
        let expected: Set<String> = [
            "lateral_left", "lateral_right", "medial_left", "medial_right",
            "dorsal", "ventral", "anterior", "posterior",
            "frontal_three_quarter", "custom",
        ]
        XCTAssertEqual(names, expected)
        XCTAssertEqual(CameraPreset.allCases.count, 10)
    }

    func testNonCustomPresetsHaveUnitUpVectorAndNoNaNs() {
        for preset in CameraPreset.allCases where preset != .custom {
            let d = cameraDescriptor(for: preset)
            XCTAssertFalse(hasNaN(d.viewVector), "viewVector NaN in \(preset)")
            XCTAssertFalse(hasNaN(d.upVector),   "upVector NaN in \(preset)")

            XCTAssertEqual(length(d.upVector), 1.0, accuracy: 1e-5,
                           "up not unit-length in \(preset)")
            // viewVector should also be a unit direction.
            XCTAssertEqual(length(d.viewVector), 1.0, accuracy: 1e-5,
                           "view not unit-length in \(preset)")
            XCTAssertGreaterThan(d.fov, 0)
        }
    }

    func testViewOrthogonalToUpForCardinalPresets() {
        // Cardinal presets where view ⊥ up is a hard requirement.
        let cardinals: [CameraPreset] = [
            .lateralLeft, .lateralRight, .medialLeft, .medialRight,
            .dorsal, .ventral, .anterior, .posterior,
        ]
        for preset in cardinals {
            let d = cameraDescriptor(for: preset)
            XCTAssertEqual(dot(d.viewVector, d.upVector), 0.0, accuracy: 1e-5,
                           "view not orthogonal to up in \(preset)")
        }
    }

    func testCustomPresetReturnsIdentityPlaceholder() {
        let d = cameraDescriptor(for: .custom)
        XCTAssertFalse(hasNaN(d.viewVector))
        XCTAssertFalse(hasNaN(d.upVector))
        // Custom returns a sentinel; up is still a unit vector for safety.
        XCTAssertEqual(length(d.upVector), 1.0, accuracy: 1e-5)
    }

    func testFrontalThreeQuarterIsUnitVector() {
        let d = cameraDescriptor(for: .frontalThreeQuarter)
        XCTAssertEqual(length(d.viewVector), 1.0, accuracy: 1e-5)
        // All three components equal magnitude → 1/sqrt(3).
        let inv = Float(1.0 / 3.0.squareRoot())
        XCTAssertEqual(abs(Float(d.viewVector.x)), inv, accuracy: 1e-5)
        XCTAssertEqual(abs(Float(d.viewVector.y)), inv, accuracy: 1e-5)
        XCTAssertEqual(abs(Float(d.viewVector.z)), inv, accuracy: 1e-5)
    }

    func testRadiusConstantIsReasonable() {
        // Sanity: sphere big enough to contain fsaverage5 (~80 mm radius) with margin.
        XCTAssertGreaterThan(brainCameraRadius, 100)
        XCTAssertLessThan(brainCameraRadius, 1000)
    }
}
