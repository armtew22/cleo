// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit

// MARK: - CameraPreset

public enum CameraPreset: String, CaseIterable {
    case lateralLeft        = "lateral_left"
    case lateralRight       = "lateral_right"
    case medialLeft         = "medial_left"
    case medialRight        = "medial_right"
    case dorsal             = "dorsal"
    case ventral            = "ventral"
    case anterior           = "anterior"
    case posterior          = "posterior"
    case frontalThreeQuarter = "frontal_three_quarter"
    case custom             = "custom"
}

// MARK: - CameraPresetDescriptor

public struct CameraPresetDescriptor {
    /// Unit vector pointing from the camera toward the mesh origin (view direction).
    public let viewVector: SCNVector3
    /// Up vector for the camera orientation.
    public let upVector: SCNVector3

    public init(viewVector: SCNVector3, upVector: SCNVector3) {
        self.viewVector = viewVector
        self.upVector = upVector
    }
}

// MARK: - Preset table

/// Returns the camera descriptor for a given preset.
/// Vectors are derived from standard neuroimaging orientation conventions
/// (RAS coordinate system, right = +X, anterior = +Y, superior = +Z).
public func cameraDescriptor(for preset: CameraPreset) -> CameraPresetDescriptor {
    switch preset {
    case .lateralLeft:
        // View from the left (negative X axis)
        return CameraPresetDescriptor(
            viewVector: SCNVector3(1, 0, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    case .lateralRight:
        // View from the right (positive X axis)
        return CameraPresetDescriptor(
            viewVector: SCNVector3(-1, 0, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    case .medialLeft:
        // Medial view of left hemisphere (from positive X side, flipped)
        return CameraPresetDescriptor(
            viewVector: SCNVector3(-1, 0, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    case .medialRight:
        // Medial view of right hemisphere (from negative X side, flipped)
        return CameraPresetDescriptor(
            viewVector: SCNVector3(1, 0, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    case .dorsal:
        // Top-down (superior) view
        return CameraPresetDescriptor(
            viewVector: SCNVector3(0, 0, -1),
            upVector: SCNVector3(0, 1, 0)
        )
    case .ventral:
        // Bottom-up (inferior) view
        return CameraPresetDescriptor(
            viewVector: SCNVector3(0, 0, 1),
            upVector: SCNVector3(0, -1, 0)
        )
    case .anterior:
        // Front (anterior) view
        return CameraPresetDescriptor(
            viewVector: SCNVector3(0, -1, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    case .posterior:
        // Back (posterior) view
        return CameraPresetDescriptor(
            viewVector: SCNVector3(0, 1, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    case .frontalThreeQuarter:
        // 45-degree oblique: anterior-left-superior
        let inv = Float(1.0 / 3.0.squareRoot())
        return CameraPresetDescriptor(
            viewVector: SCNVector3(-inv, -inv, -inv),
            upVector: SCNVector3(0, 0, 1)
        )
    case .custom:
        // Identity placeholder; host sets camera position directly.
        return CameraPresetDescriptor(
            viewVector: SCNVector3(0, -1, 0),
            upVector: SCNVector3(0, 0, 1)
        )
    }
}
