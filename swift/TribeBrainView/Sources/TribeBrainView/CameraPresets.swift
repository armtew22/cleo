// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit
import CoreGraphics

// MARK: - CameraPreset

public enum CameraPreset: String, CaseIterable {
    case lateralLeft         = "lateral_left"
    case lateralRight        = "lateral_right"
    case medialLeft          = "medial_left"
    case medialRight         = "medial_right"
    case dorsal              = "dorsal"
    case ventral             = "ventral"
    case anterior            = "anterior"
    case posterior           = "posterior"
    case frontalThreeQuarter = "frontal_three_quarter"
    case custom              = "custom"
}

// MARK: - CameraPresetDescriptor

/// Descriptor for a camera preset.
///
/// Coordinate convention (MNI / RAS):
///   +X = right, +Y = anterior, +Z = superior.
///
/// `viewVector` is a UNIT vector pointing from the *origin* toward the camera
/// (i.e. the camera sits at `viewVector * radius`, looking back at the origin).
/// `upVector` is the camera's roll-fixing up vector, kept unit-length and
/// orthogonal to `viewVector` where geometrically meaningful.
public struct CameraPresetDescriptor {
    public let viewVector: SCNVector3
    public let upVector: SCNVector3
    public let fov: CGFloat

    public init(viewVector: SCNVector3, upVector: SCNVector3, fov: CGFloat) {
        self.viewVector = viewVector
        self.upVector = upVector
        self.fov = fov
    }
}

// MARK: - Constants

/// Bounding-sphere radius used to position the camera (mm).
/// fsaverage5 pial mesh extends ~80 mm from origin in any direction; 320 mm
/// places the camera comfortably outside with room for a 30deg FOV.
public let brainCameraRadius: Float = 320.0

private let defaultFOV: CGFloat = 30.0

// MARK: - Preset table

public func cameraDescriptor(for preset: CameraPreset) -> CameraPresetDescriptor {
    let s = Float(1.0 / 3.0.squareRoot())   // 1/sqrt(3)

    switch preset {
    case .lateralLeft:
        // Camera sits on -X (subject's left), looking toward +X. Up = +Z.
        return CameraPresetDescriptor(
            viewVector: SCNVector3(-1, 0, 0),
            upVector:   SCNVector3( 0, 0, 1),
            fov: defaultFOV
        )
    case .lateralRight:
        // Camera on +X, up = +Z.
        return CameraPresetDescriptor(
            viewVector: SCNVector3( 1, 0, 0),
            upVector:   SCNVector3( 0, 0, 1),
            fov: defaultFOV
        )
    case .medialLeft:
        // Medial view of LH: camera on +X looking back through the midline.
        return CameraPresetDescriptor(
            viewVector: SCNVector3( 1, 0, 0),
            upVector:   SCNVector3( 0, 0, 1),
            fov: defaultFOV
        )
    case .medialRight:
        // Medial view of RH: camera on -X looking back through the midline.
        return CameraPresetDescriptor(
            viewVector: SCNVector3(-1, 0, 0),
            upVector:   SCNVector3( 0, 0, 1),
            fov: defaultFOV
        )
    case .dorsal:
        // Top-down: camera on +Z, looking toward -Z. Up = +Y (anterior toward top of frame).
        return CameraPresetDescriptor(
            viewVector: SCNVector3( 0, 0, 1),
            upVector:   SCNVector3( 0, 1, 0),
            fov: defaultFOV
        )
    case .ventral:
        // Bottom-up: camera on -Z. Up = -Y so anterior still appears toward top of frame.
        return CameraPresetDescriptor(
            viewVector: SCNVector3( 0, 0, -1),
            upVector:   SCNVector3( 0, -1, 0),
            fov: defaultFOV
        )
    case .anterior:
        // Front (face-on): camera on +Y looking toward -Y. Up = +Z.
        return CameraPresetDescriptor(
            viewVector: SCNVector3( 0, 1, 0),
            upVector:   SCNVector3( 0, 0, 1),
            fov: defaultFOV
        )
    case .posterior:
        // Back: camera on -Y. Up = +Z.
        return CameraPresetDescriptor(
            viewVector: SCNVector3( 0, -1, 0),
            upVector:   SCNVector3( 0, 0, 1),
            fov: defaultFOV
        )
    case .frontalThreeQuarter:
        // 45deg oblique: camera at (+X anterior, +Y, +Z) octant; unit length.
        // Up vector chosen orthogonal-ish (will be re-orthogonalized at apply time
        // by SceneKit's lookAt; we keep a sensible bias toward +Z).
        return CameraPresetDescriptor(
            viewVector: SCNVector3(s, s, s),
            upVector:   SCNVector3(0, 0, 1),
            fov: defaultFOV
        )
    case .custom:
        // Identity placeholder; host sets camera transform directly.
        return CameraPresetDescriptor(
            viewVector: SCNVector3(0, 0, 0),
            upVector:   SCNVector3(0, 0, 1),
            fov: defaultFOV
        )
    }
}
