// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainMeshSource

/// Describes where brain mesh data should be loaded from.
public struct BrainMeshSource {
    /// The backing storage kind for a ``BrainMeshSource``.
    public enum Kind {
        /// Load from a local directory of `.bin` + `brain_meta.json` files.
        case binaryDirectory(URL)
        /// Fetch static mesh once and stream per-poll colors via a ``BrainMeshClient``.
        case remote(BrainMeshClient)
        /// Load from a JSON export (compatibility shim; binary is canonical).
        case json(URL)
        /// Load from a GLB file via ModelIO (gated behind `canImport(ModelIO)`).
        case glb(URL)
    }

    /// The backing storage kind.
    public let kind: Kind

    /// Creates a source from the given kind.
    public init(kind: Kind) {
        self.kind = kind
    }
}

// MARK: - BrainViewConfiguration

/// Rendering configuration passed to ``BrainSceneView`` at init time.
public struct BrainViewConfiguration {
    /// Initial camera orientation applied when the view first appears.
    public var cameraPreset: CameraPreset = .lateralLeft
    /// Scene background color (default: transparent).
    public var background: BrainColor = .clear
    /// SceneKit lighting model applied to the brain material.
    public var lighting: LightingMode = .physicallyBased
    /// When `true`, the user can orbit/zoom the brain with touch or mouse gestures.
    public var allowsCameraControl: Bool = true
    /// When `true`, SceneKit smooth-shades across triangle boundaries.
    public var smoothShading: Bool = true

    /// Creates a configuration with all default values.
    public init() {}
}

// MARK: - LightingMode

/// SceneKit lighting model for the cortical surface material.
public enum LightingMode: String, CaseIterable {
    case physicallyBased
    case lambert
    case blinn
    case phong
    case constant
}

// MARK: - BrainColor (platform-agnostic stand-in)

/// A minimal color type so the core target has zero UIKit / AppKit dependency.
/// Consumers can bridge to SwiftUI.Color or UIColor as needed.
public struct BrainColor: Equatable {
    public let red: Double
    public let green: Double
    public let blue: Double
    public let alpha: Double

    public static let clear = BrainColor(red: 0, green: 0, blue: 0, alpha: 0)
    public static let black = BrainColor(red: 0, green: 0, blue: 0, alpha: 1)
    public static let white = BrainColor(red: 1, green: 1, blue: 1, alpha: 1)

    public init(red: Double, green: Double, blue: Double, alpha: Double = 1) {
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    }
}

// MARK: - AggregationMethod

/// Temporal aggregation method forwarded to the server's `POST /v1/inference/colors`.
public enum AggregationMethod: String, CaseIterable, Codable {
    /// Average activation across the inference window (default).
    case windowMean = "window_mean"
    /// Single peak-activation frame.
    case peak
    /// Peak frame within a sliding window.
    case peakWindow = "peak_window"
}
