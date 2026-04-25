// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainMeshSource

public struct BrainMeshSource {
    public enum Kind {
        case binaryDirectory(URL)
        case remote(BrainMeshClient)
        case json(URL)
        case glb(URL)
    }

    public let kind: Kind

    public init(kind: Kind) {
        self.kind = kind
    }
}

// MARK: - BrainViewConfiguration

public struct BrainViewConfiguration {
    public var cameraPreset: CameraPreset = .lateralLeft
    public var background: BrainColor = .clear
    public var lighting: LightingMode = .physicallyBased
    public var allowsCameraControl: Bool = true
    public var smoothShading: Bool = true

    public init() {}
}

// MARK: - LightingMode

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

public enum AggregationMethod: String, CaseIterable, Codable {
    case windowMean = "window_mean"
    case peak
    case peakWindow = "peak_window"
}
