// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainReport

/// Codable mirror of GlasserParcellationUnit.Report from the Python backend.
public struct BrainReport: Codable {
    public let windowId: String
    public let method: String
    public let topRegions: [RegionActivation]
    public let vmin: Double
    public let vmax: Double

    public init(
        windowId: String,
        method: String,
        topRegions: [RegionActivation],
        vmin: Double,
        vmax: Double
    ) {
        self.windowId = windowId
        self.method = method
        self.topRegions = topRegions
        self.vmin = vmin
        self.vmax = vmax
    }

    private enum CodingKeys: String, CodingKey {
        case windowId = "window_id"
        case method
        case topRegions = "top_regions"
        case vmin
        case vmax
    }
}

// MARK: - RegionActivation

public struct RegionActivation: Codable {
    public let label: String
    public let parcelId: Int
    public let zScore: Double
    public let strengthLabel: String
    public let hemisphere: Hemisphere

    public init(
        label: String,
        parcelId: Int,
        zScore: Double,
        strengthLabel: String,
        hemisphere: Hemisphere
    ) {
        self.label = label
        self.parcelId = parcelId
        self.zScore = zScore
        self.strengthLabel = strengthLabel
        self.hemisphere = hemisphere
    }

    private enum CodingKeys: String, CodingKey {
        case label
        case parcelId = "parcel_id"
        case zScore = "z_score"
        case strengthLabel = "strength_label"
        case hemisphere
    }
}

// MARK: - Hemisphere

public enum Hemisphere: String, Codable, CaseIterable {
    case left = "lh"
    case right = "rh"
}
