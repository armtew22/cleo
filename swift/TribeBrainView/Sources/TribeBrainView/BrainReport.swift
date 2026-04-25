// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BrainReport
//
// Codable mirror of `tribe_backend.parcellation.unit.Report.to_json()`:
//
//   {
//     "top_regions": [
//        { "name": "...", "z_score": 2.4, "direction": "activation",
//          "description": "..." }, ...
//     ],
//     "text": "Window summary: ...",
//     "method": "window_mean",
//     "z_threshold": 1.5,
//     "window_id": "abc123" | null
//   }

public struct BrainReport: Codable, Equatable {
    public let topRegions: [RegionActivation]
    public let text: String
    public let method: String
    public let zThreshold: Double
    public let windowId: String?

    public init(
        topRegions: [RegionActivation],
        text: String,
        method: String,
        zThreshold: Double,
        windowId: String? = nil
    ) {
        self.topRegions = topRegions
        self.text = text
        self.method = method
        self.zThreshold = zThreshold
        self.windowId = windowId
    }

    private enum CodingKeys: String, CodingKey {
        case topRegions = "top_regions"
        case text
        case method
        case zThreshold = "z_threshold"
        case windowId = "window_id"
    }
}

// MARK: - RegionActivation

public struct RegionActivation: Codable, Equatable {
    public let name: String
    public let zScore: Double
    /// "activation" | "deactivation".
    public let direction: String
    public let description: String

    public init(name: String, zScore: Double, direction: String, description: String = "") {
        self.name = name
        self.zScore = zScore
        self.direction = direction
        self.description = description
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case zScore = "z_score"
        case direction
        case description
    }
}
