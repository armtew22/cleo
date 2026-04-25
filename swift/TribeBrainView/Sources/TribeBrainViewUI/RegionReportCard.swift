// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import TribeBrainView

// MARK: - RegionReportCard

/// A card showing the top-N activated brain regions from a `BrainReport`.
///
/// Each row shows the region label, a z-score chip, and an L/R hemisphere badge.
public struct RegionReportCard: View {
    public let report: BrainReport
    public let topN: Int

    public init(report: BrainReport, topN: Int = 5) {
        self.report = report
        self.topN = topN
    }

    private var regions: [RegionActivation] {
        Array(report.topRegions.prefix(topN))
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Top regions")
                .font(.headline)

            if regions.isEmpty {
                Text("No regions")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(regions, id: \.parcelId) { region in
                    RegionRow(region: region)
                }
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - RegionRow

private struct RegionRow: View {
    let region: RegionActivation

    var body: some View {
        HStack(spacing: 6) {
            Text(region.label)
                .font(.subheadline)
                .lineLimit(1)

            Spacer(minLength: 4)

            // Hemisphere badge
            Text(hemisphereBadge)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            // Z-score chip
            Text(String(format: "z %.2f", region.zScore))
                .font(.caption)
                .fontWeight(.medium)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(chipColor.opacity(0.15), in: Capsule())
                .foregroundStyle(chipColor)
        }
    }

    private var hemisphereBadge: String {
        switch region.hemisphere {
        case .left:  return "L"
        case .right: return "R"
        }
    }

    /// Positive z → warm, negative z → cool, near-zero → neutral.
    private var chipColor: Color {
        if region.zScore > 0 {
            return .orange
        } else if region.zScore < 0 {
            return .blue
        } else {
            return .secondary
        }
    }
}
