// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import TribeBrainView

// MARK: - ColorbarLegend

/// A horizontal gradient strip annotated with vmin/vmax labels.
///
/// Default colormap: RdBu_r diverging (red → white → blue), matching the
/// matplotlib default used by the Python exporter.
///
/// When `meta.colormap_name == "viridis"`, uses a viridis-approximation
/// gradient (purple → teal → yellow).
///
/// Missing `vmin`/`vmax` values render as "—".
public struct ColorbarLegend: View {
    public let meta: ColormapMeta

    public init(meta: ColormapMeta) {
        self.meta = meta
    }

    private var gradient: LinearGradient {
        switch meta.colormapName?.lowercased() {
        case "viridis":
            return LinearGradient(
                colors: [
                    Color(red: 0.267, green: 0.005, blue: 0.329), // purple
                    Color(red: 0.128, green: 0.563, blue: 0.551), // teal
                    Color(red: 0.993, green: 0.906, blue: 0.144), // yellow
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
        default:
            // RdBu_r: red → white → blue
            return LinearGradient(
                colors: [
                    Color(red: 0.698, green: 0.094, blue: 0.168), // RdBu_r red end
                    Color.white,
                    Color(red: 0.192, green: 0.408, blue: 0.663), // RdBu_r blue end
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
        }
    }

    private var vminLabel: String {
        guard let v = meta.vmin else { return "—" }
        return String(format: "%.2g", v)
    }

    private var vmaxLabel: String {
        guard let v = meta.vmax else { return "—" }
        return String(format: "%.2g", v)
    }

    public var body: some View {
        HStack(spacing: 6) {
            Text(vminLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()

            gradient
                .frame(height: 14)
                .clipShape(RoundedRectangle(cornerRadius: 4))

            Text(vmaxLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        .padding(.horizontal)
    }
}
