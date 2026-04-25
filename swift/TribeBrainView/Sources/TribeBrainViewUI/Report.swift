// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import TribeBrainView

// See TribeBrainView.BrainReport — that struct is the canonical Codable mirror of
// GlasserParcellationUnit.Report. A parallel struct here would be redundant; the
// typealias below exposes it under the shorter name for UI-layer call sites.
// Decision (Phase 7): collapsed to a single struct in the core target. Phase 8 docs
// should update §3 file-tree note for Report.swift to reflect this.
public typealias Report = BrainReport
