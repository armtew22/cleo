// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import TribeBrainView

// MARK: - BrainTimelineScrubber

public struct BrainTimelineScrubber: View {
    @ObservedObject public var controller: BrainAnimationController

    public init(controller: BrainAnimationController) {
        self.controller = controller
    }

    public var body: some View {
        fatalError("unimplemented")
    }
}
