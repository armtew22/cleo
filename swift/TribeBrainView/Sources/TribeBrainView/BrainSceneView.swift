// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import SceneKit

// MARK: - BrainSceneView

#if canImport(UIKit)
import UIKit

public struct BrainSceneView: UIViewRepresentable {
    public let source: BrainMeshSource
    public let configuration: BrainViewConfiguration
    public let currentFrame: Binding<Int>?
    public let animation: BrainAnimationBundle?

    public init(
        source: BrainMeshSource,
        configuration: BrainViewConfiguration = .init(),
        currentFrame: Binding<Int>? = nil,
        animation: BrainAnimationBundle? = nil
    ) {
        self.source = source
        self.configuration = configuration
        self.currentFrame = currentFrame
        self.animation = animation
    }

    public func makeCoordinator() -> BrainSceneCoordinator {
        BrainSceneCoordinator(source: source, configuration: configuration)
    }

    public func makeUIView(context: Context) -> SCNView {
        fatalError("unimplemented")
    }

    public func updateUIView(_ uiView: SCNView, context: Context) {
        fatalError("unimplemented")
    }
}

#elseif canImport(AppKit)
import AppKit

public struct BrainSceneView: NSViewRepresentable {
    public let source: BrainMeshSource
    public let configuration: BrainViewConfiguration
    public let currentFrame: Binding<Int>?
    public let animation: BrainAnimationBundle?

    public init(
        source: BrainMeshSource,
        configuration: BrainViewConfiguration = .init(),
        currentFrame: Binding<Int>? = nil,
        animation: BrainAnimationBundle? = nil
    ) {
        self.source = source
        self.configuration = configuration
        self.currentFrame = currentFrame
        self.animation = animation
    }

    public func makeCoordinator() -> BrainSceneCoordinator {
        BrainSceneCoordinator(source: source, configuration: configuration)
    }

    public func makeNSView(context: Context) -> SCNView {
        fatalError("unimplemented")
    }

    public func updateNSView(_ nsView: SCNView, context: Context) {
        fatalError("unimplemented")
    }
}
#endif
