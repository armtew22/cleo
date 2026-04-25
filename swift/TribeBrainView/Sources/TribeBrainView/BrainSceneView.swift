// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import SwiftUI
import SceneKit

// MARK: - Platform aliases

#if canImport(UIKit)
import UIKit
public typealias _PlatformViewRepresentable = UIViewRepresentable
#elseif canImport(AppKit)
import AppKit
public typealias _PlatformViewRepresentable = NSViewRepresentable
#endif

// MARK: - BrainSceneView

/// SwiftUI representable that hosts an `SCNView` and drives it through a
/// `BrainSceneCoordinator`. iOS / visionOS use UIViewRepresentable, macOS uses
/// NSViewRepresentable; both share the same coordinator and update logic.
public struct BrainSceneView: _PlatformViewRepresentable {
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

    // MARK: - Shared make/update

    private func makeView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero)
        let coord = context.coordinator
        coord.attach(view)

        // Load initial geometry from the source.
        switch source.kind {
        case .binaryDirectory:
            do {
                let assets = try BrainMeshLoader.load(source)
                coord.setGeometry(assets.geometry, vertexCount: assets.vertexCount)
            } catch {
                // Surface as a debug print; production hosts should preflight the path.
                #if DEBUG
                print("BrainSceneView: failed to load source: \(error)")
                #endif
            }
        case .remote:
            // Phase 6c (Swift networking client) will bootstrap via BrainMeshClient.
            fatalError("Phase 6c")
        case .json, .glb:
            fatalError("unimplemented")
        }

        return view
    }

    private func updateView(_ view: SCNView, context: Context) {
        let coord = context.coordinator
        coord.applyConfiguration(configuration)
        coord.applyCameraPreset(configuration.cameraPreset)

        // Drive animation: if a bundle and binding are present and the frame has
        // changed since last apply, swap the color buffer.
        if let frameBinding = currentFrame,
           let bundle = animation,
           coord.vertexCount > 0
        {
            let frame = max(0, min(frameBinding.wrappedValue, bundle.frameCount - 1))
            if frame != coord.lastAppliedFrame {
                do {
                    try coord.swapColors(fromAnimationBundle: bundle, frame: frame)
                } catch {
                    #if DEBUG
                    print("BrainSceneView: swapColors failed at frame \(frame): \(error)")
                    #endif
                }
            }
        }
    }

    // MARK: - Platform-specific entry points

    #if canImport(UIKit)
    public func makeUIView(context: Context) -> SCNView { makeView(context: context) }
    public func updateUIView(_ uiView: SCNView, context: Context) { updateView(uiView, context: context) }
    #elseif canImport(AppKit)
    public func makeNSView(context: Context) -> SCNView { makeView(context: context) }
    public func updateNSView(_ nsView: SCNView, context: Context) { updateView(nsView, context: context) }
    #endif
}
