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
        case .remote(let client):
            // Bootstrap asynchronously: fetch (or reuse cached) static mesh,
            // ensure a placeholder colors.bin exists (server omits it from
            // /v1/mesh/static — colors arrive on the hot path), then load
            // and install on the coordinator. Subsequent color updates
            // flow through coordinator.updateColors(buffer:) once the host
            // calls fetchColors / pollFull.
            Task { [weak coord] in
                guard let coord else { return }
                do {
                    let dir = try await client.bootstrapStaticMesh()
                    try Self.ensurePlaceholderColors(in: dir)
                    let assets = try BrainMeshLoader.load(
                        BrainMeshSource(kind: .binaryDirectory(dir))
                    )
                    await MainActor.run {
                        coord.setGeometry(assets.geometry, vertexCount: assets.vertexCount)
                    }
                } catch {
                    #if DEBUG
                    print("BrainSceneView: remote bootstrap failed: \(error)")
                    #endif
                }
            }
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

    // MARK: - Remote bootstrap helpers

    /// Ensure the bootstrap directory has a `brain_colors.bin` file the loader
    /// can ingest. The server's /v1/mesh/static tar emits only vertices /
    /// normals / faces / meta — colors arrive over /v1/inference/colors. We
    /// drop a zero-filled placeholder (RGBA × vertex_count) so the loader can
    /// build an SCNGeometry with a neutral color source; the first inference
    /// response will replace it via `coordinator.updateColors(buffer:)`.
    static func ensurePlaceholderColors(in dir: URL) throws {
        let colorsURL = dir.appendingPathComponent("brain_colors.bin")
        if FileManager.default.fileExists(atPath: colorsURL.path) { return }
        let metaURL = dir.appendingPathComponent("brain_meta.json")
        let metaData = try Data(contentsOf: metaURL)
        let meta = try JSONDecoder().decode(ColormapMeta.self, from: metaData)
        let bytes = meta.vertexCount * meta.bytesPerColor
        // Mid-gray fully-opaque (0x80 0x80 0x80 0xFF) so the placeholder is
        // visible if a host renders before the first colors arrive.
        var buf = Data(count: bytes)
        buf.withUnsafeMutableBytes { raw in
            guard let p = raw.baseAddress else { return }
            var i = 0
            while i < bytes {
                p.storeBytes(of: 0x80, toByteOffset: i, as: UInt8.self)
                p.storeBytes(of: 0x80, toByteOffset: i + 1, as: UInt8.self)
                p.storeBytes(of: 0x80, toByteOffset: i + 2, as: UInt8.self)
                p.storeBytes(of: 0xFF, toByteOffset: i + 3, as: UInt8.self)
                i += 4
            }
        }
        try buf.write(to: colorsURL, options: .atomic)
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
