// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation
import SceneKit
import CoreGraphics

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - BrainSceneCoordinator

/// Owns the SCNView lifecycle: scene setup, lighting, camera, and color-swap on frame changes.
///
/// Lifecycle:
///   1. `init(source:configuration:)`
///   2. SwiftUI representable creates the SCNView, then calls `attach(_:)` and
///      `setGeometry(_:vertexCount:)`.
///   3. As bindings change, the representable calls `swapColors(...)` and
///      `applyCameraPreset(...)` / `applyConfiguration(...)`.
public final class BrainSceneCoordinator: NSObject {
    public let source: BrainMeshSource
    public private(set) var configuration: BrainViewConfiguration

    public private(set) var scnView: SCNView?
    public let scene: SCNScene
    public let cameraNode: SCNNode
    public let brainNode: SCNNode
    public let ambientNode: SCNNode
    public let keyLightNode: SCNNode

    /// Vertex count carried by the currently installed geometry. Used by `swapColors`.
    public private(set) var vertexCount: Int = 0

    /// Last frame index applied via `swapColors(fromAnimationBundle:frame:)`. -1 means none.
    public private(set) var lastAppliedFrame: Int = -1

    public init(source: BrainMeshSource, configuration: BrainViewConfiguration) {
        self.source = source
        self.configuration = configuration

        // Scene + nodes (built once; applyConfiguration mutates properties on these).
        let scene = SCNScene()
        self.scene = scene

        let camera = SCNCamera()
        camera.fieldOfView = 30
        camera.zNear = 1
        camera.zFar = 5000
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        scene.rootNode.addChildNode(cameraNode)
        self.cameraNode = cameraNode

        let brainNode = SCNNode()
        brainNode.name = "brain"
        scene.rootNode.addChildNode(brainNode)
        self.brainNode = brainNode

        // Default lighting rig: ambient fill + a key directional light.
        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 400
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)
        self.ambientNode = ambientNode

        let key = SCNLight()
        key.type = .directional
        key.intensity = 800
        let keyNode = SCNNode()
        keyNode.light = key
        // Aim from upper-front-right.
        keyNode.position = SCNVector3(200, 200, 200)
        keyNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(keyNode)
        self.keyLightNode = keyNode

        super.init()
    }

    // MARK: - Attach

    /// Attach to an SCNView created by the SwiftUI representable.
    public func attach(_ view: SCNView) {
        self.scnView = view
        view.scene = scene
        applyConfiguration(configuration)
        applyCameraPreset(configuration.cameraPreset)
    }

    // MARK: - Geometry

    /// Install (or replace) the brain geometry. Call once from `make*View`.
    public func setGeometry(_ geometry: SCNGeometry, vertexCount: Int) {
        configureMaterial(on: geometry)
        brainNode.geometry = geometry
        self.vertexCount = vertexCount
    }

    // MARK: - Color swap

    /// Swap only the per-vertex color source from a binary file.
    /// Vertex/normal sources keep object identity (BrainMeshLoader.updateColors
    /// reuses them); we only replace `brainNode.geometry`.
    public func swapColors(from url: URL, vertexCount: Int) throws {
        guard let current = brainNode.geometry else {
            throw BrainMeshError.malformed("BrainSceneCoordinator.swapColors called before setGeometry")
        }
        let updated = try BrainMeshLoader.updateColors(
            on: current,
            from: url,
            vertexCount: vertexCount
        )
        // Preserve material configuration on the rebuilt geometry.
        configureMaterial(on: updated)
        brainNode.geometry = updated
    }

    /// Convenience: resolve the per-frame URL from the bundle and swap.
    public func swapColors(fromAnimationBundle bundle: BrainAnimationBundle, frame: Int) throws {
        let url = bundle.frameURL(at: frame)
        try swapColors(from: url, vertexCount: vertexCount)
        lastAppliedFrame = frame
    }

    // MARK: - Configuration

    /// Apply (or re-apply) a `BrainViewConfiguration`. Safe to call repeatedly.
    public func applyConfiguration(_ configuration: BrainViewConfiguration) {
        self.configuration = configuration

        if let view = scnView {
            view.allowsCameraControl = configuration.allowsCameraControl
            #if canImport(UIKit)
            view.backgroundColor = brainColorToUIColor(configuration.background)
            #elseif canImport(AppKit)
            view.backgroundColor = brainColorToNSColor(configuration.background)
            #endif
        }

        if let geometry = brainNode.geometry {
            configureMaterial(on: geometry)
        }
    }

    // MARK: - Camera

    /// Position the camera according to the given preset.
    public func applyCameraPreset(_ preset: CameraPreset) {
        let descriptor = cameraDescriptor(for: preset)

        if preset == .custom {
            // Host owns the transform; do not touch.
            return
        }

        // SCNVector3 components are Float on iOS/visionOS, CGFloat on macOS;
        // round-trip through Float to keep this branch portable.
        let r = brainCameraRadius
        let vx = Float(descriptor.viewVector.x) * r
        let vy = Float(descriptor.viewVector.y) * r
        let vz = Float(descriptor.viewVector.z) * r
        cameraNode.position = SCNVector3(vx, vy, vz)
        cameraNode.look(
            at: SCNVector3(0, 0, 0),
            up: descriptor.upVector,
            localFront: SCNNode.localFront
        )
        cameraNode.camera?.fieldOfView = descriptor.fov
    }

    // MARK: - Material

    private func configureMaterial(on geometry: SCNGeometry) {
        let mat = geometry.firstMaterial ?? SCNMaterial()
        switch configuration.lighting {
        case .physicallyBased:
            mat.lightingModel = .physicallyBased
            mat.roughness.contents = 0.6
            mat.metalness.contents = 0.0
        case .lambert:
            mat.lightingModel = .lambert
        case .blinn:
            mat.lightingModel = .blinn
        case .phong:
            mat.lightingModel = .phong
        case .constant:
            mat.lightingModel = .constant
        }
        mat.isDoubleSided = false
        if geometry.firstMaterial == nil {
            geometry.materials = [mat]
        }
    }

    // MARK: - Platform color bridge

    #if canImport(UIKit)
    private func brainColorToUIColor(_ c: BrainColor) -> UIColor {
        UIColor(
            red: CGFloat(c.red),
            green: CGFloat(c.green),
            blue: CGFloat(c.blue),
            alpha: CGFloat(c.alpha)
        )
    }
    #elseif canImport(AppKit)
    private func brainColorToNSColor(_ c: BrainColor) -> NSColor {
        NSColor(
            calibratedRed: CGFloat(c.red),
            green: CGFloat(c.green),
            blue: CGFloat(c.blue),
            alpha: CGFloat(c.alpha)
        )
    }
    #endif
}
