// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import XCTest
@testable import TribeBrainView

final class AnimationBundleTests: XCTestCase {

    static func animationFixtureDirectory() throws -> URL {
        guard let url = Bundle.module.url(forResource: "Fixtures", withExtension: nil) else {
            throw XCTSkip("Fixtures directory not present in test bundle")
        }
        return url.appendingPathComponent("animation")
    }

    func testParsesAnimationMetaFromFixture() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)

        XCTAssertEqual(bundle.frameCount, 31)
        XCTAssertEqual(bundle.vertexCount, 20484)
        // Meta does not specify fps; should fall back to the documented default.
        XCTAssertEqual(bundle.fps, BrainAnimationBundle.defaultFPS)
        XCTAssertEqual(bundle.frameFilenames.count, 31)
        XCTAssertEqual(bundle.frameFilenames.first, "brain_colors_t0000.bin")
        XCTAssertEqual(bundle.frameFilenames.last,  "brain_colors_t0030.bin")
    }

    func testFrameURLForFirstAndLastIndexResolvesToExistingFiles() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)

        let first = bundle.frameURL(at: 0)
        let last = bundle.frameURL(at: bundle.frameCount - 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: last.path))
        XCTAssertEqual(first.lastPathComponent, "brain_colors_t0000.bin")
        XCTAssertEqual(last.lastPathComponent, "brain_colors_t0030.bin")
    }

    func testFrameURLOutOfRangeClamps() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)

        // Documented behavior: clamp to [0, frameCount-1].
        XCTAssertEqual(bundle.frameURL(at: -5).lastPathComponent, "brain_colors_t0000.bin")
        XCTAssertEqual(bundle.frameURL(at: 9999).lastPathComponent, "brain_colors_t0030.bin")
    }

    func testFrameCountMatchesFilesOnDisk() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)

        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        let perFrame = names.filter { $0.hasPrefix("brain_colors_t") && $0.hasSuffix(".bin") }
        XCTAssertEqual(bundle.frameCount, perFrame.count)
    }

    func testInitThrowsWhenMetaMissing() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("brain-anim-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        XCTAssertThrowsError(try BrainAnimationBundle(directory: tmp)) { error in
            guard case BrainMeshError.missingFile = error else {
                XCTFail("Expected missingFile, got \(error)")
                return
            }
        }
    }

    func testControllerHonorsExplicitFPSOverride() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)
        // Bundle default is 10.0; explicit param wins.
        let ctrl = BrainAnimationController(bundle: bundle, fps: 24.0)
        XCTAssertEqual(ctrl.fps, 24.0)
    }

    func testControllerFallsBackToBundleFPSWhenParamIsNaN() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)
        let ctrl = BrainAnimationController(bundle: bundle, fps: .nan)
        XCTAssertEqual(ctrl.fps, bundle.fps)
    }

    func testControllerSeekClampsToValidRange() throws {
        let dir = try Self.animationFixtureDirectory()
        let bundle = try BrainAnimationBundle(directory: dir)
        let ctrl = BrainAnimationController(bundle: bundle, fps: 10)

        ctrl.seek(to: -10)
        XCTAssertEqual(ctrl.currentFrame, 0)
        ctrl.seek(to: 10_000)
        XCTAssertEqual(ctrl.currentFrame, bundle.frameCount - 1)
        ctrl.seek(to: 5)
        XCTAssertEqual(ctrl.currentFrame, 5)
    }
}
