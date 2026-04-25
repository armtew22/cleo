// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - ColorsResponse

public struct ColorsResponse {
    public let windowId: String
    public let frameCount: Int
    public let vmin: Double
    public let vmax: Double
    /// Raw uint8 RGBA × 20484 × frameCount bytes.
    public let buffer: Data

    public init(windowId: String, frameCount: Int, vmin: Double, vmax: Double, buffer: Data) {
        self.windowId = windowId
        self.frameCount = frameCount
        self.vmin = vmin
        self.vmax = vmax
        self.buffer = buffer
    }
}

// MARK: - InferenceEnvelope

public struct InferenceEnvelope {
    public let windowId: String
    public let colors: ColorsResponse
    public let report: BrainReport

    public init(windowId: String, colors: ColorsResponse, report: BrainReport) {
        self.windowId = windowId
        self.colors = colors
        self.report = report
    }
}

// MARK: - BrainMeshClient

public actor BrainMeshClient {
    public let baseURL: URL
    public let session: URLSession
    public let cacheDirectory: URL?

    public init(
        baseURL: URL,
        session: URLSession = .shared,
        cacheDirectory: URL? = nil
    ) {
        self.baseURL = baseURL
        self.session = session
        self.cacheDirectory = cacheDirectory
    }

    /// Fetch and cache the static mesh bundle (vertices, normals, faces, meta).
    /// Returns the local directory URL where files were unpacked.
    public func bootstrapStaticMesh() async throws -> URL {
        fatalError("unimplemented")
    }

    /// POST an image (and optional audio/text) and receive a color buffer response.
    public func fetchColors(
        image: Data,
        audio: Data? = nil,
        text: String? = nil,
        method: AggregationMethod = .windowMean
    ) async throws -> ColorsResponse {
        fatalError("unimplemented")
    }

    /// POST and receive a full inference envelope (colors + region report).
    public func fetchFull(
        image: Data,
        audio: Data? = nil,
        text: String? = nil,
        method: AggregationMethod = .windowMean
    ) async throws -> InferenceEnvelope {
        fatalError("unimplemented")
    }
}
