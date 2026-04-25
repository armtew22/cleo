// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - ColorsResponse

/// Decoded response from `POST /v1/inference/colors`.
public struct ColorsResponse {
    public let windowId: String
    public let frameCount: Int
    public let vmin: Double
    public let vmax: Double
    public let method: String
    /// Raw uint8 RGBA × 20484 × frameCount bytes.
    public let buffer: Data

    public init(
        windowId: String,
        frameCount: Int,
        vmin: Double,
        vmax: Double,
        method: String,
        buffer: Data
    ) {
        self.windowId = windowId
        self.frameCount = frameCount
        self.vmin = vmin
        self.vmax = vmax
        self.method = method
        self.buffer = buffer
    }
}

// MARK: - InferenceEnvelope

/// Decoded response from `POST /v1/inference/full`: colors buffer + qualitative report.
public struct InferenceEnvelope {
    public let windowId: String
    /// Echoed from the JSON envelope (vmin/vmax/method) and populated with the
    /// colors bytes fetched in parallel via `colors_url`.
    public let colors: ColorsResponse
    public let report: BrainReport
    public let colorsURL: String
    public let vmin: Double
    public let vmax: Double
    public let method: String

    public init(
        windowId: String,
        colors: ColorsResponse,
        report: BrainReport,
        colorsURL: String,
        vmin: Double,
        vmax: Double,
        method: String
    ) {
        self.windowId = windowId
        self.colors = colors
        self.report = report
        self.colorsURL = colorsURL
        self.vmin = vmin
        self.vmax = vmax
        self.method = method
    }
}

// MARK: - BrainMeshClientError

/// Errors thrown by ``BrainMeshClient`` during network or decoding operations.
public enum BrainMeshClientError: Error, Equatable {
    case server(status: Int, body: String?)
    case invalidResponse(String)
    case missingHeader(String)
    case tarMalformed(String)
}

// MARK: - Internal envelope decoder

/// Codable mirror of the /v1/inference/full JSON body.
/// We decode the envelope locally and then GET colors_url in parallel.
private struct _FullEnvelopeJSON: Decodable {
    let window_id: String
    let colors_url: String
    let report: BrainReport          // best-effort decoder; fields may be absent
    let vmin: Double
    let vmax: Double
    let method: String
}

// MARK: - BrainMeshClient

/// Talks to the FastAPI server (Phase 6b: tribe_backend/api/mesh_routes.py).
///
/// Server contract used here:
///   GET  /v1/mesh/static?surface=...   → tar(brain_vertices.bin,
///                                            brain_normals.bin,
///                                            brain_faces.bin,
///                                            brain_meta.json)
///                                       ETag: "fsaverage5-<surface>-v1"
///   POST /v1/inference/colors          → multipart in (image,audio,text,
///                                            method,vmin,vmax,cmap)
///                                            → octet-stream out + X-Window-Id /
///                                              X-Tribe-Method / X-Vmin / X-Vmax /
///                                              X-Frame-Count headers.
///   POST /v1/inference/full            → JSON envelope; callers fetch
///                                            colors_url separately.
///   GET  /v1/inference/colors/{wid}    → cached colors bytes.
public actor BrainMeshClient {
    public let baseURL: URL
    public let session: URLSession
    public let cacheDirectory: URL

    public init(
        baseURL: URL,
        session: URLSession = .shared,
        cacheDirectory: URL? = nil
    ) {
        self.baseURL = baseURL
        self.session = session
        self.cacheDirectory = cacheDirectory ?? Self.defaultCacheDirectory()
    }

    private static func defaultCacheDirectory() -> URL {
        let fm = FileManager.default
        let base = (try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("TribeBrainView", isDirectory: true)
    }

    // MARK: - Bootstrap

    /// Fetch (or reuse cached) static mesh archive. Returns the local directory
    /// containing brain_vertices.bin / brain_normals.bin / brain_faces.bin /
    /// brain_meta.json.
    public func bootstrapStaticMesh(surface: String = "pial") async throws -> URL {
        let surfaceDir = cacheDirectory
            .appendingPathComponent("static", isDirectory: true)
            .appendingPathComponent(surface, isDirectory: true)
        try FileManager.default.createDirectory(
            at: surfaceDir,
            withIntermediateDirectories: true
        )

        var url = baseURL.appendingPathComponent("v1/mesh/static")
        if var comps = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            comps.queryItems = [URLQueryItem(name: "surface", value: surface)]
            if let resolved = comps.url { url = resolved }
        }

        var req = URLRequest(url: url)
        req.httpMethod = "GET"

        let etagKey = "TribeBrainView.staticMesh.etag.\(surface)"
        let storedEtag = UserDefaults.standard.string(forKey: etagKey)
        if let storedEtag,
           hasCachedSurface(at: surfaceDir) {
            req.setValue(storedEtag, forHTTPHeaderField: "If-None-Match")
        }

        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw BrainMeshClientError.invalidResponse("expected HTTPURLResponse")
        }

        switch http.statusCode {
        case 304:
            // Cache hit; trust on-disk files.
            guard hasCachedSurface(at: surfaceDir) else {
                // Server said 304 but we lost local bytes; force a fresh GET.
                UserDefaults.standard.removeObject(forKey: etagKey)
                return try await bootstrapStaticMesh(surface: surface)
            }
            return surfaceDir
        case 200:
            try untar(data, into: surfaceDir)
            if let etag = http.value(forHTTPHeaderField: "ETag")
                ?? http.value(forHTTPHeaderField: "Etag") {
                UserDefaults.standard.set(etag, forKey: etagKey)
            }
            return surfaceDir
        default:
            let body = String(data: data, encoding: .utf8)
            throw BrainMeshClientError.server(status: http.statusCode, body: body)
        }
    }

    private func hasCachedSurface(at dir: URL) -> Bool {
        let fm = FileManager.default
        let required = ["brain_vertices.bin", "brain_faces.bin", "brain_meta.json"]
        for name in required {
            if !fm.fileExists(atPath: dir.appendingPathComponent(name).path) {
                return false
            }
        }
        return true
    }

    // MARK: - Hot path

    public func fetchColors(
        image: Data,
        audio: Data? = nil,
        text: String? = nil,
        method: AggregationMethod = .windowMean,
        vmin: Double? = nil,
        vmax: Double? = nil,
        cmap: String? = nil
    ) async throws -> ColorsResponse {
        let req = try buildInferenceRequest(
            path: "v1/inference/colors",
            image: image,
            audio: audio,
            text: text,
            method: method,
            vmin: vmin,
            vmax: vmax,
            cmap: cmap
        )
        let (data, response) = try await session.data(for: req)
        return try parseColorsResponse(data: data, response: response)
    }

    public func fetchFull(
        image: Data,
        audio: Data? = nil,
        text: String? = nil,
        method: AggregationMethod = .windowMean
    ) async throws -> InferenceEnvelope {
        let req = try buildInferenceRequest(
            path: "v1/inference/full",
            image: image,
            audio: audio,
            text: text,
            method: method,
            vmin: nil,
            vmax: nil,
            cmap: nil
        )
        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw BrainMeshClientError.invalidResponse("expected HTTPURLResponse")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw BrainMeshClientError.server(
                status: http.statusCode,
                body: String(data: data, encoding: .utf8)
            )
        }
        let env: _FullEnvelopeJSON
        do {
            env = try JSONDecoder().decode(_FullEnvelopeJSON.self, from: data)
        } catch {
            throw BrainMeshClientError.invalidResponse("envelope decode failed: \(error)")
        }

        // Fetch the colors bytes in parallel via the colors_url path.
        let colorsURL = try resolveColorsURL(env.colors_url)
        async let colorsBytes: Data = fetchColorsBytesByURL(colorsURL)
        let buffer = try await colorsBytes

        let colors = ColorsResponse(
            windowId: env.window_id,
            frameCount: max(1, buffer.count / 81936),
            vmin: env.vmin,
            vmax: env.vmax,
            method: env.method,
            buffer: buffer
        )
        return InferenceEnvelope(
            windowId: env.window_id,
            colors: colors,
            report: env.report,
            colorsURL: env.colors_url,
            vmin: env.vmin,
            vmax: env.vmax,
            method: env.method
        )
    }

    private func resolveColorsURL(_ path: String) throws -> URL {
        if let abs = URL(string: path), abs.scheme != nil { return abs }
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        return baseURL.appendingPathComponent(trimmed)
    }

    private func fetchColorsBytesByURL(_ url: URL) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw BrainMeshClientError.invalidResponse("expected HTTPURLResponse")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw BrainMeshClientError.server(
                status: http.statusCode,
                body: String(data: data, encoding: .utf8)
            )
        }
        return data
    }

    // MARK: - Request building

    private func buildInferenceRequest(
        path: String,
        image: Data,
        audio: Data?,
        text: String?,
        method: AggregationMethod,
        vmin: Double?,
        vmax: Double?,
        cmap: String?
    ) throws -> URLRequest {
        let url = baseURL.appendingPathComponent(path)
        var req = URLRequest(url: url)
        req.httpMethod = "POST"

        let boundary = "TribeBrainView-\(UUID().uuidString)"
        req.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        var builder = MultipartBodyBuilder(boundary: boundary)
        builder.appendFile(
            name: "image",
            filename: "frame.jpg",
            contentType: "image/jpeg",
            data: image
        )
        if let audio {
            builder.appendFile(
                name: "audio",
                filename: "audio.wav",
                contentType: "audio/wav",
                data: audio
            )
        }
        if let text {
            builder.appendField(name: "text", value: text)
        }
        builder.appendField(name: "method", value: method.rawValue)
        if let vmin { builder.appendField(name: "vmin", value: String(vmin)) }
        if let vmax { builder.appendField(name: "vmax", value: String(vmax)) }
        if let cmap { builder.appendField(name: "cmap", value: cmap) }
        req.httpBody = builder.finalize()
        return req
    }

    private func parseColorsResponse(
        data: Data,
        response: URLResponse
    ) throws -> ColorsResponse {
        guard let http = response as? HTTPURLResponse else {
            throw BrainMeshClientError.invalidResponse("expected HTTPURLResponse")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw BrainMeshClientError.server(
                status: http.statusCode,
                body: String(data: data, encoding: .utf8)
            )
        }
        func require(_ name: String) throws -> String {
            guard let v = http.value(forHTTPHeaderField: name) else {
                throw BrainMeshClientError.missingHeader(name)
            }
            return v
        }
        let windowId = try require("X-Window-Id")
        let methodStr = try require("X-Tribe-Method")
        let vminStr = try require("X-Vmin")
        let vmaxStr = try require("X-Vmax")
        let frameStr = try require("X-Frame-Count")
        guard let vmin = Double(vminStr) else {
            throw BrainMeshClientError.invalidResponse("X-Vmin not a Double: \(vminStr)")
        }
        guard let vmax = Double(vmaxStr) else {
            throw BrainMeshClientError.invalidResponse("X-Vmax not a Double: \(vmaxStr)")
        }
        guard let frameCount = Int(frameStr) else {
            throw BrainMeshClientError.invalidResponse("X-Frame-Count not an Int: \(frameStr)")
        }
        return ColorsResponse(
            windowId: windowId,
            frameCount: frameCount,
            vmin: vmin,
            vmax: vmax,
            method: methodStr,
            buffer: data
        )
    }

    // MARK: - Tar untar

    /// Untar a POSIX/ustar archive into `directory`. Only the file entries needed
    /// for this client are extracted; subdirectories are flattened by basename.
    private func untar(_ archive: Data, into directory: URL) throws {
        let blockSize = 512
        guard archive.count >= blockSize else {
            throw BrainMeshClientError.tarMalformed("archive < 512 bytes")
        }
        var offset = 0
        let fm = FileManager.default
        while offset + blockSize <= archive.count {
            let header = archive.subdata(in: offset..<offset + blockSize)
            // Empty (all-zero) block signals end-of-archive.
            if header.allSatisfy({ $0 == 0 }) { break }

            let name = Self.readCString(header, range: 0..<100)
            let sizeStr = Self.readCString(header, range: 124..<136)
            // Tar size field is octal, possibly space- or null-padded.
            let cleanedSize = sizeStr.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let size = Int(cleanedSize, radix: 8) else {
                throw BrainMeshClientError.tarMalformed("bad size field: \(sizeStr)")
            }
            // typeflag at offset 156. '0' or NUL → regular file. '5' → dir.
            let typeByte = header[156]
            offset += blockSize

            if size > 0 && (typeByte == 0x30 || typeByte == 0x00) && !name.isEmpty {
                guard offset + size <= archive.count else {
                    throw BrainMeshClientError.tarMalformed("entry \(name) exceeds archive")
                }
                let payload = archive.subdata(in: offset..<offset + size)
                let basename = (name as NSString).lastPathComponent
                let outURL = directory.appendingPathComponent(basename)
                try payload.write(to: outURL, options: .atomic)
            }

            // Advance over the data + zero-padding to next 512-aligned boundary.
            let padded = ((size + blockSize - 1) / blockSize) * blockSize
            offset += padded
            _ = fm  // silence unused-warning if elision occurs
        }
    }

    private static func readCString(_ data: Data, range: Range<Int>) -> String {
        let slice = data.subdata(in: range)
        // Trim at first NUL.
        let bytes = slice.prefix { $0 != 0 }
        return String(data: Data(bytes), encoding: .utf8) ?? ""
    }
}

// MARK: - MultipartBodyBuilder

/// Minimal multipart/form-data body builder (no streaming — bodies in this
/// app are small JPEGs + a handful of form fields).
struct MultipartBodyBuilder {
    let boundary: String
    private var data = Data()

    init(boundary: String) {
        self.boundary = boundary
    }

    mutating func appendField(name: String, value: String) {
        data.append("--\(boundary)\r\n")
        data.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        data.append("\(value)\r\n")
    }

    mutating func appendFile(
        name: String,
        filename: String,
        contentType: String,
        data fileData: Data
    ) {
        data.append("--\(boundary)\r\n")
        data.append(
            "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n"
        )
        data.append("Content-Type: \(contentType)\r\n\r\n")
        data.append(fileData)
        data.append("\r\n")
    }

    mutating func finalize() -> Data {
        data.append("--\(boundary)--\r\n")
        return data
    }
}

private extension Data {
    mutating func append(_ s: String) {
        if let d = s.data(using: .utf8) { append(d) }
    }
}
