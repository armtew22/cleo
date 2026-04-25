// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import XCTest
@testable import TribeBrainView

// MARK: - Mock URLProtocol

/// Thread-safe stub that returns canned responses keyed by path. Captures the
/// most recent request body for shape assertions.
final class StubURLProtocol: URLProtocol {
    struct Stub {
        let status: Int
        let headers: [String: String]
        let body: Data
    }

    static let lock = NSLock()
    static var stubsByPath: [String: Stub] = [:]
    static var lastRequest: URLRequest?
    static var lastBody: Data?

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        stubsByPath = [:]
        lastRequest = nil
        lastBody = nil
    }

    static func register(path: String, stub: Stub) {
        lock.lock()
        defer { lock.unlock() }
        stubsByPath[path] = stub
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.lock()
        Self.lastRequest = request
        // URLProtocol consumes httpBodyStream lazily; capture explicitly.
        if let stream = request.httpBodyStream {
            Self.lastBody = Self.readAll(stream: stream)
        } else {
            Self.lastBody = request.httpBody
        }
        let key = request.url?.path ?? ""
        let stub = Self.stubsByPath[key]
        Self.lock.unlock()

        let url = request.url ?? URL(string: "http://localhost/")!
        let (status, headers, body): (Int, [String: String], Data) = {
            if let stub { return (stub.status, stub.headers, stub.body) }
            return (404, [:], Data())
        }()
        let resp = HTTPURLResponse(
            url: url,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: headers
        )!
        client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
        if !body.isEmpty {
            client?.urlProtocol(self, didLoad: body)
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func readAll(stream: InputStream) -> Data {
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufSize = 4096
        var buf = [UInt8](repeating: 0, count: bufSize)
        while stream.hasBytesAvailable {
            let n = stream.read(&buf, maxLength: bufSize)
            if n <= 0 { break }
            data.append(buf, count: n)
        }
        return data
    }
}

// MARK: - Tar test helper

/// Build a minimal POSIX/ustar archive from in-memory entries.
enum TestTar {
    static func build(_ entries: [(name: String, data: Data)]) -> Data {
        var archive = Data()
        for (name, payload) in entries {
            archive.append(makeHeader(name: name, size: payload.count))
            archive.append(payload)
            // Pad to next 512 boundary.
            let pad = (512 - (payload.count % 512)) % 512
            if pad > 0 { archive.append(Data(repeating: 0, count: pad)) }
        }
        // Two zero blocks signal end-of-archive.
        archive.append(Data(repeating: 0, count: 1024))
        return archive
    }

    private static func makeHeader(name: String, size: Int) -> Data {
        var h = [UInt8](repeating: 0, count: 512)
        // name: bytes 0..<100
        let nameBytes = Array(name.utf8.prefix(100))
        for (i, b) in nameBytes.enumerated() { h[i] = b }
        // mode: octal "0000644\0" at 100..108
        let mode = Array("0000644".utf8)
        for (i, b) in mode.enumerated() { h[100 + i] = b }
        // size: octal at 124..136 (11 chars + NUL)
        let sizeOct = String(size, radix: 8).leftPadded(to: 11, with: "0")
        for (i, b) in Array(sizeOct.utf8).enumerated() { h[124 + i] = b }
        // mtime at 136..148
        let mtime = String(0, radix: 8).leftPadded(to: 11, with: "0")
        for (i, b) in Array(mtime.utf8).enumerated() { h[136 + i] = b }
        // typeflag '0' (regular file) at 156
        h[156] = 0x30
        // ustar magic+version
        let magic = Array("ustar\0".utf8)
        for (i, b) in magic.enumerated() { h[257 + i] = b }
        h[263] = 0x30; h[264] = 0x30  // "00"
        // checksum: fill with spaces, sum, then write octal in 148..155
        for i in 148..<156 { h[i] = 0x20 }
        let sum = h.reduce(0, { $0 + Int($1) })
        let chk = String(sum, radix: 8).leftPadded(to: 6, with: "0")
        for (i, b) in Array(chk.utf8).enumerated() { h[148 + i] = b }
        h[148 + 6] = 0       // NUL
        h[148 + 7] = 0x20    // space

        return Data(h)
    }
}

private extension String {
    func leftPadded(to length: Int, with char: Character) -> String {
        if count >= length { return String(suffix(length)) }
        return String(repeating: String(char), count: length - count) + self
    }
}

// MARK: - Tests

final class MeshClientTests: XCTestCase {
    var session: URLSession!
    let baseURL = URL(string: "http://localhost:8000")!

    override func setUp() {
        super.setUp()
        StubURLProtocol.reset()
        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = [StubURLProtocol.self]
        session = URLSession(configuration: cfg)
        // Clear any persisted ETag from prior runs.
        UserDefaults.standard.removeObject(forKey: "TribeBrainView.staticMesh.etag.pial")
    }

    override func tearDown() {
        StubURLProtocol.reset()
        UserDefaults.standard.removeObject(forKey: "TribeBrainView.staticMesh.etag.pial")
        super.tearDown()
    }

    // MARK: bootstrap

    func test_bootstrap_200_untarsAndStoresEtag() async throws {
        let cacheDir = uniqueCacheDir()
        let client = BrainMeshClient(
            baseURL: baseURL,
            session: session,
            cacheDirectory: cacheDir
        )

        let metaJSON = #"{"format":"tribe_brain_mesh_v1","surface_type":"pial","vertex_count":4,"face_count":2,"bytes_per_vertex":12,"bytes_per_face":12,"bytes_per_color":4,"files":{"vertices":"brain_vertices.bin","faces":"brain_faces.bin"}}"#
        let tar = TestTar.build([
            ("brain_vertices.bin", Data(repeating: 0xAA, count: 48)),
            ("brain_normals.bin",  Data(repeating: 0xBB, count: 48)),
            ("brain_faces.bin",    Data(repeating: 0xCC, count: 24)),
            ("brain_meta.json",    Data(metaJSON.utf8)),
        ])
        StubURLProtocol.register(
            path: "/v1/mesh/static",
            stub: .init(status: 200,
                        headers: ["ETag": "\"fsaverage5-pial-v1\""],
                        body: tar)
        )

        let dir = try await client.bootstrapStaticMesh(surface: "pial")
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("brain_meta.json").path))
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("brain_vertices.bin").path))
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("brain_faces.bin").path))
        XCTAssertEqual(
            UserDefaults.standard.string(forKey: "TribeBrainView.staticMesh.etag.pial"),
            "\"fsaverage5-pial-v1\""
        )
    }

    func test_bootstrap_304_reusesCachedDir() async throws {
        let cacheDir = uniqueCacheDir()
        let client = BrainMeshClient(
            baseURL: baseURL,
            session: session,
            cacheDirectory: cacheDir
        )

        // Pre-stage a cached surface dir.
        let surfaceDir = cacheDir
            .appendingPathComponent("static")
            .appendingPathComponent("pial")
        try FileManager.default.createDirectory(
            at: surfaceDir, withIntermediateDirectories: true
        )
        try Data(repeating: 0, count: 4).write(
            to: surfaceDir.appendingPathComponent("brain_vertices.bin")
        )
        try Data(repeating: 0, count: 4).write(
            to: surfaceDir.appendingPathComponent("brain_faces.bin")
        )
        try Data(#"{"format":"x"}"#.utf8).write(
            to: surfaceDir.appendingPathComponent("brain_meta.json")
        )
        UserDefaults.standard.set(
            "\"fsaverage5-pial-v1\"",
            forKey: "TribeBrainView.staticMesh.etag.pial"
        )

        StubURLProtocol.register(
            path: "/v1/mesh/static",
            stub: .init(status: 304, headers: [:], body: Data())
        )

        let dir = try await client.bootstrapStaticMesh(surface: "pial")
        XCTAssertEqual(dir.standardizedFileURL, surfaceDir.standardizedFileURL)
        // If-None-Match must have been sent.
        let sent = StubURLProtocol.lastRequest?.value(forHTTPHeaderField: "If-None-Match")
        XCTAssertEqual(sent, "\"fsaverage5-pial-v1\"")
    }

    // MARK: fetchColors

    func test_fetchColors_200_populatesResponse() async throws {
        let client = BrainMeshClient(
            baseURL: baseURL,
            session: session,
            cacheDirectory: uniqueCacheDir()
        )
        let body = Data(repeating: 0x12, count: 81936)
        StubURLProtocol.register(
            path: "/v1/inference/colors",
            stub: .init(
                status: 200,
                headers: [
                    "X-Window-Id": "abc123",
                    "X-Tribe-Method": "window_mean",
                    "X-Vmin": "-3.0",
                    "X-Vmax": "3.0",
                    "X-Frame-Count": "1",
                ],
                body: body
            )
        )
        let r = try await client.fetchColors(image: Data([0xFF, 0xD8, 0xFF]))
        XCTAssertEqual(r.windowId, "abc123")
        XCTAssertEqual(r.frameCount, 1)
        XCTAssertEqual(r.vmin, -3.0, accuracy: 1e-9)
        XCTAssertEqual(r.vmax,  3.0, accuracy: 1e-9)
        XCTAssertEqual(r.method, "window_mean")
        XCTAssertEqual(r.buffer.count, 81936)
    }

    func test_fetchColors_5xx_throwsServer() async {
        let client = BrainMeshClient(
            baseURL: baseURL,
            session: session,
            cacheDirectory: uniqueCacheDir()
        )
        StubURLProtocol.register(
            path: "/v1/inference/colors",
            stub: .init(status: 503, headers: [:], body: Data("nope".utf8))
        )
        do {
            _ = try await client.fetchColors(image: Data([0x00]))
            XCTFail("expected throw")
        } catch let BrainMeshClientError.server(status, body) {
            XCTAssertEqual(status, 503)
            XCTAssertEqual(body, "nope")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func test_fetchColors_missingWindowId_throwsMissingHeader() async {
        let client = BrainMeshClient(
            baseURL: baseURL,
            session: session,
            cacheDirectory: uniqueCacheDir()
        )
        StubURLProtocol.register(
            path: "/v1/inference/colors",
            stub: .init(
                status: 200,
                headers: [
                    "X-Tribe-Method": "window_mean",
                    "X-Vmin": "-3.0",
                    "X-Vmax": "3.0",
                    "X-Frame-Count": "1",
                ],
                body: Data(repeating: 0, count: 81936)
            )
        )
        do {
            _ = try await client.fetchColors(image: Data([0x00]))
            XCTFail("expected throw")
        } catch let BrainMeshClientError.missingHeader(name) {
            XCTAssertEqual(name, "X-Window-Id")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func test_fetchColors_multipartBody_containsImageAndMethod() async throws {
        let client = BrainMeshClient(
            baseURL: baseURL,
            session: session,
            cacheDirectory: uniqueCacheDir()
        )
        StubURLProtocol.register(
            path: "/v1/inference/colors",
            stub: .init(
                status: 200,
                headers: [
                    "X-Window-Id": "w",
                    "X-Tribe-Method": "peak",
                    "X-Vmin": "0",
                    "X-Vmax": "1",
                    "X-Frame-Count": "1",
                ],
                body: Data(repeating: 0x00, count: 81936)
            )
        )
        _ = try await client.fetchColors(
            image: Data([0xAA, 0xBB]),
            method: .peak
        )
        let body = StubURLProtocol.lastBody ?? Data()
        let asString = String(data: body, encoding: .utf8) ?? ""
        XCTAssertTrue(asString.contains(#"name="image""#),
                      "multipart body missing name=\"image\"; got: \(asString.prefix(400))")
        XCTAssertTrue(asString.contains(#"name="method""#),
                      "multipart body missing name=\"method\"")
        XCTAssertTrue(asString.contains("peak"),
                      "multipart body missing method value 'peak'")
        let ct = StubURLProtocol.lastRequest?.value(forHTTPHeaderField: "Content-Type") ?? ""
        XCTAssertTrue(ct.hasPrefix("multipart/form-data; boundary="), "content-type=\(ct)")
    }

    // MARK: AggregationMethod rawValues

    func test_aggregationMethod_rawValuesMatchServer() {
        XCTAssertEqual(AggregationMethod.windowMean.rawValue, "window_mean")
        XCTAssertEqual(AggregationMethod.peak.rawValue, "peak")
        XCTAssertEqual(AggregationMethod.peakWindow.rawValue, "peak_window")
    }

    // MARK: helpers

    private func uniqueCacheDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("TribeBrainView-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true
        )
        return url
    }
}
