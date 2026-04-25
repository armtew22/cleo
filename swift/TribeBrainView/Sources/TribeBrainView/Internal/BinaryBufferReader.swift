// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BinaryBufferReader (internal)

/// Wraps a `Data` buffer and provides stride-validated typed reads.
///
/// Used by `BrainMeshLoader` to validate the on-disk shape of `brain_*.bin` files
/// before handing the bytes to SceneKit.
struct BinaryBufferReader {
    let data: Data
    let label: String

    init(data: Data, label: String = "buffer") {
        self.data = data
        self.label = label
    }

    /// Validate that the buffer is exactly `expected` bytes.
    func validateSize(expected: Int) throws {
        guard data.count == expected else {
            throw BrainMeshError.sizeMismatch(expected: expected, actual: data.count)
        }
    }

    /// Validate that the buffer contains exactly `count` elements of `stride` bytes each.
    func validate(expectedCount: Int, stride: Int) throws {
        let expected = expectedCount * stride
        guard data.count == expected else {
            throw BrainMeshError.sizeMismatch(expected: expected, actual: data.count)
        }
    }

    /// Read all elements as an `Array<T>` by reinterpreting the underlying bytes.
    /// Caller is responsible for `T` having the correct in-memory layout (POD types only).
    func elements<T>(as type: T.Type, stride: Int, count: Int) throws -> [T] {
        try validate(expectedCount: count, stride: stride)
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [T] in
            guard let base = raw.baseAddress else { return [] }
            let typed = base.assumingMemoryBound(to: T.self)
            return Array(UnsafeBufferPointer(start: typed, count: count))
        }
    }
}
