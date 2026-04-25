// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - BinaryBufferReader (internal)

/// Wraps a Data buffer and provides stride-validated typed reads.
struct BinaryBufferReader {
    let data: Data

    init(data: Data) {
        self.data = data
    }

    /// Validate that the buffer contains exactly `count` elements of `stride` bytes each.
    func validate(expectedCount: Int, stride: Int) throws {
        let expected = expectedCount * stride
        guard data.count == expected else {
            throw BrainMeshError.sizeMismatch(expected: expected, actual: data.count)
        }
    }

    /// Read all elements as an array of `T` without copying.
    func elements<T>(as type: T.Type, stride: Int, count: Int) throws -> [T] {
        fatalError("unimplemented")
    }
}
