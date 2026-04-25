// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - Polling AsyncSequence helpers

extension BrainMeshClient {
    /// Poll the server for color updates, emitting one ColorsResponse per interval.
    /// Latest-wins: an in-flight request is cancelled if a newer image arrives.
    public func poll(
        images: AsyncStream<Data>,
        every interval: Duration
    ) -> AsyncThrowingStream<ColorsResponse, Error> {
        fatalError("unimplemented")
    }

    /// Poll returning full inference envelopes (colors + report).
    public func pollFull(
        images: AsyncStream<Data>,
        every interval: Duration
    ) -> AsyncThrowingStream<InferenceEnvelope, Error> {
        fatalError("unimplemented")
    }
}
