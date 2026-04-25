// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import Foundation

// MARK: - Polling AsyncSequence helpers
//
// Backpressure model: latest-wins. While a request is in flight, any newly-
// arriving image cancels the in-flight Task and starts a new request from the
// freshest image. Older images are dropped — the brain only ever shows the
// freshest data the server has been asked about.

extension BrainMeshClient {
    /// Returns an `AsyncThrowingStream` that posts each image from `images` and yields ``ColorsResponse`` values.
    public nonisolated func poll(
        images: AsyncStream<Data>,
        every interval: Duration,
        method: AggregationMethod = .windowMean
    ) -> AsyncThrowingStream<ColorsResponse, Error> {
        AsyncThrowingStream { continuation in
            let pumpTask = Task {
                var inFlight: Task<Void, Never>?
                var lastDispatch: ContinuousClock.Instant?
                let clock = ContinuousClock()

                for await image in images {
                    // Throttle: skip if a previous dispatch is younger than `interval`.
                    if let last = lastDispatch,
                       clock.now - last < interval {
                        continue
                    }
                    inFlight?.cancel()
                    lastDispatch = clock.now

                    let task = Task { [image] in
                        do {
                            let r = try await self.fetchColors(
                                image: image,
                                method: method
                            )
                            if Task.isCancelled { return }
                            continuation.yield(r)
                        } catch is CancellationError {
                            return
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    inFlight = task
                }
                inFlight?.cancel()
                continuation.finish()
            }
            continuation.onTermination = { _ in
                pumpTask.cancel()
            }
        }
    }

    /// Returns an `AsyncThrowingStream` that posts each image from `images` and yields ``InferenceEnvelope`` values.
    public nonisolated func pollFull(
        images: AsyncStream<Data>,
        every interval: Duration,
        method: AggregationMethod = .windowMean
    ) -> AsyncThrowingStream<InferenceEnvelope, Error> {
        AsyncThrowingStream { continuation in
            let pumpTask = Task {
                var inFlight: Task<Void, Never>?
                var lastDispatch: ContinuousClock.Instant?
                let clock = ContinuousClock()

                for await image in images {
                    if let last = lastDispatch,
                       clock.now - last < interval {
                        continue
                    }
                    inFlight?.cancel()
                    lastDispatch = clock.now

                    let task = Task { [image] in
                        do {
                            let env = try await self.fetchFull(
                                image: image,
                                method: method
                            )
                            if Task.isCancelled { return }
                            continuation.yield(env)
                        } catch is CancellationError {
                            return
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    inFlight = task
                }
                inFlight?.cancel()
                continuation.finish()
            }
            continuation.onTermination = { _ in
                pumpTask.cancel()
            }
        }
    }
}
