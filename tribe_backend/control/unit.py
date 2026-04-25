"""ControlUnit — the orchestrator at the top of the backend.

Uses Protocol-typed dependency injection so this compartment does not import
parcellation/, mesh/, or inference/ source. Real implementations are wired in
at the composition root (a follow-up commit after sibling phases merge).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from tribe_backend.contracts import (
    LiveFeedPoller,
    StimulusWindow,
    TribeInference,
    TribeOutput,
)
from tribe_backend.control.dispatcher import ArtifactSink, Bundle

logger = logging.getLogger(__name__)


@runtime_checkable
class ParcellationLike(Protocol):
    """Structural type for the parcellation dependency.

    Phase 1's `GlasserParcellationUnit` will satisfy this. Tests use a fake.
    """

    def generate_report(self, output: TribeOutput) -> Any: ...


@runtime_checkable
class MeshLike(Protocol):
    """Structural type for the mesh-export dependency.

    Phase 2's `BrainMeshExporter` will satisfy this.
    """

    def export_binary(self, output: TribeOutput, out_dir: Path) -> Path: ...


def _make_window_id(t_start: float | None) -> str:
    """Monotonic-ish window id; uses ISO timestamp + short uuid suffix to avoid
    collision when two windows arrive within the same second."""
    ts = time.time() if t_start is None else t_start
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    return f"{iso}-{uuid.uuid4().hex[:6]}"


class ControlUnit:
    """Pulls 30s windows off a poller, runs inference, fans out to parcellation
    and mesh export. Failure-isolated, backpressure-aware.

    Both `run_forever()` (live) and `process_window(...)` (one-shot) funnel
    through `_process(window) -> Bundle`. The single-window API returns the
    Report directly; the streaming API yields full Bundles.
    """

    def __init__(
        self,
        poller: LiveFeedPoller,
        inference: TribeInference,
        parcellation: ParcellationLike,
        mesh: MeshLike,
        sink: ArtifactSink,
        *,
        backpressure_max_lag: int | None = None,
    ) -> None:
        """
        backpressure_max_lag: maximum number of windows allowed to sit in
        the poller buffer. When the count exceeds this, the oldest are
        dropped (with a warning log) and only the most recent
        `backpressure_max_lag` are kept. `None` (default) disables
        backpressure entirely — every window is processed in order, which is
        the correct semantics for an offline replay or a well-paced source.
        """
        self._poller = poller
        self._inference = inference
        self._parcellation = parcellation
        self._mesh = mesh
        self._sink = sink
        self._backpressure_max_lag = backpressure_max_lag
        self._stop_event = threading.Event()
        self._dropped_count = 0

    # ------------------------------------------------------------------ public
    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def stop(self) -> None:
        """Signal the run loop to exit after the current window finishes.

        Also closes the poller if it has a close() method.
        """
        self._stop_event.set()
        close = getattr(self._poller, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover — close should be benign
                logger.exception("poller close() raised")

    def run_once(self) -> Bundle:
        """Pull exactly one window off the poller and process it."""
        window = next(iter(self._poller))
        return self._process(window)

    def run_forever(self) -> None:
        """Blocking loop. Drains the poller until StopIteration or stop()."""
        for _ in self.run_stream(self._poller):
            if self._stop_event.is_set():
                break

    def run_stream(self, poller: LiveFeedPoller) -> Iterator[Bundle]:
        """Iterate over the poller, applying backpressure, yielding Bundles.

        Backpressure semantics: the loop pulls one window and processes it.
        Once processing finishes, if the poller has more than
        `backpressure_max_lag` windows already buffered (inferred from
        `pending()`), we drop the older ones and only process the freshest.
        Pollers without a `pending()` method receive no draining — every
        window is processed in order.

        This ensures that when the poller keeps up (pending stays low),
        we never drop, and when inference falls behind, we drop oldest.
        """
        it = iter(poller)
        in_flight = False
        while True:
            if self._stop_event.is_set():
                return

            if in_flight:
                # We just processed a window. Before pulling the next one,
                # drain anything that piled up while inference was running.
                self._drain_pending(it)

            try:
                window = next(it)
            except StopIteration:
                return
            in_flight = True
            yield self._process(window)

    def process_window(
        self,
        *,
        video: Any,
        audio: Any,
        audio_sr: int,
        text: str,
        video_fps: float = 30.0,
        t_start: float | None = None,
    ) -> Any:
        """One-shot user API: hand in raw media, get a parcellation Report."""
        # Compute duration from audio length (most reliable signal).
        n_samples = audio.shape[0]
        duration_s = float(n_samples) / float(audio_sr)
        window = StimulusWindow(
            window_id=_make_window_id(t_start),
            t_start=time.time() if t_start is None else t_start,
            duration_s=duration_s,
            video=video,
            video_fps=video_fps,
            audio=audio,
            audio_sr=audio_sr,
            text=text,
        )
        return self._process(window).report

    # ----------------------------------------------------------------- private
    def _drain_pending(self, it: Iterator[StimulusWindow]) -> None:
        """If more than `backpressure_max_lag` windows are pending, drop all
        but the most recent. Non-blocking — relies on `pending()` to know
        how many windows are buffered.
        """
        if self._backpressure_max_lag is None:
            return
        pending_fn = getattr(it, "pending", None)
        if not callable(pending_fn):
            return
        try:
            n = int(pending_fn())
        except Exception:  # pragma: no cover
            return
        if n <= self._backpressure_max_lag:
            return
        to_drop = n - self._backpressure_max_lag
        dropped = 0
        for _ in range(to_drop):
            try:
                next(it)
            except StopIteration:
                break
            dropped += 1
        if dropped:
            self._dropped_count += dropped
            logger.warning(
                "backpressure: dropped %d stale window(s)", dropped
            )

    def _process(self, window: StimulusWindow) -> Bundle:
        """Inference -> parcellation + mesh fan-out, with failure isolation."""
        out = self._inference(window)
        # Defensive: stamp window_id if inference forgot to.
        if out.window_id != window.window_id:
            out = out.with_window_id(window.window_id)

        report: Any = None
        try:
            report = self._parcellation.generate_report(out)
        except Exception:
            logger.exception(
                "parcellation.generate_report failed for window_id=%s",
                window.window_id,
            )
            report = None

        mesh_dir: Path | None = None
        try:
            target = self._sink.path_for(window)
            mesh_dir = self._mesh.export_binary(out, target)
        except Exception:
            logger.exception(
                "mesh.export_binary failed for window_id=%s",
                window.window_id,
            )
            mesh_dir = None

        return Bundle(window_id=window.window_id, report=report, mesh_dir=mesh_dir)
