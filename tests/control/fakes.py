"""Fakes used across control/ unit tests.

Lives under tests/ so it cannot leak into production code paths. Provides:
  * FakeTribeInference   — protocol-conformant stub stamping window_id.
  * FakeParcellation     — minimal generate_report() returning a Report-like.
  * FakeMesh             — minimal export_binary(out, out_dir) -> Path.
  * MockPoller           — programmable LiveFeedPoller yielding queued windows.
  * SlowInference        — sleeps for N seconds before returning, for backpressure tests.
  * RaisingParcellation / RaisingMesh — for failure-isolation tests.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

from tribe_backend.contracts import StimulusWindow, TribeOutput


@dataclass
class FakeReport:
    window_id: str | None
    text: str = "fake report"
    top_regions: list[str] = field(default_factory=list)


class FakeTribeInference:
    """Stamps the input window_id onto a fixture TribeOutput."""

    def __init__(self, fixture: TribeOutput) -> None:
        self._fixture = fixture
        self.calls: list[StimulusWindow] = []

    def __call__(self, window: StimulusWindow) -> TribeOutput:
        self.calls.append(window)
        return replace(self._fixture, window_id=window.window_id)


class SlowInference:
    """Inference that sleeps for `delay_s` before stamping window_id and returning."""

    def __init__(self, fixture: TribeOutput, delay_s: float) -> None:
        self._fixture = fixture
        self._delay_s = delay_s
        self.calls: list[StimulusWindow] = []

    def __call__(self, window: StimulusWindow) -> TribeOutput:
        time.sleep(self._delay_s)
        self.calls.append(window)
        return replace(self._fixture, window_id=window.window_id)


class FakeParcellation:
    """Returns a FakeReport echoing the output's window_id."""

    def __init__(self) -> None:
        self.calls: list[TribeOutput] = []

    def generate_report(self, output: TribeOutput) -> FakeReport:
        self.calls.append(output)
        return FakeReport(window_id=output.window_id, text=f"report::{output.window_id}")


class FakeMesh:
    """Writes a stub `mesh.bin` and returns out_dir."""

    def __init__(self) -> None:
        self.calls: list[tuple[TribeOutput, Path]] = []

    def export_binary(self, output: TribeOutput, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "brain_meta.json").write_text(f'{{"window_id":"{output.window_id}"}}')
        (out_dir / "brain_colors.bin").write_bytes(b"\x00" * 16)
        self.calls.append((output, out_dir))
        return out_dir


class RaisingParcellation:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("parcellation boom")

    def generate_report(self, output: TribeOutput) -> Any:
        raise self.exc


class RaisingMesh:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("mesh boom")

    def export_binary(self, output: TribeOutput, out_dir: Path) -> Path:
        raise self.exc


class MockPoller:
    """Programmable LiveFeedPoller. Iterates over a queue of windows.

    Supports:
      * pre-loaded windows passed at construction
      * `push(window)` to add more after construction (used by backpressure tests)
      * `interval_s` between yields (default 0)
      * `close()` for graceful shutdown semantics
    """

    def __init__(
        self,
        windows: list[StimulusWindow] | None = None,
        *,
        interval_s: float = 0.0,
    ) -> None:
        self._queue: deque[StimulusWindow] = deque(windows or [])
        self._interval_s = interval_s
        self._closed = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._block_when_empty = False

    def push(self, window: StimulusWindow) -> None:
        with self._cond:
            self._queue.append(window)
            self._cond.notify_all()

    def set_blocking(self, blocking: bool) -> None:
        with self._cond:
            self._block_when_empty = blocking
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> "MockPoller":
        return self

    def __next__(self) -> StimulusWindow:
        if self._interval_s:
            time.sleep(self._interval_s)
        with self._cond:
            while not self._queue and self._block_when_empty and not self._closed:
                self._cond.wait(timeout=0.1)
            if self._closed and not self._queue:
                raise StopIteration
            if not self._queue:
                raise StopIteration
            return self._queue.popleft()
