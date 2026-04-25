"""LiveFeedPoller adapters.

* MockLivePoller — programmable, used by tests.
* NoopPoller     — empty iterator, for one-shot users of process_window().
* FilePoller     — stub: enumerates pre-recorded clips on disk into windows.
                   Marked integration; full implementation lands when media
                   loaders are available.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

from tribe_backend.contracts import StimulusWindow


class NoopPoller:
    """Iterator that immediately raises StopIteration. For one-shot users."""

    def __iter__(self) -> "NoopPoller":
        return self

    def __next__(self) -> StimulusWindow:
        raise StopIteration

    def close(self) -> None:
        return None

    def pending(self) -> int:
        return 0


class MockLivePoller:
    """Programmable, thread-safe poller for tests and offline replay.

    Construct with a list of windows; or push them in over time. The poller is
    a valid `LiveFeedPoller` (iter + next). It also exposes `pending()` so
    ControlUnit's backpressure logic can drain stale windows non-blockingly.
    """

    def __init__(
        self,
        windows: list[StimulusWindow] | None = None,
        *,
        interval_s: float = 0.0,
        block_when_empty: bool = False,
        block_timeout_s: float = 0.5,
    ) -> None:
        self._queue: deque[StimulusWindow] = deque(windows or [])
        self._interval_s = interval_s
        self._block_when_empty = block_when_empty
        self._block_timeout_s = block_timeout_s
        self._closed = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def push(self, window: StimulusWindow) -> None:
        with self._cond:
            self._queue.append(window)
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

    def __iter__(self) -> "MockLivePoller":
        return self

    def __next__(self) -> StimulusWindow:
        if self._interval_s:
            time.sleep(self._interval_s)
        with self._cond:
            if not self._queue and self._block_when_empty and not self._closed:
                self._cond.wait(timeout=self._block_timeout_s)
            if self._queue:
                return self._queue.popleft()
            raise StopIteration


class FilePoller:
    """Reads a directory of pre-recorded media clips and yields StimulusWindows.

    NOTE: this is a stub. The full implementation will use ffmpeg/decord. It is
    marked `integration` in tests because it touches the filesystem and an
    optional media-loading dependency.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(f"FilePoller root does not exist: {self._root}")
        self._files = sorted(p for p in self._root.iterdir() if p.is_file())
        self._idx = 0

    def __iter__(self) -> "FilePoller":
        return self

    def __next__(self) -> StimulusWindow:
        if self._idx >= len(self._files):
            raise StopIteration
        # Stub: full media loading is beyond this compartment. Tests that need
        # real loading should mark themselves @pytest.mark.integration and
        # provide their own loader.
        raise NotImplementedError(
            "FilePoller is a stub; supply a real media loader before use."
        )

    def pending(self) -> int:
        return max(0, len(self._files) - self._idx)

    def close(self) -> None:
        self._idx = len(self._files)


# Camera/RTSP would land here. Real implementation requires a streaming media
# stack and is integration-only — it is intentionally not provided in the unit
# build.
