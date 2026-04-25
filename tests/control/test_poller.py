"""Tests for LiveFeedPoller adapters: NoopPoller, MockLivePoller, FilePoller."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from tribe_backend.contracts import LiveFeedPoller, StimulusWindow
from tribe_backend.control import FilePoller, MockLivePoller, NoopPoller
from tests.conftest import make_stimulus_window


pytestmark = pytest.mark.unit


# ----------------------------------------------------- protocol conformance

def test_noop_poller_satisfies_protocol() -> None:
    p = NoopPoller()
    assert isinstance(p, LiveFeedPoller)
    with pytest.raises(StopIteration):
        next(iter(p))


def test_noop_poller_pending_is_zero() -> None:
    assert NoopPoller().pending() == 0


def test_mock_live_poller_satisfies_protocol() -> None:
    p = MockLivePoller([make_stimulus_window(window_id="x")])
    assert isinstance(p, LiveFeedPoller)


def test_mock_live_poller_yields_then_stops() -> None:
    windows = [make_stimulus_window(window_id=f"w-{i}") for i in range(3)]
    p = MockLivePoller(windows)
    out = list(p)
    assert [w.window_id for w in out] == ["w-0", "w-1", "w-2"]


def test_mock_live_poller_push_after_construction() -> None:
    p = MockLivePoller()
    p.push(make_stimulus_window(window_id="late"))
    assert next(p).window_id == "late"
    with pytest.raises(StopIteration):
        next(p)


def test_mock_live_poller_pending_reflects_queue() -> None:
    p = MockLivePoller([make_stimulus_window(window_id="a"), make_stimulus_window(window_id="b")])
    assert p.pending() == 2
    next(p)
    assert p.pending() == 1


def test_mock_live_poller_close_stops_blocking_iteration() -> None:
    p = MockLivePoller(block_when_empty=True, block_timeout_s=0.05)
    results: list[StimulusWindow | str] = []

    def waiter() -> None:
        try:
            results.append(next(p))
        except StopIteration:
            results.append("stopped")

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.05)
    p.close()
    th.join(timeout=1.0)
    assert not th.is_alive()
    assert results == ["stopped"]


# ------------------------------------------------------------ FilePoller stub

@pytest.mark.integration
def test_file_poller_initializes_when_dir_exists(tmp_path: Path) -> None:
    (tmp_path / "clip1.mp4").write_bytes(b"")
    (tmp_path / "clip2.mp4").write_bytes(b"")
    p = FilePoller(tmp_path)
    assert p.pending() == 2


def test_file_poller_raises_when_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FilePoller(tmp_path / "does-not-exist")


@pytest.mark.integration
def test_file_poller_iter_is_stub_not_implemented(tmp_path: Path) -> None:
    (tmp_path / "clip1.mp4").write_bytes(b"")
    p = FilePoller(tmp_path)
    with pytest.raises(NotImplementedError):
        next(iter(p))


# ----------------------------------- camera/RTSP marker (skipped in unit runs)

@pytest.mark.integration
@pytest.mark.skip(reason="RTSP/camera poller not implemented in this compartment")
def test_camera_poller_placeholder() -> None:  # pragma: no cover
    pass
