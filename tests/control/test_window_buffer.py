"""Tests for WindowBuffer — rolls 30s windows from variable-rate inputs and
emits StimulusWindows with `25 ≤ duration_s ≤ 35` and modality alignment."""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.control import WindowBuffer


pytestmark = pytest.mark.unit


def _push_audio(buf: WindowBuffer, t0: float, duration_s: float, sr: int = 16000) -> None:
    n = int(round(duration_s * sr))
    samples = np.zeros(n, dtype=np.float32)
    buf.push_audio(samples, sr=sr, t_start=t0)


def _push_video(buf: WindowBuffer, t0: float, duration_s: float, fps: float = 4.0) -> None:
    n = max(1, int(round(duration_s * fps)))
    frames = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    buf.push_video(frames, fps=fps, t_start=t0)


# ------------------------------------------------------------- core behaviors

def test_buffer_not_ready_when_empty() -> None:
    buf = WindowBuffer()
    assert not buf.ready()


def test_buffer_emits_30s_window_after_30s_of_audio() -> None:
    buf = WindowBuffer()
    _push_audio(buf, 0.0, 30.0)
    _push_video(buf, 0.0, 30.0)
    assert buf.ready()
    w = buf.emit()
    assert 25.0 <= w.duration_s <= 35.0
    assert w.audio_sr == 16000
    assert w.video_fps == 4.0


def test_buffer_alignment_audio_video_text_within_window() -> None:
    buf = WindowBuffer()
    _push_audio(buf, 0.0, 30.0)
    _push_video(buf, 0.0, 30.0)
    buf.push_text("first caption", t=2.0)
    buf.push_text("second caption", t=15.0)
    buf.push_text("late caption (next window)", t=45.0)
    w = buf.emit()
    # The text picked up is the most recent ONE within [t0, t1)=[0, 30).
    assert w.text == "second caption"


def test_buffer_emits_with_variable_rate_inputs() -> None:
    """Audio arrives in many small chunks; video in a few large ones — still aligned."""
    buf = WindowBuffer()
    # 30s of audio in 1s chunks
    for i in range(30):
        _push_audio(buf, float(i), 1.0)
    # 30s of video in 5s chunks
    for i in range(6):
        _push_video(buf, float(i * 5), 5.0)
    assert buf.ready()
    w = buf.emit()
    assert 25.0 <= w.duration_s <= 35.0
    # Audio length matches duration ~30s @ sr=16000
    assert abs(w.audio.shape[0] - 30 * 16000) < 16000  # within 1s tolerance
    # Video length matches duration ~30s @ fps=4
    assert abs(w.video.shape[0] - 30 * 4) <= 4


def test_buffer_iterates_multiple_windows() -> None:
    buf = WindowBuffer()
    _push_audio(buf, 0.0, 75.0)
    _push_video(buf, 0.0, 75.0)
    windows = list(buf)
    # 75s of audio at stride 30 -> two complete 30s windows.
    assert len(windows) == 2
    assert windows[0].t_start == 0.0
    assert windows[1].t_start == 30.0


def test_buffer_garbage_collects_old_chunks_after_emit() -> None:
    buf = WindowBuffer()
    _push_audio(buf, 0.0, 60.0)
    _push_video(buf, 0.0, 60.0)
    buf.emit()
    # After emitting one 30s window, audio chunks fully before t=30 are gone.
    # We had a single 60s chunk starting at 0 — it overlaps the second window
    # at [30, 60), so it should NOT be GC'd; just the head moves up.
    assert buf.ready()  # second window still emittable


def test_buffer_rejects_invalid_duration_targets() -> None:
    with pytest.raises(ValueError):
        WindowBuffer(target_duration_s=20.0, min_duration_s=25.0)
    with pytest.raises(ValueError):
        WindowBuffer(target_duration_s=40.0, max_duration_s=35.0)


def test_buffer_emit_before_ready_raises() -> None:
    buf = WindowBuffer()
    with pytest.raises(RuntimeError):
        buf.emit()


def test_buffer_audio_sr_mismatch_raises() -> None:
    buf = WindowBuffer()
    # Two overlapping audio chunks within the same 30s window but different sr.
    _push_audio(buf, 0.0, 20.0, sr=16000)
    _push_audio(buf, 20.0, 15.0, sr=22050)
    _push_video(buf, 0.0, 35.0)
    assert buf.ready()
    with pytest.raises(ValueError):
        buf.emit()


def test_buffer_rejects_non_4d_video() -> None:
    buf = WindowBuffer()
    bad = np.zeros((10, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        buf.push_video(bad, fps=4.0, t_start=0.0)


def test_buffer_rejects_non_1d_or_2d_audio() -> None:
    buf = WindowBuffer()
    bad = np.zeros((10, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        buf.push_audio(bad, sr=16000, t_start=0.0)


@pytest.mark.parametrize("target", [27.0, 30.0, 33.0])
def test_buffer_emitted_duration_within_25_35(target: float) -> None:
    buf = WindowBuffer(target_duration_s=target, stride_s=target)
    _push_audio(buf, 0.0, target + 1.0)
    _push_video(buf, 0.0, target + 1.0)
    w = buf.emit()
    assert 25.0 <= w.duration_s <= 35.0
