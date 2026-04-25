"""Phase 4 — pure-CPU per-modality preprocessing.

These functions match the input ports of the upstream tribev2 engine
(see plan §5 Phase 5 port-matching table). No torch import — numpy + scipy.
"""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.inference.preprocessing import _encode_video


pytestmark = pytest.mark.unit


# ---- video ------------------------------------------------------------------


def test_encode_video_resamples_to_2hz_over_30s() -> None:
    # 30s @ 4 fps == 120 frames; 2 Hz target == 60 frames.
    video = np.zeros((120, 64, 64, 3), dtype=np.uint8)
    out = _encode_video(video, fps=4.0)
    assert out.shape[0] == 60


def test_encode_video_output_is_float32_channel_first() -> None:
    video = np.zeros((120, 64, 64, 3), dtype=np.uint8)
    out = _encode_video(video, fps=4.0)
    assert out.dtype == np.float32
    # (T_2hz, C=3, H, W) — channel axis is 1
    assert out.ndim == 4
    assert out.shape[1] == 3
    assert out.shape[2] == 64 and out.shape[3] == 64


def test_encode_video_normalizes_to_unit_range() -> None:
    video = np.full((60, 8, 8, 3), 255, dtype=np.uint8)
    out = _encode_video(video, fps=2.0)
    assert out.max() <= 1.0 + 1e-6
    assert out.min() >= 0.0 - 1e-6
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_encode_video_zero_input_maps_to_zero() -> None:
    video = np.zeros((60, 8, 8, 3), dtype=np.uint8)
    out = _encode_video(video, fps=2.0)
    np.testing.assert_allclose(out, 0.0)


def test_encode_video_rejects_non_uint8() -> None:
    bad = np.zeros((60, 8, 8, 3), dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        _encode_video(bad, fps=2.0)


def test_encode_video_rejects_wrong_ndim() -> None:
    with pytest.raises((TypeError, ValueError)):
        _encode_video(np.zeros((60, 8, 8), dtype=np.uint8), fps=2.0)


def test_encode_video_rejects_non_rgb_channels() -> None:
    with pytest.raises((TypeError, ValueError)):
        _encode_video(np.zeros((60, 8, 8, 4), dtype=np.uint8), fps=2.0)


def test_encode_video_higher_fps_downsamples() -> None:
    # 30 fps over 30s -> 900 frames -> 60 @ 2 Hz
    video = np.zeros((900, 16, 16, 3), dtype=np.uint8)
    out = _encode_video(video, fps=30.0)
    assert out.shape[0] == 60


def test_encode_video_lower_fps_handled() -> None:
    # 1 fps over 30s -> 30 frames -> upsample to 60 @ 2 Hz
    video = np.zeros((30, 8, 8, 3), dtype=np.uint8)
    out = _encode_video(video, fps=1.0)
    assert out.shape[0] == 60
