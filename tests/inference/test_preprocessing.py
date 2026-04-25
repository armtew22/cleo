"""Phase 4 — pure-CPU per-modality preprocessing.

These functions match the input ports of the upstream tribev2 engine
(see plan §5 Phase 5 port-matching table). No torch import — numpy + scipy.
"""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.inference.preprocessing import (
    _encode_audio,
    _encode_text,
    _encode_video,
)


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


# ---- audio ------------------------------------------------------------------


def test_encode_audio_resamples_to_16khz() -> None:
    sr = 44100
    n = sr * 30
    audio = np.zeros(n, dtype=np.float32)
    out = _encode_audio(audio, sr=sr)
    # Target sample rate is 16000; tolerate tiny rounding from polyphase.
    assert abs(out.shape[0] - 16000 * 30) <= 4


def test_encode_audio_output_is_mono_float32() -> None:
    audio = np.zeros(44100, dtype=np.float32)
    out = _encode_audio(audio, sr=44100)
    assert out.dtype == np.float32
    assert out.ndim == 1


def test_encode_audio_passthrough_when_already_16k_mono() -> None:
    audio = np.linspace(-0.5, 0.5, 16000 * 5, dtype=np.float32)
    out = _encode_audio(audio, sr=16000)
    assert out.shape[0] == 16000 * 5
    np.testing.assert_allclose(out, audio, atol=1e-6)


def test_encode_audio_mixes_stereo_to_mono() -> None:
    n = 16000 * 2
    left = np.full(n, 0.25, dtype=np.float32)
    right = np.full(n, 0.75, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # (S, 2)
    out = _encode_audio(stereo, sr=16000)
    assert out.ndim == 1
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_encode_audio_preserves_signal_energy_roughly() -> None:
    sr = 44100
    t = np.arange(sr * 2) / sr
    sig = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    out = _encode_audio(sig, sr=sr)
    rms_in = float(np.sqrt(np.mean(sig**2)))
    rms_out = float(np.sqrt(np.mean(out**2)))
    assert abs(rms_in - rms_out) < 0.05


def test_encode_audio_rejects_3d_array() -> None:
    with pytest.raises((TypeError, ValueError)):
        _encode_audio(np.zeros((10, 2, 2), dtype=np.float32), sr=16000)


def test_encode_audio_rejects_non_positive_sr() -> None:
    with pytest.raises(ValueError):
        _encode_audio(np.zeros(100, dtype=np.float32), sr=0)


def test_encode_audio_30s_at_44100_yields_30s_at_16000() -> None:
    sr = 44100
    audio = np.zeros(sr * 30, dtype=np.float32).astype(np.float32)
    out = _encode_audio(audio, sr=sr)
    assert abs(out.shape[0] - 16000 * 30) <= 4


# ---- text -------------------------------------------------------------------


def test_encode_text_returns_string_placeholder() -> None:
    out = _encode_text("a busy intersection")
    # Phase 4 placeholder: real Llama-3.2-3B tokenizer arrives in Phase 5.
    assert out == "a busy intersection"


def test_encode_text_handles_empty_string() -> None:
    assert _encode_text("") == ""


def test_encode_text_is_callable() -> None:
    assert callable(_encode_text)


@pytest.mark.skip(reason="Phase 5 — needs Llama-3.2-3B tokenizer for shape check")
def test_encode_text_produces_2hz_token_tensor_shape() -> None:
    # Will assert (T_2hz, D_text=2048) once the real tokenizer lands.
    raise NotImplementedError
