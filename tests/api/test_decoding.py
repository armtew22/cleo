"""mp4 -> (video ndarray, audio ndarray, audio_sr) decoder."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tribe_backend.api.decoding import DecodeError, decode_mp4

REAL_30S_CLIP = Path("/home/md2292/tribev2-bench/stimuli/clip_30s.mp4")


@pytest.mark.skipif(not REAL_30S_CLIP.is_file(), reason="real clip fixture not present")
def test_decode_real_clip():
    data = REAL_30S_CLIP.read_bytes()
    video, audio, sr = decode_mp4(data)
    assert isinstance(video, np.ndarray)
    assert video.ndim == 4 and video.shape[-1] == 3 and video.dtype == np.uint8
    assert video.shape[0] > 0 and video.shape[1] > 0 and video.shape[2] > 0
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert isinstance(sr, int) and sr > 0
    assert audio.shape[0] > 0


def test_decode_garbage_raises():
    with pytest.raises(DecodeError):
        decode_mp4(b"not an mp4 at all")


def test_decode_returns_video_fps_when_requested():
    """A small helper: decoder also exposes the source video fps."""
    if not REAL_30S_CLIP.is_file():
        pytest.skip("real clip fixture not present")
    from tribe_backend.api.decoding import probe_video_fps
    fps = probe_video_fps(REAL_30S_CLIP.read_bytes())
    assert fps is not None and fps > 0
