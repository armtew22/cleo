"""Per-modality CPU preprocessing for the TRIBE v2 inference engine.

These encoders translate from :class:`StimulusWindow` shapes (whatever the
poller produces) into the exact tensor shapes the upstream tribev2 model
expects at its input ports. They are pure CPU — **no torch import** — so the
control loop can run them off the GPU thread.

Port-matching contract (see plan §5 Phase 5):

    video : (F, H, W, 3) uint8 @ video_fps  ->  (T_2hz, 3, H, W) float32 in [0, 1]
            via nearest-neighbor temporal resample to 2 Hz, channel reorder
            (HWC -> CHW), and /255 normalization. T_2hz = round(F / video_fps * 2).
    audio : (S,) or (S, C) float32 @ sr     ->  (S_resampled,) float32 @ 16000 Hz
            stereo collapsed to mono via channel mean; resampled with
            scipy.signal.resample_poly.
    text  : str                              ->  str (placeholder)
            real Llama-3.2-3B tokenization arrives in Phase 5.

These functions intentionally accept and return numpy arrays so they can be
unit-tested without touching CUDA or the model.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

__all__ = ["_encode_video", "_encode_audio", "_encode_text"]

# Target sample rate / frame rate for the upstream encoders.
_TARGET_VIDEO_HZ: float = 2.0       # V-JEPA-2 expects ~2 Hz video tokens
_TARGET_AUDIO_SR: int = 16000       # Wav2Vec-Bert-2.0 expects 16 kHz mono


def _encode_video(video: np.ndarray, fps: float) -> np.ndarray:
    """Resample video to 2 Hz, normalize to [0, 1], reorder to channel-first.

    Parameters
    ----------
    video : np.ndarray of shape (F, H, W, 3), dtype uint8
        Raw frames at ``fps``.
    fps : float
        Source frame rate.

    Returns
    -------
    np.ndarray of shape (T_2hz, 3, H, W), dtype float32, values in [0, 1]
        Where ``T_2hz = max(1, round(F / fps * 2))``. Frames are picked by
        nearest-neighbor along the time axis (good enough for matching the
        2 Hz V-JEPA-2 input rate; the engine does its own internal resample).
    """
    if not isinstance(video, np.ndarray):
        raise TypeError(f"video must be ndarray, got {type(video).__name__}")
    if video.dtype != np.uint8:
        raise TypeError(f"video must be uint8, got dtype={video.dtype}")
    if video.ndim != 4:
        raise ValueError(f"video must be 4D (F,H,W,3), got ndim={video.ndim}")
    if video.shape[-1] != 3:
        raise ValueError(f"video must have 3 channels (RGB), got shape={video.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    F, H, W, _ = video.shape
    duration_s = F / fps
    T_target = max(1, int(round(duration_s * _TARGET_VIDEO_HZ)))

    # Nearest-neighbor temporal resample. linspace places samples at the
    # centers of T_target buckets across [0, F-1].
    if T_target == 1:
        idx = np.array([F // 2], dtype=np.int64)
    else:
        idx = np.linspace(0, F - 1, num=T_target).round().astype(np.int64)
    sampled = video[idx]                              # (T, H, W, 3) uint8

    # Reorder HWC -> CHW and normalize.
    chw = np.transpose(sampled, (0, 3, 1, 2))         # (T, 3, H, W) uint8
    return (chw.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def _encode_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16 kHz mono float32.

    Parameters
    ----------
    audio : np.ndarray
        Either ``(S,)`` mono or ``(S, C)`` multichannel. Float32 preferred but
        any numeric dtype is cast on entry.
    sr : int
        Source sample rate.

    Returns
    -------
    np.ndarray of shape (S_resampled,), dtype float32
        Where ``S_resampled ~= S * 16000 / sr``. Polyphase resampling reduces
        ringing relative to FFT resampling on real waveforms.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be ndarray, got {type(audio).__name__}")
    if audio.ndim not in (1, 2):
        raise ValueError(f"audio must be 1D or 2D, got ndim={audio.ndim}")
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise ValueError(f"sr must be a positive int, got {sr!r}")

    # Cast to float32, collapse multichannel to mono.
    x = audio.astype(np.float32, copy=False)
    if x.ndim == 2:
        x = x.mean(axis=1, dtype=np.float32)

    sr = int(sr)
    if sr == _TARGET_AUDIO_SR:
        return np.ascontiguousarray(x, dtype=np.float32)

    # Polyphase resample with up=target, down=src reduced by gcd.
    from math import gcd
    g = gcd(_TARGET_AUDIO_SR, sr)
    up = _TARGET_AUDIO_SR // g
    down = sr // g
    y = resample_poly(x, up, down).astype(np.float32, copy=False)
    return y


def _encode_text(text: str) -> str:
    """Placeholder — real Llama-3.2-3B tokenization arrives in Phase 5.

    Returning the raw string keeps the call-site contract (a function that
    takes a stimulus caption and returns a tokenized representation) without
    pulling in a 3 GB tokenizer dependency for Phase 4 unit tests.
    """
    raise NotImplementedError  # implemented in the text commit
