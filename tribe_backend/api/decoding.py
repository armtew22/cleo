"""mp4 bytes -> (video ndarray, audio ndarray, audio_sr).

Video frames are decoded with OpenCV (`cv2.VideoCapture`), which is already
used by the Phase 5 GPU smoke test. Audio is extracted by piping the bytes
through `ffmpeg` to raw PCM s16le on stdout. ffmpeg is a system dependency on
the deploy box; it is not vendored.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np


class DecodeError(Exception):
    """Raised when an uploaded blob cannot be decoded as mp4."""


def _read_video_frames(path: Path) -> tuple[np.ndarray, float]:
    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError("could not open media as a video stream")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise DecodeError("video stream contained zero frames")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False), fps


def _extract_audio(path: Path) -> tuple[np.ndarray, int]:
    """Use ffmpeg to extract a 1-D float32 mono waveform at the file's native sr.

    We probe the source sr with ffprobe, then ask ffmpeg to resample to that
    same rate (effectively a passthrough) so the output sample rate is known.
    On files with no audio track, returns a zero-length array and sr=16000.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise DecodeError("ffmpeg/ffprobe are required to decode audio")

    # Probe source sample rate (returns empty string if no audio track).
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    sr_str = probe.stdout.strip()
    if not sr_str:
        return np.zeros(0, dtype=np.float32), 16000
    try:
        src_sr = int(sr_str)
    except ValueError:
        src_sr = 16000

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-vn",                      # discard video
            "-ac", "1",                 # mono
            "-ar", str(src_sr),         # native sample rate
            "-f", "s16le",              # signed 16-bit little-endian PCM
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise DecodeError(
            f"ffmpeg audio decode failed: {proc.stderr.decode('utf-8', errors='replace')[:200]}"
        )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    audio = (pcm.astype(np.float32) / 32768.0)
    return audio, src_sr


def decode_mp4(data: bytes) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decode mp4 bytes to (video (F,H,W,3) uint8, audio (S,) float32, sr).

    Raises
    ------
    DecodeError
        If the bytes cannot be opened as a video stream or audio extraction
        fails. Callers should map this to HTTP 400.
    """
    if not data:
        raise DecodeError("empty media payload")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as fh:
        fh.write(data)
        fh.flush()
        path = Path(fh.name)
        video, _fps = _read_video_frames(path)
        audio, sr = _extract_audio(path)
    return video, audio, sr


def probe_video_fps(data: bytes) -> float | None:
    """Return the source video fps, or None if it can't be determined."""
    if not data:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as fh:
        fh.write(data)
        fh.flush()
        try:
            _, fps = _read_video_frames(Path(fh.name))
        except DecodeError:
            return None
    return fps if fps > 0 else None
