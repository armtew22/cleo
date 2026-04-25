"""WindowBuffer — rolls 30s multimodal windows from variable-rate inputs.

Each modality (video, audio, text) is appended with an absolute timestamp.
When the buffer holds at least `target_duration_s` worth of audio, it emits a
`StimulusWindow` and slides the head forward by `stride_s`.

Constraints:
  * emitted duration_s lies in [25, 35] (per StimulusWindow contract).
  * audio/video/text are temporally aligned within the window — text picks up
    the most recent caption that arrived during the window's time range.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from tribe_backend.contracts import StimulusWindow


@dataclass(frozen=True)
class _AudioChunk:
    t_start: float
    samples: np.ndarray  # (S,) or (S, C) float32
    sr: int


@dataclass(frozen=True)
class _VideoChunk:
    t_start: float
    frames: np.ndarray  # (F, H, W, 3) uint8
    fps: float


@dataclass(frozen=True)
class _TextChunk:
    t: float
    text: str


class WindowBuffer:
    """Rolling buffer that emits 30s windows aligned across modalities."""

    def __init__(
        self,
        *,
        target_duration_s: float = 30.0,
        stride_s: float = 30.0,
        min_duration_s: float = 25.0,
        max_duration_s: float = 35.0,
    ) -> None:
        if not (min_duration_s <= target_duration_s <= max_duration_s):
            raise ValueError(
                "target_duration_s must lie in [min_duration_s, max_duration_s]"
            )
        self._target = float(target_duration_s)
        self._stride = float(stride_s)
        self._min_dur = float(min_duration_s)
        self._max_dur = float(max_duration_s)
        self._audio: list[_AudioChunk] = []
        self._video: list[_VideoChunk] = []
        self._text: list[_TextChunk] = []
        self._head_t: float | None = None  # earliest t we still consider

    # -------------------------------------------------------------- ingest API
    def push_audio(self, samples: np.ndarray, sr: int, t_start: float | None = None) -> None:
        if samples.ndim not in (1, 2):
            raise ValueError("audio must be 1D or 2D")
        ts = time.time() if t_start is None else float(t_start)
        self._audio.append(_AudioChunk(ts, np.asarray(samples), int(sr)))
        if self._head_t is None:
            self._head_t = ts

    def push_video(self, frames: np.ndarray, fps: float, t_start: float | None = None) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("video must be (F, H, W, 3)")
        ts = time.time() if t_start is None else float(t_start)
        self._video.append(_VideoChunk(ts, np.asarray(frames), float(fps)))
        if self._head_t is None:
            self._head_t = ts

    def push_text(self, text: str, t: float | None = None) -> None:
        ts = time.time() if t is None else float(t)
        self._text.append(_TextChunk(ts, text))
        if self._head_t is None:
            self._head_t = ts

    # ------------------------------------------------------------ window emit
    def _audio_duration_from(self, t0: float) -> float:
        total_s = 0.0
        for chunk in self._audio:
            if chunk.t_start + len(chunk.samples) / chunk.sr <= t0:
                continue
            chunk_start = max(t0, chunk.t_start)
            chunk_end = chunk.t_start + chunk.samples.shape[0] / chunk.sr
            total_s += max(0.0, chunk_end - chunk_start)
        return total_s

    def ready(self) -> bool:
        if self._head_t is None:
            return False
        return self._audio_duration_from(self._head_t) >= self._target

    def emit(self) -> StimulusWindow:
        """Emit a window starting at the current head, of duration `target_duration_s`.

        Raises RuntimeError if not enough data buffered.
        """
        if not self.ready() or self._head_t is None:
            raise RuntimeError("WindowBuffer.emit() called before ready()")
        t0 = self._head_t
        t1 = t0 + self._target
        window = self._build_window(t0, t1)
        # Advance head.
        self._head_t = t0 + self._stride
        self._gc_before(self._head_t)
        return window

    def __iter__(self) -> Iterator[StimulusWindow]:
        while self.ready():
            yield self.emit()

    # --------------------------------------------------------------- internal
    def _build_window(self, t0: float, t1: float) -> StimulusWindow:
        # AUDIO: concatenate samples that fall within [t0, t1].
        audio_arrays: list[np.ndarray] = []
        sr_used: int | None = None
        for chunk in self._audio:
            chunk_end = chunk.t_start + chunk.samples.shape[0] / chunk.sr
            if chunk_end <= t0 or chunk.t_start >= t1:
                continue
            sr_used = chunk.sr if sr_used is None else sr_used
            if chunk.sr != sr_used:
                raise ValueError(
                    "WindowBuffer does not resample; all audio chunks must share sr"
                )
            local_start = max(0.0, t0 - chunk.t_start)
            local_end = min(
                chunk.samples.shape[0] / chunk.sr,
                t1 - chunk.t_start,
            )
            i0 = int(round(local_start * chunk.sr))
            i1 = int(round(local_end * chunk.sr))
            if i1 > i0:
                audio_arrays.append(chunk.samples[i0:i1])
        if not audio_arrays:
            raise RuntimeError("no audio in window")
        audio = np.concatenate(audio_arrays, axis=0)
        assert sr_used is not None

        # VIDEO: concatenate frames whose chunk overlaps the window. Mirror the
        # audio slicing logic so the resulting frame count tracks the requested
        # window precisely.
        video_arrays: list[np.ndarray] = []
        fps_used: float | None = None
        for chunk in self._video:
            chunk_end = chunk.t_start + chunk.frames.shape[0] / chunk.fps
            if chunk_end <= t0 or chunk.t_start >= t1:
                continue
            fps_used = chunk.fps if fps_used is None else fps_used
            local_start = max(0.0, t0 - chunk.t_start)
            local_end = min(
                chunk.frames.shape[0] / chunk.fps,
                t1 - chunk.t_start,
            )
            i0 = int(round(local_start * chunk.fps))
            i1 = int(round(local_end * chunk.fps))
            if i1 > i0:
                video_arrays.append(chunk.frames[i0:i1])
        if not video_arrays:
            raise RuntimeError("no video in window")
        video = np.concatenate(video_arrays, axis=0)
        assert fps_used is not None

        # TEXT: most recent text whose timestamp falls in the window.
        text_in = [c for c in self._text if t0 <= c.t < t1]
        text = text_in[-1].text if text_in else ""

        # Compute true duration from audio (most reliable).
        duration_s = float(audio.shape[0]) / float(sr_used)
        if not (self._min_dur <= duration_s <= self._max_dur):
            # Clamp to allowed range — only happens when target_duration_s is
            # set wildly outside [25, 35] which we already guarded above.
            raise RuntimeError(
                f"window duration_s={duration_s:.2f} outside [{self._min_dur}, {self._max_dur}]"
            )

        return StimulusWindow(
            window_id=f"buf-{uuid.uuid4().hex[:8]}",
            t_start=t0,
            duration_s=duration_s,
            video=video,
            video_fps=fps_used,
            audio=audio,
            audio_sr=sr_used,
            text=text,
        )

    def _gc_before(self, t0: float) -> None:
        """Drop chunks fully before t0."""
        self._audio = [
            c for c in self._audio if c.t_start + c.samples.shape[0] / c.sr > t0
        ]
        self._video = [
            c for c in self._video if c.t_start + c.frames.shape[0] / c.fps > t0
        ]
        self._text = [c for c in self._text if c.t >= t0]
