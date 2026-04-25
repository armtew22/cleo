"""Phase 5 — real GPU adapter for the upstream tribev2 engine.

Wraps `tribev2.TribeModel` so it satisfies the `TribeInference` protocol from
`tribe_backend.contracts`. Upstream tribev2 is NOT modified — this file is the
boundary.

The upstream engine takes a video file path; our `StimulusWindow` carries raw
arrays. The adapter materializes the StimulusWindow to a temp .mp4 (with audio
muxed in) and feeds that path to `TribeModel.predict(...)`.

Imports torch and tribev2 lazily so the rest of `tribe_backend` stays pure-CPU.
The reference implementation pattern lives at /home/md2292/cleo/tribe_inference.py
(do not modify it).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
    StimulusWindow,
    TribeOutput,
)
from tribe_backend.inference.protocol import InferenceFailure


DEFAULT_HF_HOME = "/home/md2292/tribev2-bench/hfcache"
DEFAULT_CACHE = "/home/md2292/tribev2-bench/cache"
DEFAULT_HF_TOKEN_FILE = "/share/goyal/lio/huggingface/token"
DEFAULT_MODEL_ID = "facebook/tribev2"


class WeightsNotFoundError(InferenceFailure):
    """Raised at __init__ when model weights cannot be located/loaded."""


def _ensure_ffmpeg() -> str:
    bin_ = shutil.which("ffmpeg")
    if not bin_:
        raise RuntimeError(
            "ffmpeg is required to mux StimulusWindow video+audio into an mp4 "
            "for the tribev2 engine; install it or pre-render the clip"
        )
    return bin_


def _materialize_window_to_mp4(window: StimulusWindow, out_path: Path) -> Path:
    """Write the window's video+audio into a single .mp4 the engine can read.

    Uses ffmpeg via two pipes: raw video frames on stdin (rgb24) plus a
    temporary .wav for the audio track, then mux.
    """
    ffmpeg = _ensure_ffmpeg()
    F, H, W, _ = window.video.shape

    # Write audio to a temp .wav (avoid a second pipe — ffmpeg single-stdin only).
    import wave
    audio = window.audio
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    audio_path = out_path.with_suffix(".wav")
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(window.audio_sr))
        wf.writeframes(pcm.tobytes())

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", f"{window.video_fps}",
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        input=window.video.astype(np.uint8, copy=False).tobytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise InferenceFailure(
            f"ffmpeg failed muxing StimulusWindow {window.window_id!r}: "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )
    audio_path.unlink(missing_ok=True)
    return out_path


class GpuTribeInference:
    """Adapter wrapping `tribev2.TribeModel` to satisfy `TribeInference`.

    Loads the upstream model once at __init__. Per-window forward pass:
      StimulusWindow -> temp .mp4 -> TribeModel.predict(...) -> TribeOutput.

    Subcortical predictions: the upstream model exposes only the cortical
    head in `predict()`; subcortical is zero-filled with shape (T, 8802) to
    satisfy the contract. (When the upstream subcortical head is exposed,
    populate it here without changing any other compartment.)
    """

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        hf_home: str = DEFAULT_HF_HOME,
        cache_folder: str = DEFAULT_CACHE,
        hf_token_file: str | None = DEFAULT_HF_TOKEN_FILE,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", f"{hf_home}/hub")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if (
            hf_token_file
            and Path(hf_token_file).exists()
            and "HF_TOKEN" not in os.environ
        ):
            tok = Path(hf_token_file).read_text().strip()
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGINGFACE_HUB_TOKEN"] = tok

        try:
            import torch  # type: ignore[import-not-found]
            from tribev2 import TribeModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise WeightsNotFoundError(
                f"torch/tribev2 not importable in this env: {e}"
            ) from e

        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        self._device = device
        self._torch = torch
        try:
            self._model = TribeModel.from_pretrained(model_id, cache_folder=cache_folder)
            self._model._model.eval()
        except Exception as e:
            raise WeightsNotFoundError(
                f"failed to load tribev2 weights for {model_id!r}: {e}"
            ) from e

    def __call__(self, window: StimulusWindow) -> TribeOutput:
        torch = self._torch
        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / f"{window.window_id}.mp4"
            try:
                _materialize_window_to_mp4(window, mp4)
            except Exception as e:
                raise InferenceFailure(
                    f"failed to materialize window {window.window_id!r}: {e}"
                ) from e

            try:
                events = self._model.get_events_dataframe(video_path=str(mp4))
                with torch.inference_mode():
                    preds, _segments = self._model.predict(events=events, verbose=False)
            except torch.cuda.OutOfMemoryError as e:                       # type: ignore[attr-defined]
                raise InferenceFailure(
                    f"CUDA OOM on window {window.window_id!r}: {e}"
                ) from e
            except Exception as e:
                raise InferenceFailure(
                    f"tribev2 forward pass failed on window "
                    f"{window.window_id!r}: {e}"
                ) from e

        arr = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
        cortical = arr.astype(np.float32, copy=False)

        if cortical.ndim != 2 or cortical.shape[1] != CORTICAL_VERTICES:
            raise InferenceFailure(
                f"engine returned unexpected cortical shape {cortical.shape}; "
                f"expected (T, {CORTICAL_VERTICES})"
            )
        if not np.isfinite(cortical).all():
            raise InferenceFailure(
                f"engine returned non-finite values on window {window.window_id!r}"
            )

        T = cortical.shape[0]
        subcortical = np.zeros((T, SUBCORTICAL_VOXELS), dtype=np.float32)

        return TribeOutput(
            cortical=cortical,
            subcortical=subcortical,
            fps=1.0,
            surface="fsaverage5",
            subcortical_atlas="harvard_oxford_2mm",
            window_id=window.window_id,
        )


__all__ = ["GpuTribeInference", "WeightsNotFoundError"]
