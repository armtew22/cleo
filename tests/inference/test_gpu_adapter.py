"""Phase 5 — GpuTribeInference adapter tests.

Three layers:
  - structural (CPU, always run): GpuTribeInference satisfies the
    TribeInference protocol; importing the module does not import torch
    until the adapter is touched.
  - port-matching (CPU, always run): the temp-mp4 muxer produces a file the
    upstream engine can ingest (uses ffmpeg).
  - GPU smoke (`@pytest.mark.gpu`): one real forward pass on a real 30s clip
    via ControlUnit.process_window. Skipped unless CUDA + tribev2 weights
    are available.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_stimulus_window
from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    StimulusWindow,
    TribeInference,
)


REAL_30S_CLIP = Path("/home/md2292/tribev2-bench/stimuli/clip_30s.mp4")


# ---------------------------------------------------------------------------
# Structural — no GPU, no torch import
# ---------------------------------------------------------------------------


def test_importing_inference_does_not_import_torch():
    """The package's lazy __getattr__ must keep torch out of sys.modules
    until GpuTribeInference is actually accessed."""
    # Drop torch from sys.modules so we can detect re-imports.
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    # Re-import the inference package fresh.
    if "tribe_backend.inference" in sys.modules:
        importlib.reload(sys.modules["tribe_backend.inference"])
    import tribe_backend.inference  # noqa: F401
    assert "torch" not in sys.modules, (
        "importing tribe_backend.inference should not import torch"
    )


def test_gpu_runner_module_has_required_symbols():
    from tribe_backend.inference import gpu_runner
    assert hasattr(gpu_runner, "GpuTribeInference")
    assert hasattr(gpu_runner, "WeightsNotFoundError")
    assert issubclass(gpu_runner.WeightsNotFoundError, Exception)


def test_gpu_tribe_inference_class_is_tribe_inference_protocol_compatible():
    """Static structural check — class has the expected method shape.

    We do NOT instantiate (that would load weights). We just verify the
    class has __call__(self, window) -> ... so isinstance() against the
    runtime-checkable Protocol would pass once instantiated.
    """
    from tribe_backend.inference.gpu_runner import GpuTribeInference
    assert callable(getattr(GpuTribeInference, "__call__", None))
    # A trivial subclass instance with stubbed __init__ satisfies the protocol.
    class _Stub(GpuTribeInference):
        def __init__(self): pass  # type: ignore[no-untyped-def]
        def __call__(self, window):
            raise NotImplementedError
    assert isinstance(_Stub(), TribeInference)


# ---------------------------------------------------------------------------
# Port-matching — exercise the muxer
# ---------------------------------------------------------------------------


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_materialize_window_to_mp4_produces_playable_file(tmp_path):
    from tribe_backend.inference.gpu_runner import _materialize_window_to_mp4

    window = make_stimulus_window(window_id="mux-1", duration_s=30.0,
                                  video_fps=4.0, audio_sr=16000)
    out = tmp_path / "out.mp4"
    _materialize_window_to_mp4(window, out)
    assert out.exists() and out.stat().st_size > 0

    # ffprobe (ships with ffmpeg) round-trips the duration.
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    duration = float(proc.stdout.strip())
    # Allow generous slack: encoder may round, ffmpeg may drop a frame.
    assert 25.0 <= duration <= 35.0, f"muxed mp4 duration={duration}"


# ---------------------------------------------------------------------------
# GPU smoke — real forward pass via ControlUnit composition root
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA not available in this venv (torch lives in tribev2-bench/.venv)",
)
@pytest.mark.skipif(
    not REAL_30S_CLIP.exists(),
    reason=f"real 30s clip not available at {REAL_30S_CLIP}",
)
def test_gpu_smoke_real_clip_through_control_unit(tmp_path):
    """End-to-end: real GPU + real clip + real composition root.

    Reads the 30s clip, decodes to (F, H, W, 3) + audio via ffmpeg,
    constructs a StimulusWindow, runs through default_control_unit with
    a real GpuTribeInference, and checks the produced report + mesh.
    """
    import cv2  # type: ignore[import-not-found]
    import wave

    cap = cv2.VideoCapture(str(REAL_30S_CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.stack(frames, axis=0).astype(np.uint8)  # (F, H, W, 3)

    wav_path = REAL_30S_CLIP.with_suffix(".wav")
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if wf.getnchannels() == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    from tribe_backend.control import default_control_unit
    from tribe_backend.inference import GpuTribeInference

    inf = GpuTribeInference(device="cuda:0")
    control = default_control_unit(out_dir=tmp_path, inference=inf)

    report = control.process_window(
        video=video, audio=audio, audio_sr=sr,
        text="a 30 second test clip from tribev2-bench/stimuli",
        video_fps=fps,
    )

    # Report is non-empty and traceable.
    assert hasattr(report, "top_regions")
    assert report.window_id is not None
    # Mesh artifacts written.
    out_subdir = tmp_path / report.window_id
    assert (out_subdir / "brain_meta.json").exists()
    assert (out_subdir / "brain_colors.bin").exists()
    assert (out_subdir / "brain_colors.bin").stat().st_size == CORTICAL_VERTICES * 4
