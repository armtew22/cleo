"""Shared test fixtures for tribe_backend.

This module owns ONLY the synthetic TribeOutput / StimulusWindow factories.
Compartment-specific fixtures live in the compartment's own conftest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
    StimulusWindow,
    TribeOutput,
)

# Tiny, deterministic "parcels" used by synthetic factories so unit tests can
# plant a known signal in a known parcel without loading real atlases.
# These are not the real Glasser parcels — Phase 1 will load those.
_SYNTHETIC_PARCELS = {
    # name -> (start_vertex, end_vertex)  [end exclusive]
    "Amygdala": (0, 100),
    "FFC": (5000, 5200),
    "V1": (10242, 10500),  # first parcel of RH
}


@dataclass(frozen=True)
class HotSpot:
    """A planted signal at a contiguous range of vertices."""

    name: str
    start: int
    end: int
    value: float


def _make_cortical(
    T: int,
    hotspots: Iterable[HotSpot],
    *,
    seed: int,
    noise: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cort = rng.standard_normal((T, CORTICAL_VERTICES)).astype(np.float32) * noise
    for hs in hotspots:
        cort[:, hs.start : hs.end] = hs.value
    return cort


def make_synthetic(
    T: int = EXPECTED_T_PER_30S_WINDOW,
    *,
    seed: int = 0,
    noise: float = 0.1,
    hotspots: Iterable[HotSpot] | None = None,
    window_id: str | None = "synth-window",
) -> TribeOutput:
    """Deterministically produce a valid TribeOutput with planted hot spots.

    Default hot spots (when `hotspots is None`):
      Amygdala vertices [0:100]    -> +3.0
      FFC vertices    [5000:5200]  -> +2.0
      V1 vertices    [10242:10500] -> +4.0
    """
    if hotspots is None:
        hotspots = [
            HotSpot("Amygdala", *_SYNTHETIC_PARCELS["Amygdala"], 3.0),
            HotSpot("FFC", *_SYNTHETIC_PARCELS["FFC"], 2.0),
            HotSpot("V1", *_SYNTHETIC_PARCELS["V1"], 4.0),
        ]
    hotspots = list(hotspots)
    cort = _make_cortical(T, hotspots, seed=seed, noise=noise)
    rng = np.random.default_rng(seed + 1)
    sub = rng.standard_normal((T, SUBCORTICAL_VOXELS)).astype(np.float32) * noise
    return TribeOutput(
        cortical=cort,
        subcortical=sub,
        window_id=window_id,
    )


def make_stimulus_window(
    *,
    window_id: str = "synth-stim",
    duration_s: float = 30.0,
    video_fps: float = 4.0,
    audio_sr: int = 16000,
    text: str = "synthetic",
    seed: int = 0,
) -> StimulusWindow:
    """Tiny synthetic StimulusWindow for protocol tests (downscaled to keep RAM low)."""
    rng = np.random.default_rng(seed)
    F = max(1, int(round(video_fps * duration_s)))
    video = rng.integers(0, 256, size=(F, 16, 16, 3), dtype=np.uint8)
    S = max(1, int(round(audio_sr * duration_s)))
    audio = rng.standard_normal(S, dtype=np.float32) * 0.01
    return StimulusWindow(
        window_id=window_id,
        t_start=0.0,
        duration_s=duration_s,
        video=video,
        video_fps=video_fps,
        audio=audio,
        audio_sr=audio_sr,
        text=text,
    )


@pytest.fixture
def synth_output() -> TribeOutput:
    return make_synthetic()


@pytest.fixture
def synth_window() -> StimulusWindow:
    return make_stimulus_window()


@pytest.fixture
def synthetic_parcels() -> dict[str, tuple[int, int]]:
    return dict(_SYNTHETIC_PARCELS)


# Re-export factories so compartment tests can import them directly.
__all__ = [
    "HotSpot",
    "make_synthetic",
    "make_stimulus_window",
    "synth_output",
    "synth_window",
    "synthetic_parcels",
]
