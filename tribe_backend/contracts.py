"""Frozen data contract between TRIBE inference and downstream consumers.

This module is the ONLY shared module across compartments. It must not import
from anything else inside `tribe_backend`.

Vertex ordering convention (single source of truth):
    cortical[..., 0:10242]      = left hemisphere  (fsaverage5)
    cortical[..., 10242:20484]  = right hemisphere (fsaverage5)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

CORTICAL_VERTICES: int = 20484
SUBCORTICAL_VOXELS: int = 8802
LH_VERTICES: int = 10242
RH_VERTICES: int = 10242
EXPECTED_T_PER_30S_WINDOW: int = 31  # 1 Hz inclusive of t=0 and t=30


@dataclass(frozen=True)
class StimulusWindow:
    """A 30-second multimodal window emitted by the live feed poller."""

    window_id: str
    t_start: float
    duration_s: float
    video: np.ndarray          # (F, H, W, 3) uint8
    video_fps: float
    audio: np.ndarray          # (S,) or (S, C) float32
    audio_sr: int
    text: str

    def __post_init__(self) -> None:
        if not (isinstance(self.video, np.ndarray) and self.video.ndim == 4 and self.video.shape[-1] == 3):
            raise ValueError(
                f"video must be ndarray of shape (F,H,W,3); got "
                f"{type(self.video).__name__} shape="
                f"{getattr(self.video, 'shape', None)}"
            )
        if not (isinstance(self.audio, np.ndarray) and self.audio.ndim in (1, 2)):
            raise ValueError(
                f"audio must be 1D or 2D ndarray; got ndim="
                f"{getattr(self.audio, 'ndim', None)}"
            )
        if not (25.0 <= self.duration_s <= 35.0):
            raise ValueError(f"duration_s must be in [25,35]; got {self.duration_s}")


@dataclass(frozen=True)
class TribeOutput:
    """Output of the TRIBE v2 model for one stimulus window.

    cortical:    (T, 20484) float32, z-scored BOLD; T == 31 for a 30s window.
    subcortical: (T,  8802) float32, z-scored BOLD; same T as cortical.
    """

    cortical: np.ndarray
    subcortical: np.ndarray
    fps: float = 1.0
    surface: str = "fsaverage5"
    subcortical_atlas: str = "harvard_oxford_2mm"
    window_id: str | None = None

    EXPECTED_T_PER_30S_WINDOW: int = field(default=EXPECTED_T_PER_30S_WINDOW, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cortical, np.ndarray):
            raise TypeError(f"cortical must be ndarray; got {type(self.cortical).__name__}")
        if not isinstance(self.subcortical, np.ndarray):
            raise TypeError(f"subcortical must be ndarray; got {type(self.subcortical).__name__}")
        if self.cortical.ndim != 2 or self.cortical.shape[1] != CORTICAL_VERTICES:
            raise ValueError(
                f"cortical must be 2D with shape (T, {CORTICAL_VERTICES}); "
                f"got {self.cortical.shape}"
            )
        if self.subcortical.ndim != 2 or self.subcortical.shape[1] != SUBCORTICAL_VOXELS:
            raise ValueError(
                f"subcortical must be 2D with shape (T, {SUBCORTICAL_VOXELS}); "
                f"got {self.subcortical.shape}"
            )
        if self.cortical.shape[0] != self.subcortical.shape[0]:
            raise ValueError(
                f"cortical and subcortical must share T; "
                f"got {self.cortical.shape[0]} vs {self.subcortical.shape[0]}"
            )

    @property
    def T(self) -> int:
        return int(self.cortical.shape[0])

    def to_npz(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez(
            path,
            cortical=self.cortical,
            subcortical=self.subcortical,
            fps=np.asarray(self.fps),
            surface=np.asarray(self.surface),
            subcortical_atlas=np.asarray(self.subcortical_atlas),
            window_id=np.asarray("" if self.window_id is None else self.window_id),
        )
        # numpy may append .npz; normalize.
        if not path.suffix:
            path = path.with_suffix(".npz")
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> "TribeOutput":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            wid = str(z["window_id"]) if "window_id" in z.files else None
            if wid == "":
                wid = None
            return cls(
                cortical=np.asarray(z["cortical"], dtype=np.float32),
                subcortical=np.asarray(z["subcortical"], dtype=np.float32),
                fps=float(z["fps"]) if "fps" in z.files else 1.0,
                surface=str(z["surface"]) if "surface" in z.files else "fsaverage5",
                subcortical_atlas=(
                    str(z["subcortical_atlas"]) if "subcortical_atlas" in z.files else "harvard_oxford_2mm"
                ),
                window_id=wid,
            )

    def with_window_id(self, window_id: str) -> "TribeOutput":
        return replace(self, window_id=window_id)


@runtime_checkable
class TribeInference(Protocol):
    """The seam between control/ and the GPU model host."""

    def __call__(self, window: StimulusWindow) -> TribeOutput: ...


@runtime_checkable
class LiveFeedPoller(Protocol):
    """Yields 30s windows from a live source."""

    def __iter__(self) -> Iterator[StimulusWindow]: ...
    def __next__(self) -> StimulusWindow: ...
