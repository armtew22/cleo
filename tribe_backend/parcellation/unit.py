"""GlasserParcellationUnit — interpret TRIBE v2 cortical/subcortical output as ROI activations.

Built bottom-up:
    1. parcellate_cortical / parcellate_subcortical  — vertex-array -> per-region time series
    2. aggregate                                      — (T, N) -> (N,) reduction
    3. rank_and_threshold                             — top-k z-score selection
    4. generate_report                                — TribeOutput -> Report
"""
from __future__ import annotations

import numpy as np

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
    TribeOutput,
)
import warnings

from .atlas import load_glasser_cortical, load_harvard_oxford_subcortical


class WindowSizeWarning(UserWarning):
    """Emitted when an aggregation method receives T != EXPECTED_T_PER_30S_WINDOW."""


class RegionActivation:
    """Placeholder — populated in the report layer."""


class Report:
    """Placeholder — populated in the report layer."""


class GlasserParcellationUnit:
    """ROI interpretation unit.

    Constructor accepts optional pre-loaded labelings — primarily for tests.
    Default behavior loads Glasser-360 + Harvard-Oxford via `atlas.py`.
    """

    def __init__(
        self,
        *,
        cortical_labels: np.ndarray | None = None,
        cortical_names: dict[int, str] | None = None,
        subcortical_labels: np.ndarray | None = None,
        subcortical_names: dict[int, str] | None = None,
    ):
        if cortical_labels is None or cortical_names is None:
            cortical_labels, cortical_names = load_glasser_cortical()
        if subcortical_labels is None or subcortical_names is None:
            subcortical_labels, subcortical_names = load_harvard_oxford_subcortical()

        cortical_labels = np.asarray(cortical_labels, dtype=np.int32)
        subcortical_labels = np.asarray(subcortical_labels, dtype=np.int32)
        if cortical_labels.shape != (CORTICAL_VERTICES,):
            raise ValueError(
                f"cortical_labels must be shape ({CORTICAL_VERTICES},); "
                f"got {cortical_labels.shape}"
            )
        if subcortical_labels.shape != (SUBCORTICAL_VOXELS,):
            raise ValueError(
                f"subcortical_labels must be shape ({SUBCORTICAL_VOXELS},); "
                f"got {subcortical_labels.shape}"
            )

        self._cort_labels = cortical_labels
        self._cort_names = dict(cortical_names)
        self._sub_labels = subcortical_labels
        self._sub_names = dict(subcortical_names)

        # Precompute name -> mask for fast parcellation.
        self._cort_masks: dict[str, np.ndarray] = self._build_masks(
            self._cort_labels, self._cort_names
        )
        self._sub_masks: dict[str, np.ndarray] = self._build_masks(
            self._sub_labels, self._sub_names
        )

    # -- properties ----------------------------------------------------------
    @property
    def cortical_names(self) -> list[str]:
        return list(self._cort_masks.keys())

    @property
    def subcortical_names(self) -> list[str]:
        return list(self._sub_masks.keys())

    @staticmethod
    def _build_masks(labels: np.ndarray, names: dict[int, str]) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for lid, name in names.items():
            if lid == 0:
                continue
            masks[name] = labels == lid
        return masks

    # -- layer 1: parcellate -------------------------------------------------
    def parcellate_cortical(self, cortical: np.ndarray) -> dict[str, np.ndarray]:
        """Reduce (T, 20484) cortical activations to per-parcel time series.

        Returns dict[parcel_name -> (T,) float32]; one entry per named parcel.
        """
        if cortical.ndim != 2 or cortical.shape[1] != CORTICAL_VERTICES:
            raise ValueError(
                f"cortical must be shape (T, {CORTICAL_VERTICES}); got {cortical.shape}"
            )
        return self._parcellate(cortical, self._cort_masks)

    def parcellate_subcortical(self, subcortical: np.ndarray) -> dict[str, np.ndarray]:
        """Reduce (T, 8802) subcortical activations to per-region time series."""
        if subcortical.ndim != 2 or subcortical.shape[1] != SUBCORTICAL_VOXELS:
            raise ValueError(
                f"subcortical must be shape (T, {SUBCORTICAL_VOXELS}); got {subcortical.shape}"
            )
        return self._parcellate(subcortical, self._sub_masks)

    @staticmethod
    def _parcellate(
        arr: np.ndarray, masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        T = arr.shape[0]
        for name, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                out[name] = np.zeros(T, dtype=np.float32)
            else:
                out[name] = arr[:, mask].mean(axis=1).astype(np.float32)
        return out

    # -- layer 2: aggregate --------------------------------------------------
    _AGG_METHODS = ("window_mean", "peak", "peak_window")

    def aggregate(
        self,
        series: dict[str, np.ndarray],
        *,
        method: str = "window_mean",
        peak_window_radius: int = 2,
    ) -> dict[str, float]:
        """Reduce per-region time series to a single z-score per region.

        method:
            "window_mean" (DEFAULT)  mean across all timesteps.
            "peak"                   max-abs across timesteps (signed value preserved).
            "peak_window"            mean of a (2r+1) frame window around |peak|.

        Emits WindowSizeWarning when input T != EXPECTED_T_PER_30S_WINDOW.
        """
        if method not in self._AGG_METHODS:
            raise ValueError(
                f"unknown aggregate method {method!r}; expected one of {self._AGG_METHODS}"
            )
        if not series:
            return {}
        # All series share the same T.
        T = next(iter(series.values())).shape[0]
        if T != EXPECTED_T_PER_30S_WINDOW:
            warnings.warn(
                f"aggregate received T={T}; expected {EXPECTED_T_PER_30S_WINDOW} "
                f"(1 Hz over a 30s window inclusive of t=0 and t=30). "
                f"Aggregation will proceed but results may be partial.",
                WindowSizeWarning,
                stacklevel=2,
            )

        out: dict[str, float] = {}
        for name, ts in series.items():
            if method == "window_mean":
                out[name] = float(ts.mean())
            elif method == "peak":
                idx = int(np.argmax(np.abs(ts)))
                out[name] = float(ts[idx])
            elif method == "peak_window":
                # Find the (2r+1)-frame window with the largest |mean|.
                w = 2 * peak_window_radius + 1
                if T <= w:
                    out[name] = float(ts.mean())
                else:
                    # Sliding window mean via cumulative sum.
                    csum = np.concatenate(([0.0], np.cumsum(ts, dtype=np.float64)))
                    means = (csum[w:] - csum[:-w]) / w
                    best = int(np.argmax(np.abs(means)))
                    out[name] = float(means[best])
        return out

    # -- layer 3: rank -------------------------------------------------------
    def rank_and_threshold(
        self,
        aggregated: dict[str, float],
        *,
        top_k: int = 10,
        z_threshold: float = 1.5,
    ) -> list[tuple[str, float]]:
        """Return [(name, z_value), ...] sorted descending by |z|, filtered by threshold.

        Both positive activations and strong negative deactivations are ranked
        by absolute magnitude; the original (signed) value is returned.
        """
        survivors = [
            (name, float(val))
            for name, val in aggregated.items()
            if abs(val) >= z_threshold
        ]
        survivors.sort(key=lambda nv: abs(nv[1]), reverse=True)
        return survivors[:top_k]
