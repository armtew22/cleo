"""GlasserParcellationUnit — interpret TRIBE v2 cortical/subcortical output as ROI activations.

Built bottom-up:
    1. parcellate_cortical / parcellate_subcortical  — vertex-array -> per-region time series
    2. aggregate                                      — (T, N) -> (N,) reduction
    3. rank_and_threshold                             — top-k z-score selection
    4. generate_report                                — TribeOutput -> Report
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
    TribeOutput,
)
import warnings

from .atlas import load_glasser_cortical, load_harvard_oxford_subcortical


_DESCRIPTIONS_PATH = Path(__file__).with_name("descriptions.json")


def _load_descriptions() -> dict[str, str]:
    if _DESCRIPTIONS_PATH.is_file():
        with _DESCRIPTIONS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


class WindowSizeWarning(UserWarning):
    """Emitted when an aggregation method receives T != EXPECTED_T_PER_30S_WINDOW."""


@dataclass(frozen=True)
class RegionActivation:
    """A single ranked region with its aggregated z-score and direction."""

    name: str
    z_score: float
    direction: str  # "activation" or "deactivation"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Report:
    """Output of GlasserParcellationUnit.generate_report.

    top_regions  — ranked list (descending |z|) above z_threshold.
    text         — qualitative narrative composed from the top regions.
    method       — aggregation method used ("window_mean" by default).
    z_threshold  — the threshold applied to filter regions.
    window_id    — traces back to the StimulusWindow / TribeOutput.
    """

    top_regions: list[RegionActivation]
    text: str
    method: str
    z_threshold: float
    window_id: str | None = None

    def to_json(self) -> str:
        payload = {
            "top_regions": [r.to_dict() for r in self.top_regions],
            "text": self.text,
            "method": self.method,
            "z_threshold": self.z_threshold,
            "window_id": self.window_id,
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, blob: str) -> "Report":
        d = json.loads(blob)
        regions = [RegionActivation(**r) for r in d.get("top_regions", [])]
        return cls(
            top_regions=regions,
            text=d.get("text", ""),
            method=d.get("method", "window_mean"),
            z_threshold=float(d.get("z_threshold", 1.5)),
            window_id=d.get("window_id"),
        )


def _strength_template(z: float) -> str:
    """Map an absolute z-score to a qualitative strength label."""
    a = abs(z)
    if a >= 4.0:
        return "strong"
    if a >= 2.5:
        return "pronounced"
    if a >= 1.5:
        return "moderate"
    return "weak"


def _compose_narrative(
    regions: list[RegionActivation],
    descriptions: dict[str, str],
) -> str:
    if not regions:
        return "No significant activations detected in this window."
    parts: list[str] = []
    for r in regions:
        strength = _strength_template(r.z_score)
        verb = "activation" if r.direction == "activation" else "deactivation"
        desc = descriptions.get(r.name) or r.description or ""
        suffix = f" ({desc})" if desc else ""
        parts.append(
            f"{strength} {verb} in {r.name}{suffix} (z={r.z_score:+.2f})"
        )
    return "Window summary: " + "; ".join(parts) + "."


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
        self._descriptions = _load_descriptions()

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

    # -- layer 4: generate_report -------------------------------------------
    def generate_report(
        self,
        output: TribeOutput,
        *,
        method: str = "window_mean",
        top_k: int = 10,
        z_threshold: float = 1.5,
    ) -> Report:
        """Convert a TribeOutput into a ranked, narrated Report.

        Defaults to window_mean — averages across all 31 timesteps before
        ranking — so the narrative reflects sustained activations, not
        transients. Pass method="peak" or "peak_window" to surface transients.
        """
        cort_series = self.parcellate_cortical(output.cortical)
        sub_series = self.parcellate_subcortical(output.subcortical)
        all_series = {**cort_series, **sub_series}

        agg = self.aggregate(all_series, method=method)
        ranked = self.rank_and_threshold(agg, top_k=top_k, z_threshold=z_threshold)

        regions = [
            RegionActivation(
                name=name,
                z_score=float(z),
                direction="activation" if z >= 0 else "deactivation",
                description=self._descriptions.get(name, ""),
            )
            for name, z in ranked
        ]
        text = _compose_narrative(regions, self._descriptions)
        return Report(
            top_regions=regions,
            text=text,
            method=method,
            z_threshold=float(z_threshold),
            window_id=output.window_id,
        )
