"""Tests for aggregate (window_mean / peak / peak_window) and rank_and_threshold."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
)
from tribe_backend.parcellation.unit import (
    GlasserParcellationUnit,
    WindowSizeWarning,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def unit(
    synthetic_cortical_labels, synthetic_cortical_names,
    synthetic_subcortical_labels, synthetic_subcortical_names,
):
    return GlasserParcellationUnit(
        cortical_labels=synthetic_cortical_labels,
        cortical_names=synthetic_cortical_names,
        subcortical_labels=synthetic_subcortical_labels,
        subcortical_names=synthetic_subcortical_names,
    )


# ---------------------------------------------------------------------------
# aggregate(method=...)
# ---------------------------------------------------------------------------
def test_aggregate_window_mean_default(unit):
    """vertex with values [0..30] -> window_mean = 15.0."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    # Plant the [0..30] series across all vertices in Amygdala (rows = t).
    cort[:, 0:100] = np.arange(T, dtype=np.float32)[:, None]
    series = unit.parcellate_cortical(cort)
    agg = unit.aggregate(series)  # default == window_mean
    assert np.isclose(agg["Amygdala"], 15.0)


def test_aggregate_window_mean_explicit(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = np.arange(T, dtype=np.float32)[:, None]
    series = unit.parcellate_cortical(cort)
    agg = unit.aggregate(series, method="window_mean")
    assert np.isclose(agg["Amygdala"], 15.0)


def test_aggregate_peak(unit):
    """peak returns the per-region max-abs across time."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[15, 0:100] = 10.0  # peak at t=15
    cort[5, 0:100] = -7.0   # negative spike, smaller magnitude
    series = unit.parcellate_cortical(cort)
    agg = unit.aggregate(series, method="peak")
    assert np.isclose(agg["Amygdala"], 10.0)


def test_aggregate_peak_window(unit):
    """peak_window returns the mean over a small window around the peak frame."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[13:18, 0:100] = 5.0  # 5-frame plateau centered on t=15
    series = unit.parcellate_cortical(cort)
    agg = unit.aggregate(series, method="peak_window")
    # Peak window covers the plateau -> ~5.0
    assert agg["Amygdala"] > 4.0


def test_aggregate_unknown_method_raises(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    series = unit.parcellate_cortical(cort)
    with pytest.raises(ValueError):
        unit.aggregate(series, method="bogus")


def test_aggregate_warns_on_non_31_T(unit):
    """T != EXPECTED_T_PER_30S_WINDOW emits WindowSizeWarning, not an error."""
    T = 17  # not 31
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    series = unit.parcellate_cortical(cort)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agg = unit.aggregate(series)
    assert any(issubclass(w.category, WindowSizeWarning) for w in caught)
    # Still returns a usable result.
    assert "Amygdala" in agg


def test_aggregate_T_31_does_not_warn(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    series = unit.parcellate_cortical(cort)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unit.aggregate(series)
    assert not any(issubclass(w.category, WindowSizeWarning) for w in caught)


# ---------------------------------------------------------------------------
# rank_and_threshold
# ---------------------------------------------------------------------------
def test_rank_and_threshold_top_k(unit):
    """Returns the top-k regions sorted descending, filtered by z_threshold."""
    agg = {f"P{i:03d}": float(i) for i in range(10)}
    agg["Amygdala"] = 100.0
    agg["FFC"] = 50.0
    ranked = unit.rank_and_threshold(agg, top_k=3, z_threshold=1.5)
    assert len(ranked) == 3
    # Sorted descending.
    assert ranked[0][0] == "Amygdala"
    assert ranked[1][0] == "FFC"
    # All values exceed threshold.
    for _, v in ranked:
        assert v >= 1.5


def test_rank_and_threshold_filters_below_threshold(unit):
    agg = {"A": 1.0, "B": 2.0, "C": 3.0}
    ranked = unit.rank_and_threshold(agg, top_k=10, z_threshold=2.5)
    assert len(ranked) == 1
    assert ranked[0][0] == "C"


def test_rank_and_threshold_empty_when_nothing_passes(unit):
    agg = {"A": 0.1, "B": 0.2}
    ranked = unit.rank_and_threshold(agg, top_k=5, z_threshold=1.5)
    assert ranked == []


def test_rank_and_threshold_uses_absolute_value(unit):
    """Strong negative deactivations should also rank high — by |z|."""
    agg = {"A": 0.1, "B": -3.0, "C": 1.0}
    ranked = unit.rank_and_threshold(agg, top_k=5, z_threshold=1.5)
    names = [n for n, _ in ranked]
    assert names[0] == "B"  # |-3.0| > others
