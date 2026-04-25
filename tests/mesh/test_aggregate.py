"""Tests for tribe_backend.mesh.exporter.aggregate_temporal."""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.contracts import CORTICAL_VERTICES, EXPECTED_T_PER_30S_WINDOW
from tribe_backend.mesh.exporter import aggregate_temporal


pytestmark = pytest.mark.unit


def _planted_peak(T=EXPECTED_T_PER_30S_WINDOW, V=CORTICAL_VERTICES, peak_t=15) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((T, V), dtype=np.float32) * 0.05
    # Plant +5 spike at peak frame across all vertices.
    arr[peak_t, :] = 5.0
    return arr


def test_mean_matches_numpy_mean():
    arr = _planted_peak()
    out = aggregate_temporal(arr, method="mean")
    assert out.shape == (CORTICAL_VERTICES,)
    np.testing.assert_allclose(out, arr.mean(axis=0), rtol=1e-6)


def test_peak_picks_planted_peak_frame_values():
    peak_t = 15
    arr = _planted_peak(peak_t=peak_t)
    out = aggregate_temporal(arr, method="peak")
    assert out.shape == (CORTICAL_VERTICES,)
    # Peak-by-abs at every vertex should be 5.0 (the planted spike dominates noise).
    np.testing.assert_allclose(out, 5.0, atol=1e-6)


def test_max_matches_numpy_max():
    arr = _planted_peak()
    out = aggregate_temporal(arr, method="max")
    np.testing.assert_allclose(out, arr.max(axis=0), rtol=1e-6)


def test_unknown_method_raises():
    arr = np.zeros((4, CORTICAL_VERTICES), dtype=np.float32)
    with pytest.raises(ValueError):
        aggregate_temporal(arr, method="not_a_method")


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        aggregate_temporal(np.zeros(20484, dtype=np.float32), method="mean")
    with pytest.raises(ValueError):
        aggregate_temporal(np.zeros((4, 100), dtype=np.float32), method="mean")
