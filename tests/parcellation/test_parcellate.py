"""Tests for parcellate_cortical / parcellate_subcortical."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
)
from tribe_backend.parcellation.unit import GlasserParcellationUnit


pytestmark = pytest.mark.unit


def _make_unit(synth_cort_labels, synth_cort_names, synth_sub_labels, synth_sub_names):
    return GlasserParcellationUnit(
        cortical_labels=synth_cort_labels,
        cortical_names=synth_cort_names,
        subcortical_labels=synth_sub_labels,
        subcortical_names=synth_sub_names,
    )


def test_parcellate_cortical_planted_signal_is_isolated(
    synthetic_cortical_labels, synthetic_cortical_names,
    synthetic_subcortical_labels, synthetic_subcortical_names,
):
    """Plant value 5.0 at FFC vertices, zero elsewhere -> result['FFC'] == 5.0."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 5000:5200] = 5.0  # synthetic FFC range from conftest

    unit = _make_unit(
        synthetic_cortical_labels, synthetic_cortical_names,
        synthetic_subcortical_labels, synthetic_subcortical_names,
    )
    series = unit.parcellate_cortical(cort)

    assert "FFC" in series
    assert series["FFC"].shape == (T,)
    assert np.allclose(series["FFC"], 5.0)
    # Other named parcels are zero.
    for name in ("Amygdala", "V1"):
        assert np.allclose(series[name], 0.0)


def test_parcellate_cortical_dict_has_360_keys(
    synthetic_cortical_labels, synthetic_cortical_names,
    synthetic_subcortical_labels, synthetic_subcortical_names,
):
    T = 4
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    unit = _make_unit(
        synthetic_cortical_labels, synthetic_cortical_names,
        synthetic_subcortical_labels, synthetic_subcortical_names,
    )
    series = unit.parcellate_cortical(cort)
    assert len(series) == 360
    for v in series.values():
        assert v.shape == (T,)


def test_parcellate_subcortical_dict_has_8_keys(
    synthetic_cortical_labels, synthetic_cortical_names,
    synthetic_subcortical_labels, synthetic_subcortical_names,
):
    T = 4
    sub = np.zeros((T, SUBCORTICAL_VOXELS), dtype=np.float32)
    unit = _make_unit(
        synthetic_cortical_labels, synthetic_cortical_names,
        synthetic_subcortical_labels, synthetic_subcortical_names,
    )
    series = unit.parcellate_subcortical(sub)
    assert len(series) == 8
    for v in series.values():
        assert v.shape == (T,)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_parcellate_cortical_mean_equals_weighted_mean_of_vertices(seed):
    """Property: mean of a parcel's time series equals the mean of its vertex
    values across both vertex and time axes."""
    # Build the unit from scratch each time so the property test is
    # self-contained (hypothesis does not play well with pytest fixtures).
    rng = np.random.default_rng(seed)
    T = 8
    cort_labels = np.zeros(CORTICAL_VERTICES, dtype=np.int32)
    cort_labels[0:100] = 1
    cort_labels[5000:5200] = 2
    cort_labels[10242:10500] = 3
    remaining = np.where(cort_labels == 0)[0]
    chunks = np.array_split(remaining, 357)
    for i, chunk in enumerate(chunks):
        cort_labels[chunk] = 4 + i
    cort_names = {i: f"P{i}" for i in range(1, 361)}
    cort_names[1] = "Amygdala"; cort_names[2] = "FFC"; cort_names[3] = "V1"
    sub_labels = np.zeros(SUBCORTICAL_VOXELS, dtype=np.int32)
    sub_chunks = np.array_split(np.arange(SUBCORTICAL_VOXELS), 8)
    for i, chunk in enumerate(sub_chunks):
        sub_labels[chunk] = i + 1
    sub_names = {i: f"S{i}" for i in range(1, 9)}

    unit = GlasserParcellationUnit(
        cortical_labels=cort_labels, cortical_names=cort_names,
        subcortical_labels=sub_labels, subcortical_names=sub_names,
    )
    cort = rng.standard_normal((T, CORTICAL_VERTICES)).astype(np.float32)
    series = unit.parcellate_cortical(cort)

    # Spot-check: for each named parcel, mean across time equals
    # the cross-vertex,cross-time mean of cort restricted to that parcel.
    for name, lid in (("Amygdala", 1), ("FFC", 2), ("V1", 3)):
        mask = cort_labels == lid
        expected = cort[:, mask].mean()
        got = series[name].mean()
        assert np.isclose(got, expected, atol=1e-5), f"{name}: {got} vs {expected}"


def test_parcellate_cortical_rejects_wrong_shape(
    synthetic_cortical_labels, synthetic_cortical_names,
    synthetic_subcortical_labels, synthetic_subcortical_names,
):
    unit = _make_unit(
        synthetic_cortical_labels, synthetic_cortical_names,
        synthetic_subcortical_labels, synthetic_subcortical_names,
    )
    with pytest.raises((ValueError, AssertionError)):
        unit.parcellate_cortical(np.zeros((4, 100), dtype=np.float32))
