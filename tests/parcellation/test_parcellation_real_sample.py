"""Integration test against the real TRIBE v2 sample output (31, 20484).

This test runs end-to-end through GlasserParcellationUnit using the real
verbatim sample committed at agent/tribev2_sample_output.txt.

The default GlasserParcellationUnit() falls back to a deterministic synthetic
atlas when real .annot files are not present on the host. The acceptance
checks below are written to be valid for any consistent labeling and so
remain meaningful in either mode.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.fixtures.load_sample import load_tribe_sample
from tribe_backend.contracts import EXPECTED_T_PER_30S_WINDOW
from tribe_backend.parcellation import GlasserParcellationUnit, Report


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_output():
    return load_tribe_sample()


@pytest.fixture(scope="module")
def unit():
    # Default labeling — falls back to synthetic atlas if real .annot files are absent.
    # Clear caches so we don't reuse a fake-atlas state cached by unit tests.
    from tribe_backend.parcellation import atlas as atlas_mod
    atlas_mod.clear_caches()
    return GlasserParcellationUnit()


def test_real_sample_has_expected_shape(real_output):
    assert real_output.cortical.shape == (EXPECTED_T_PER_30S_WINDOW, 20484)


def test_generate_report_on_real_sample_returns_nonempty(unit, real_output):
    # Real sample is z-scored BOLD with window-mean magnitudes ~0.1–0.6;
    # use a threshold tuned to that range so the report is non-empty.
    report = unit.generate_report(real_output, z_threshold=0.1)
    assert isinstance(report, Report)
    assert len(report.top_regions) >= 1, "expected at least one region above threshold"
    assert report.window_id == real_output.window_id


def test_top_region_zscore_matches_parcel_mean_on_real_sample(unit, real_output):
    """report.top_regions[0].z_score == cortical.mean(axis=0)[parcel_mask].mean()."""
    report = unit.generate_report(real_output, z_threshold=0.1)
    if not report.top_regions:
        pytest.skip("no regions above threshold on real sample")
    top = report.top_regions[0]
    # Look up the parcel's mask through the unit's internal masks.
    if top.name in unit._cort_masks:
        mask = unit._cort_masks[top.name]
        expected = float(real_output.cortical.mean(axis=0)[mask].mean())
    elif top.name in unit._sub_masks:
        mask = unit._sub_masks[top.name]
        expected = float(real_output.subcortical.mean(axis=0)[mask].mean())
    else:
        pytest.fail(f"top region {top.name!r} not in any mask")
    assert np.isclose(top.z_score, expected, atol=1e-4)


def test_all_360_cortical_parcels_have_31_timepoint_series(unit, real_output):
    series = unit.parcellate_cortical(real_output.cortical)
    assert len(series) == 360
    for v in series.values():
        assert v.shape == (EXPECTED_T_PER_30S_WINDOW,)


def test_real_sample_report_roundtrips_via_json(unit, real_output):
    report = unit.generate_report(real_output, z_threshold=0.1)
    blob = report.to_json()
    rebuilt = Report.from_json(blob)
    assert rebuilt.window_id == report.window_id
    assert rebuilt.text == report.text
    assert len(rebuilt.top_regions) == len(report.top_regions)


# ---------------------------------------------------------------------------
# IBC validation harness — deferred until real localizers are wired in
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="needs IBC localizers")
def test_ibc_face_contrast_lights_up_face_region():
    """When face-vs-baseline localizer is run, FFC / STSva / PGi rank top."""
    raise NotImplementedError


@pytest.mark.skip(reason="needs IBC localizers")
def test_ibc_place_contrast_lights_up_place_region():
    raise NotImplementedError


@pytest.mark.skip(reason="needs IBC localizers")
def test_ibc_body_contrast_lights_up_body_region():
    raise NotImplementedError
