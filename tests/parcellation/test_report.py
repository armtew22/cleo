"""Tests for GlasserParcellationUnit.generate_report and the Report dataclass."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
    TribeOutput,
)
from tribe_backend.parcellation.unit import (
    GlasserParcellationUnit,
    Report,
    RegionActivation,
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


def _make_output(cortical: np.ndarray, *, window_id: str = "wid-1") -> TribeOutput:
    sub = np.zeros((cortical.shape[0], SUBCORTICAL_VOXELS), dtype=np.float32)
    return TribeOutput(cortical=cortical.astype(np.float32), subcortical=sub, window_id=window_id)


# ---------------------------------------------------------------------------
# Window-mean drives the narrative (not the peak)
# ---------------------------------------------------------------------------
def test_amygdala_alternating_yields_moderate_template(unit):
    """Amygdala vertices alternate [+4, 0, +4, 0, ...] -> mean ~ +2 -> 'moderate'."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    pattern = np.zeros(T, dtype=np.float32)
    pattern[::2] = 4.0  # 0,2,4,...
    cort[:, 0:100] = pattern[:, None]

    out = _make_output(cort)
    report = unit.generate_report(out)

    text = report.text.lower()
    assert "amygdala" in text
    assert "moderate" in text
    assert "strong" not in text


def test_amygdala_transient_spike_filtered_by_window_mean(unit):
    """Amygdala spikes only at t=15 to +10 -> window mean ~ 0.32 < threshold 1.5."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[15, 0:100] = 10.0

    out = _make_output(cort)
    report = unit.generate_report(out, method="window_mean", z_threshold=1.5)
    names = [r.name for r in report.top_regions]
    assert "Amygdala" not in names

    # Switching to peak surfaces the transient.
    report_peak = unit.generate_report(out, method="peak", z_threshold=1.5)
    names_peak = [r.name for r in report_peak.top_regions]
    assert "Amygdala" in names_peak


def test_empty_report_when_nothing_crosses_threshold(unit):
    """No region above threshold -> non-empty Report with empty top_regions."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    out = _make_output(cort)
    report = unit.generate_report(out, z_threshold=1.5)
    assert isinstance(report, Report)
    assert report.top_regions == []
    assert "no significant" in report.text.lower()


def test_report_window_id_traces_to_output(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = 3.0
    out = _make_output(cort, window_id="my-special-window")
    report = unit.generate_report(out)
    assert report.window_id == "my-special-window"


def test_report_text_includes_top_region(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = 3.0  # Amygdala -> +3
    out = _make_output(cort)
    report = unit.generate_report(out)
    assert any(r.name == "Amygdala" for r in report.top_regions)
    assert "amygdala" in report.text.lower()


def test_report_top_regions_sorted_descending(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = 5.0      # Amygdala  +5
    cort[:, 5000:5200] = 3.0  # FFC       +3
    cort[:, 10242:10500] = 2.0  # V1      +2
    out = _make_output(cort)
    report = unit.generate_report(out, top_k=10, z_threshold=1.5)
    zs = [abs(r.z_score) for r in report.top_regions]
    assert zs == sorted(zs, reverse=True)
    # Amygdala (+5) is the top.
    assert report.top_regions[0].name == "Amygdala"


def test_report_strong_template_for_high_activation(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = 6.0  # very strong
    out = _make_output(cort)
    report = unit.generate_report(out)
    assert "strong" in report.text.lower()


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------
def test_report_serializes_to_json(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = 3.0
    cort[:, 5000:5200] = 2.0
    out = _make_output(cort, window_id="json-window")
    report = unit.generate_report(out)

    blob = report.to_json()
    parsed = json.loads(blob)
    assert parsed["window_id"] == "json-window"
    assert "top_regions" in parsed
    assert "text" in parsed


def test_report_roundtrips_via_json(unit):
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    cort[:, 0:100] = 3.0
    cort[:, 5000:5200] = 2.0
    out = _make_output(cort, window_id="rt-window")
    report = unit.generate_report(out)

    blob = report.to_json()
    rebuilt = Report.from_json(blob)
    assert rebuilt.window_id == report.window_id
    assert rebuilt.text == report.text
    assert len(rebuilt.top_regions) == len(report.top_regions)
    for a, b in zip(rebuilt.top_regions, report.top_regions):
        assert a.name == b.name
        assert np.isclose(a.z_score, b.z_score)


def test_report_default_method_is_window_mean(unit):
    """If no method specified, default must be window_mean."""
    T = EXPECTED_T_PER_30S_WINDOW
    cort = np.zeros((T, CORTICAL_VERTICES), dtype=np.float32)
    # Plant identical signal across all 31 frames to make window_mean obvious.
    cort[:, 0:100] = 3.0
    out = _make_output(cort)
    r_default = unit.generate_report(out)
    r_explicit = unit.generate_report(out, method="window_mean")
    assert r_default.method == r_explicit.method == "window_mean"
