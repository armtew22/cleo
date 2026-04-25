"""Tests for tribe_backend.parcellation.atlas — Glasser + Harvard-Oxford loaders."""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    LH_VERTICES,
    RH_VERTICES,
    SUBCORTICAL_VOXELS,
)
from tribe_backend.parcellation import atlas as atlas_mod


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — fake `nibabel.freesurfer.read_annot` output for a Glasser-like atlas
# ---------------------------------------------------------------------------
def _fake_lh_annot() -> tuple[np.ndarray, np.ndarray, list[bytes]]:
    """LH: 10242 vertices, 181 parcels (0=unknown, 1..180=L_*).

    Returns (labels, ctab, names) like nibabel.freesurfer.read_annot.
    """
    labels = np.zeros(LH_VERTICES, dtype=np.int32)
    chunks = np.array_split(np.arange(LH_VERTICES), 180)
    for i, chunk in enumerate(chunks):
        labels[chunk] = i + 1
    ctab = np.zeros((181, 5), dtype=np.int64)
    names = [b"L_unknown"] + [f"L_P{i:03d}".encode() for i in range(1, 181)]
    return labels, ctab, names


def _fake_rh_annot() -> tuple[np.ndarray, np.ndarray, list[bytes]]:
    """RH: 10242 vertices, 181 parcels (0=unknown, 1..180=R_*)."""
    labels = np.zeros(RH_VERTICES, dtype=np.int32)
    chunks = np.array_split(np.arange(RH_VERTICES), 180)
    for i, chunk in enumerate(chunks):
        labels[chunk] = i + 1
    ctab = np.zeros((181, 5), dtype=np.int64)
    names = [b"R_unknown"] + [f"R_P{i:03d}".encode() for i in range(1, 181)]
    return labels, ctab, names


# ---------------------------------------------------------------------------
# Cortical loader
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_annot_paths(tmp_path, monkeypatch):
    """Create empty placeholder .annot files and point the loader at them."""
    lh = tmp_path / "lh.HCP-MMP1.annot"
    rh = tmp_path / "rh.HCP-MMP1.annot"
    lh.write_bytes(b"")
    rh.write_bytes(b"")
    monkeypatch.setenv("TRIBE_GLASSER_LH", str(lh))
    monkeypatch.setenv("TRIBE_GLASSER_RH", str(rh))
    return lh, rh


def test_load_glasser_cortical_returns_correct_shape_and_range(monkeypatch, fake_annot_paths):
    """The cortical loader returns (20484,) int with values in [0, 360]."""
    calls = {"n": 0}

    def fake_read_annot(path):
        calls["n"] += 1
        if "lh" in str(path).lower():
            return _fake_lh_annot()
        return _fake_rh_annot()

    monkeypatch.setattr("nibabel.freesurfer.read_annot", fake_read_annot)
    atlas_mod.load_glasser_cortical.cache_clear()

    labels, names = atlas_mod.load_glasser_cortical()

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (CORTICAL_VERTICES,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= 0
    assert labels.max() <= 360
    # Each hemisphere should be covered (most vertices labeled in our fake)
    assert (labels[:LH_VERTICES] > 0).sum() > LH_VERTICES * 0.9
    assert (labels[LH_VERTICES:] > 0).sum() > RH_VERTICES * 0.9
    # Names dict maps every nonzero id present in labels.
    for lid in np.unique(labels):
        if lid != 0:
            assert int(lid) in names


def test_load_glasser_cortical_is_cached(monkeypatch, fake_annot_paths):
    """Second call must NOT re-invoke nibabel.freesurfer.read_annot."""
    calls = {"n": 0}

    def fake_read_annot(path):
        calls["n"] += 1
        if "lh" in str(path).lower():
            return _fake_lh_annot()
        return _fake_rh_annot()

    monkeypatch.setattr("nibabel.freesurfer.read_annot", fake_read_annot)
    atlas_mod.load_glasser_cortical.cache_clear()

    atlas_mod.load_glasser_cortical()
    n_after_first = calls["n"]
    atlas_mod.load_glasser_cortical()
    n_after_second = calls["n"]

    assert n_after_first == n_after_second, "Atlas was reloaded on second call"
    assert n_after_first >= 1


def test_load_glasser_cortical_lh_rh_offset(monkeypatch, fake_annot_paths):
    """LH and RH parcel ids must be disjoint after offsetting."""
    def fake_read_annot(path):
        if "lh" in str(path).lower():
            return _fake_lh_annot()
        return _fake_rh_annot()

    monkeypatch.setattr("nibabel.freesurfer.read_annot", fake_read_annot)
    atlas_mod.load_glasser_cortical.cache_clear()

    labels, names = atlas_mod.load_glasser_cortical()

    lh_ids = set(np.unique(labels[:LH_VERTICES])) - {0}
    rh_ids = set(np.unique(labels[LH_VERTICES:])) - {0}
    assert lh_ids.isdisjoint(rh_ids), "LH and RH parcel ids overlap"


# ---------------------------------------------------------------------------
# Subcortical loader
# ---------------------------------------------------------------------------
def test_load_harvard_oxford_subcortical_returns_correct_shape_and_range():
    labels, names = atlas_mod.load_harvard_oxford_subcortical()
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (SUBCORTICAL_VOXELS,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= 0
    assert labels.max() <= 8
    for lid in np.unique(labels):
        if lid != 0:
            assert int(lid) in names


def test_load_harvard_oxford_subcortical_is_cached():
    atlas_mod.load_harvard_oxford_subcortical.cache_clear()
    a1, n1 = atlas_mod.load_harvard_oxford_subcortical()
    a2, n2 = atlas_mod.load_harvard_oxford_subcortical()
    # Cached: same array object, not just equal.
    assert a1 is a2
    assert n1 is n2
