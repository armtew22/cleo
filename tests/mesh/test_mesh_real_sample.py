"""Real-sample acceptance test for BrainMeshExporter — Phase 2 PR gate.

Loads agent/tribev2_sample_output.txt and exercises every export path on it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.fixtures.load_sample import load_tribe_sample
from tribe_backend.contracts import CORTICAL_VERTICES, EXPECTED_T_PER_30S_WINDOW
from tribe_backend.mesh import aggregate_temporal
from tribe_backend.mesh.exporter import BrainMeshExporter

trimesh = pytest.importorskip("trimesh")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_output():
    return load_tribe_sample()


@pytest.fixture(scope="module")
def real_exporter():
    return BrainMeshExporter()


def test_real_sample_has_expected_shape(real_output):
    assert real_output.cortical.shape == (EXPECTED_T_PER_30S_WINDOW, CORTICAL_VERTICES)


def test_real_sample_binary_export(tmp_path, real_exporter, real_output):
    real_exporter.export_binary(real_output, tmp_path)
    meta = json.loads((tmp_path / "brain_meta.json").read_text())
    assert meta["vertex_count"] == CORTICAL_VERTICES
    assert meta["bytes_per_color"] == 4
    assert (tmp_path / "brain_colors.bin").stat().st_size == CORTICAL_VERTICES * 4


def test_real_sample_aggregate_mean_matches_numpy(real_output):
    out = aggregate_temporal(real_output.cortical, method="mean")
    assert out.shape == (CORTICAL_VERTICES,)
    np.testing.assert_allclose(
        out, real_output.cortical.mean(axis=0), rtol=1e-6, atol=1e-6
    )


def test_real_sample_animation_bundle_writes_31_frames(
    tmp_path, real_exporter, real_output
):
    real_exporter.export_animation_bundle(real_output.cortical, tmp_path)
    expected = [f"brain_colors_t{t:04d}.bin" for t in range(EXPECTED_T_PER_30S_WINDOW)]
    for name in expected:
        path = tmp_path / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size == CORTICAL_VERTICES * 4
    # First and last frame names called out explicitly by the spec.
    assert (tmp_path / "brain_colors_t0000.bin").exists()
    assert (tmp_path / "brain_colors_t0030.bin").exists()


@pytest.mark.slow
def test_real_sample_glb_roundtrip(tmp_path, real_exporter, real_output):
    path = real_exporter.export_glb(real_output, tmp_path)
    reloaded = trimesh.load(path, force="mesh")
    assert reloaded.vertices.shape[0] == CORTICAL_VERTICES
