"""GLB export round-trip via trimesh."""
from __future__ import annotations

import pytest

from tribe_backend.contracts import CORTICAL_VERTICES


trimesh = pytest.importorskip("trimesh")

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_glb_roundtrip_vertex_count(tmp_path, exporter, small_output):
    path = exporter.export_glb(small_output, tmp_path)
    assert path.exists()
    reloaded = trimesh.load(path, force="mesh")
    assert reloaded.vertices.shape[0] == CORTICAL_VERTICES
