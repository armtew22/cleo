"""Binary export — byte-exact sizes + numpy reload."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tribe_backend.contracts import CORTICAL_VERTICES


pytestmark = [pytest.mark.unit, pytest.mark.integration]


def test_binary_byte_exact_sizes_and_reload(tmp_path, exporter, small_output):
    out = exporter.export_binary(small_output, tmp_path)
    assert out == tmp_path
    meta = json.loads((tmp_path / "brain_meta.json").read_text())
    assert meta["format"] == "tribe_brain_mesh_v1"
    assert meta["vertex_count"] == CORTICAL_VERTICES
    assert meta["bytes_per_color"] == 4
    assert meta["bytes_per_vertex"] == 12

    verts_path = tmp_path / "brain_vertices.bin"
    faces_path = tmp_path / "brain_faces.bin"
    colors_path = tmp_path / "brain_colors.bin"
    assert verts_path.stat().st_size == CORTICAL_VERTICES * 12
    assert colors_path.stat().st_size == CORTICAL_VERTICES * 4
    assert faces_path.stat().st_size == 40960 * 12

    verts = np.fromfile(verts_path, dtype=np.float32).reshape(-1, 3)
    faces = np.fromfile(faces_path, dtype=np.int32).reshape(-1, 3)
    colors = np.fromfile(colors_path, dtype=np.uint8).reshape(-1, 4)
    assert verts.shape == (CORTICAL_VERTICES, 3)
    assert faces.shape == (40960, 3)
    assert colors.shape == (CORTICAL_VERTICES, 4)
