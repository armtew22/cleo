"""Animation bundle — T color files + geometry written once."""
from __future__ import annotations

import pytest

from tribe_backend.contracts import CORTICAL_VERTICES


pytestmark = [pytest.mark.unit, pytest.mark.integration]


def test_animation_bundle_writes_T_color_files_and_geometry_once(
    tmp_path, exporter, small_output
):
    T = small_output.cortical.shape[0]
    exporter.export_animation_bundle(small_output.cortical, tmp_path)

    # Geometry written exactly once.
    verts_files = list(tmp_path.glob("brain_vertices.bin"))
    faces_files = list(tmp_path.glob("brain_faces.bin"))
    assert len(verts_files) == 1
    assert len(faces_files) == 1
    assert verts_files[0].stat().st_size == CORTICAL_VERTICES * 12

    # T per-frame color files, each (V * 4) bytes.
    color_files = sorted(tmp_path.glob("brain_colors_t*.bin"))
    assert len(color_files) == T
    for cf in color_files:
        assert cf.stat().st_size == CORTICAL_VERTICES * 4
