"""JSON export round-trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tribe_backend.contracts import CORTICAL_VERTICES


pytestmark = [pytest.mark.unit, pytest.mark.integration]


def test_json_roundtrip_preserves_counts_and_color_bytes(tmp_path, exporter, small_output):
    path = exporter.export_json(small_output, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["format"] == "tribe_brain_mesh_v1"
    assert data["vertex_count"] == CORTICAL_VERTICES
    assert data["face_count"] == 40960
    assert len(data["vertices"]) == CORTICAL_VERTICES
    assert len(data["faces"]) == 40960
    assert len(data["colors"]) == CORTICAL_VERTICES
    # Each color is RGBA bytes in [0, 255].
    colors = np.asarray(data["colors"], dtype=np.uint8)
    assert colors.shape == (CORTICAL_VERTICES, 4)
    assert colors.min() >= 0 and colors.max() <= 255
    assert data["window_id"] == small_output.window_id
