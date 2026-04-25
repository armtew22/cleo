"""Tests for tribe_backend.mesh.geometry — fsaverage5 surface loader + normals."""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    LH_VERTICES,
    RH_VERTICES,
)
from tribe_backend.mesh import geometry as geom


pytestmark = pytest.mark.unit


@pytest.mark.integration
def test_load_fsaverage5_pial_returns_expected_shapes_and_dtypes():
    coords, faces = geom.load_fsaverage5("pial")
    assert coords.shape == (CORTICAL_VERTICES, 3)
    assert coords.dtype == np.float32
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert faces.dtype == np.int32
    # fsaverage5 has 20480 faces per hemisphere -> 40960 total
    assert faces.shape[0] == 40960


@pytest.mark.integration
def test_load_fsaverage5_face_indices_offset_correctly_for_rh():
    coords, faces = geom.load_fsaverage5("pial")
    # Every face index must be in [0, 20484)
    assert faces.min() >= 0
    assert faces.max() < CORTICAL_VERTICES
    # First half of faces (LH) should index into [0, LH_VERTICES)
    n_lh_faces = 20480
    lh_faces = faces[:n_lh_faces]
    rh_faces = faces[n_lh_faces:]
    assert lh_faces.max() < LH_VERTICES
    # RH faces must reference the RH block [LH_VERTICES, CORTICAL_VERTICES)
    assert rh_faces.min() >= LH_VERTICES
    assert rh_faces.max() < CORTICAL_VERTICES


@pytest.mark.integration
def test_load_fsaverage5_invalid_surface_raises():
    with pytest.raises(ValueError):
        geom.load_fsaverage5("not_a_surface")


def test_compute_vertex_normals_unit_length():
    # 4 verts, 2 triangles forming a flat plane in z=0 with normals = +z.
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    normals = geom._compute_vertex_normals(coords, faces)
    assert normals.shape == coords.shape
    assert normals.dtype == np.float32
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_compute_vertex_normals_known_flat_triangle_pointing_up():
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)  # CCW from above -> +z
    normals = geom._compute_vertex_normals(coords, faces)
    expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    np.testing.assert_allclose(normals, expected, atol=1e-6)


def test_compute_vertex_normals_handles_zero_area_without_nan():
    # Degenerate triangle (collinear) should not produce NaN normals after normalize.
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    normals = geom._compute_vertex_normals(coords, faces)
    assert not np.any(np.isnan(normals))
    # All non-degenerate verts (vertex 3) should have a unit normal.
    n3 = normals[3]
    assert abs(np.linalg.norm(n3) - 1.0) < 1e-5
