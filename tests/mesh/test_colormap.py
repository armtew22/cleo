"""Tests for tribe_backend.mesh.colormap.activation_to_rgba."""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.mesh.colormap import activation_to_rgba


pytestmark = pytest.mark.unit


def test_output_dtype_and_shape():
    activations = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    rgba = activation_to_rgba(activations, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    assert rgba.dtype == np.uint8
    assert rgba.shape == (3, 4)
    assert rgba.min() >= 0 and rgba.max() <= 255


def test_clipping_at_vmin_vmax():
    a = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
    rgba = activation_to_rgba(a, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    # Values past vmin should map to the same color as vmin.
    np.testing.assert_array_equal(rgba[0, :3], rgba[1, :3])
    # Values past vmax should map to the same color as vmax.
    np.testing.assert_array_equal(rgba[-1, :3], rgba[-2, :3])


def test_zero_maps_to_midpoint_for_RdBu_r():
    a = np.array([0.0], dtype=np.float32)
    rgba = activation_to_rgba(a, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    # RdBu_r midpoint is white-ish (R~G~B and all high). Tolerance accounts
    # for matplotlib's interpolated midpoint.
    r, g, b = int(rgba[0, 0]), int(rgba[0, 1]), int(rgba[0, 2])
    assert max(r, g, b) - min(r, g, b) <= 8, f"midpoint not gray-ish: {(r,g,b)}"
    assert min(r, g, b) > 200, f"midpoint should be light: {(r,g,b)}"


def test_alpha_default_full_opacity():
    a = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    rgba = activation_to_rgba(a, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    np.testing.assert_array_equal(rgba[:, 3], 255)


def test_sulcal_darkening_reduces_rgb_but_not_alpha():
    a = np.array([0.5, 0.5], dtype=np.float32)
    sulcal = np.array([0.0, 1.0], dtype=np.float32)  # second is fully sulcal (deep)
    rgba = activation_to_rgba(
        a, vmin=-1.0, vmax=1.0, cmap="RdBu_r", sulcal_depth=sulcal, sulcal_strength=0.5
    )
    # RGB of sulcal vertex should be <= RGB of gyral vertex elementwise (darker).
    assert (rgba[1, :3] <= rgba[0, :3]).all()
    assert rgba[1, :3].sum() < rgba[0, :3].sum()
    # Alpha untouched.
    assert rgba[0, 3] == 255 and rgba[1, 3] == 255


def test_handles_2d_input():
    # (T, V) input — colormap should broadcast and return (T, V, 4).
    a = np.zeros((2, 3), dtype=np.float32)
    rgba = activation_to_rgba(a, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    assert rgba.shape == (2, 3, 4)
    assert rgba.dtype == np.uint8
