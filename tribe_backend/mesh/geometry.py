"""fsaverage5 surface loader + vertex-normal computation.

The combined cortical mesh is a single (20484, 3) coords array and a
(40960, 3) faces array. RH face indices are offset by LH_VERTICES (10242)
so that the combined mesh references one unified vertex array — this is
the single source of truth for vertex ordering documented in contracts.py.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Tuple

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_surf_fsaverage

from tribe_backend.contracts import CORTICAL_VERTICES, LH_VERTICES

SurfaceType = Literal["pial", "white", "infl", "sphere", "flat"]
_VALID_SURFACES = {"pial", "white", "infl", "sphere", "flat"}


def _read_gii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a GIFTI surface file and return (coords, faces) as float32/int32."""
    img = nib.load(path)
    coords = None
    faces = None
    for darr in img.darrays:
        # NIFTI_INTENT_POINTSET = 1008, NIFTI_INTENT_TRIANGLE = 1009
        intent = int(darr.intent) if not isinstance(darr.intent, int) else darr.intent
        if intent == 1008 or coords is None and darr.data.ndim == 2 and darr.data.shape[1] == 3 and np.issubdtype(darr.data.dtype, np.floating):
            coords = np.asarray(darr.data, dtype=np.float32)
        elif intent == 1009 or (faces is None and darr.data.ndim == 2 and darr.data.shape[1] == 3 and np.issubdtype(darr.data.dtype, np.integer)):
            faces = np.asarray(darr.data, dtype=np.int32)
    if coords is None or faces is None:
        raise ValueError(f"could not extract coords + faces from {path}")
    return coords, faces


@lru_cache(maxsize=8)
def load_fsaverage5(surface_type: SurfaceType = "pial") -> Tuple[np.ndarray, np.ndarray]:
    """Load the fsaverage5 combined LH+RH surface.

    Parameters
    ----------
    surface_type : one of "pial", "white", "infl", "sphere", "flat"

    Returns
    -------
    coords : (20484, 3) float32 — LH coords stacked above RH coords.
    faces  : (40960, 3) int32  — LH faces (vertex indices in [0, 10242))
                                  followed by RH faces with vertex indices
                                  offset by LH_VERTICES (10242).
    """
    if surface_type not in _VALID_SURFACES:
        raise ValueError(
            f"surface_type must be one of {_VALID_SURFACES}; got {surface_type!r}"
        )
    fs = fetch_surf_fsaverage("fsaverage5")
    # nilearn keys: pial_left, pial_right, white_left, ..., infl_left, ...
    key_left = f"{surface_type}_left"
    key_right = f"{surface_type}_right"
    if not hasattr(fs, key_left) or not hasattr(fs, key_right):
        raise ValueError(
            f"fsaverage5 does not provide surface_type={surface_type!r}; "
            f"available keys must include {key_left}, {key_right}"
        )
    lh_path = getattr(fs, key_left)
    rh_path = getattr(fs, key_right)
    lh_coords, lh_faces = _read_gii(lh_path)
    rh_coords, rh_faces = _read_gii(rh_path)

    if lh_coords.shape[0] != LH_VERTICES:
        raise ValueError(
            f"LH expected {LH_VERTICES} verts, got {lh_coords.shape[0]} ({lh_path})"
        )
    coords = np.concatenate([lh_coords, rh_coords], axis=0).astype(np.float32, copy=False)
    if coords.shape[0] != CORTICAL_VERTICES:
        raise ValueError(
            f"combined cortical verts expected {CORTICAL_VERTICES}, got {coords.shape[0]}"
        )
    faces = np.concatenate(
        [lh_faces, rh_faces + LH_VERTICES],
        axis=0,
    ).astype(np.int32, copy=False)
    return coords, faces


def _compute_vertex_normals(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex unit normals via area-weighted face-normal accumulation."""
    coords = np.asarray(coords, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    # Cross product magnitude == 2 * triangle area, so this is area-weighted.
    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    normals = np.zeros_like(coords, dtype=np.float32)
    # Accumulate face normal at each of its three vertices.
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid divide-by-zero on isolated/degenerate verts.
    safe = np.where(lengths > 0, lengths, 1.0)
    return (normals / safe).astype(np.float32)
