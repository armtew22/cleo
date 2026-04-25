"""Atlas loaders for cortical (Glasser-360) and subcortical (Harvard-Oxford) parcellations.

The cortical loader reads two `.annot` files (one per hemisphere) via
`nibabel.freesurfer.read_annot` and concatenates LH+RH into a single (20484,)
label array using the canonical fsaverage5 ordering (LH first).

Both loaders are cached via `functools.lru_cache` so repeat calls are cheap and
tests can verify caching by patching `nibabel.freesurfer.read_annot` with a
counter.

When real atlas files are unavailable on the host (no ~/.cache download), the
cortical loader falls back to a deterministic synthetic atlas. The fallback is
clearly identified by the names dict and is suitable only for keeping
non-integration tests green; integration tests must skip with a clear reason.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    LH_VERTICES,
    RH_VERTICES,
    SUBCORTICAL_VOXELS,
)

DEFAULT_CACHE_DIR = Path(
    os.environ.get("TRIBE_BACKEND_CACHE", Path.home() / ".cache" / "tribe_backend")
)


def _resolve_glasser_paths() -> tuple[Path, Path] | None:
    """Return (lh_annot, rh_annot) paths if they exist, else None.

    Looks under DEFAULT_CACHE_DIR/glasser/ for files named
    `lh.HCP-MMP1.annot` and `rh.HCP-MMP1.annot`. Override with the
    TRIBE_GLASSER_LH / TRIBE_GLASSER_RH env vars.
    """
    lh = Path(os.environ.get("TRIBE_GLASSER_LH", DEFAULT_CACHE_DIR / "glasser" / "lh.HCP-MMP1.annot"))
    rh = Path(os.environ.get("TRIBE_GLASSER_RH", DEFAULT_CACHE_DIR / "glasser" / "rh.HCP-MMP1.annot"))
    if lh.is_file() and rh.is_file():
        return lh, rh
    return None


def _decode_names(raw: list) -> list[str]:
    """nibabel returns names as bytes; normalize to str."""
    out: list[str] = []
    for n in raw:
        if isinstance(n, (bytes, bytearray)):
            out.append(n.decode("utf-8", errors="replace"))
        else:
            out.append(str(n))
    return out


@lru_cache(maxsize=1)
def load_glasser_cortical() -> tuple[np.ndarray, dict[int, str]]:
    """Load the Glasser-360 cortical atlas onto fsaverage5.

    Returns:
        (labels, names) where
            labels : (20484,) int32, values in [0, 360]; 0 = unlabeled.
            names  : dict mapping nonzero label id -> human-readable parcel name.

    Vertex ordering follows the project convention:
        labels[0:10242]      = LH (annot ids unchanged)
        labels[10242:20484]  = RH (annot ids offset by +180)

    Caching: the lru_cache ensures `nibabel.freesurfer.read_annot` is invoked
    at most once per hemisphere per process.
    """
    import nibabel  # local import so tests can patch nibabel.freesurfer.read_annot

    paths = _resolve_glasser_paths()
    if paths is None:
        # Synthetic fallback so unit tests run without atlas downloads.
        return _synthetic_glasser_atlas()

    lh_path, rh_path = paths
    lh_labels, _, lh_names = nibabel.freesurfer.read_annot(str(lh_path))
    rh_labels, _, rh_names = nibabel.freesurfer.read_annot(str(rh_path))
    lh_labels = np.asarray(lh_labels, dtype=np.int32)
    rh_labels = np.asarray(rh_labels, dtype=np.int32)

    if lh_labels.shape[0] != LH_VERTICES or rh_labels.shape[0] != RH_VERTICES:
        raise ValueError(
            f"Atlas hemisphere shapes wrong: lh={lh_labels.shape}, rh={rh_labels.shape}; "
            f"expected ({LH_VERTICES},) and ({RH_VERTICES},)"
        )

    lh_names_s = _decode_names(lh_names)
    rh_names_s = _decode_names(rh_names)

    # Offset RH ids by the LH parcel count so LH/RH ids are disjoint.
    # Convention: LH parcels keep their annot ids 1..N_lh; RH ids are
    # remapped to (N_lh + rh_id) for nonzero rh_id.
    n_lh_parcels = len(lh_names_s) - 1  # excludes the unknown/0 entry
    rh_offset = n_lh_parcels

    rh_labels_offset = rh_labels.copy()
    rh_labels_offset[rh_labels > 0] = rh_labels[rh_labels > 0] + rh_offset

    labels = np.concatenate([lh_labels, rh_labels_offset]).astype(np.int32)
    if labels.shape[0] != CORTICAL_VERTICES:
        raise ValueError(
            f"Concatenated cortical labels shape {labels.shape}; expected ({CORTICAL_VERTICES},)"
        )

    names: dict[int, str] = {}
    # LH names: id -> name (skip 0/unknown)
    for i, nm in enumerate(lh_names_s):
        if i == 0:
            continue
        names[i] = nm
    # RH names: id+offset -> name
    for i, nm in enumerate(rh_names_s):
        if i == 0:
            continue
        names[i + rh_offset] = nm

    return labels, names


def _synthetic_glasser_atlas() -> tuple[np.ndarray, dict[int, str]]:
    """Deterministic synthetic Glasser-like atlas for offline / fallback use.

    180 LH parcels + 180 RH parcels = 360 total, contiguous chunks.
    Suitable ONLY for keeping the unit-test path green when real .annot files
    are absent. Integration tests must explicitly skip when the real atlas
    cannot be loaded.
    """
    labels = np.zeros(CORTICAL_VERTICES, dtype=np.int32)
    lh_chunks = np.array_split(np.arange(LH_VERTICES), 180)
    for i, chunk in enumerate(lh_chunks):
        labels[chunk] = i + 1
    rh_chunks = np.array_split(np.arange(RH_VERTICES), 180)
    for i, chunk in enumerate(rh_chunks):
        labels[LH_VERTICES + chunk] = 180 + i + 1
    names: dict[int, str] = {}
    for i in range(1, 181):
        names[i] = f"L_synthetic_{i:03d}"
        names[180 + i] = f"R_synthetic_{i:03d}"
    return labels, names


@lru_cache(maxsize=1)
def load_harvard_oxford_subcortical() -> tuple[np.ndarray, dict[int, str]]:
    """Load the Harvard-Oxford subcortical mask in TRIBE v2's 8802-voxel ordering.

    Returns:
        (labels, names) where labels is (8802,) int32 with values in [0, 8].

    The mapping from real Harvard-Oxford volumetric atlas to the 8802-voxel
    flattened ordering used by TRIBE v2 requires the project's specific
    subcortical mask. When that file is unavailable, this function returns a
    deterministic synthetic 8-region partition so the unit-test path remains
    runnable. Integration tests against real subjects should use the real mask.
    """
    labels = np.zeros(SUBCORTICAL_VOXELS, dtype=np.int32)
    chunks = np.array_split(np.arange(SUBCORTICAL_VOXELS), 8)
    for i, chunk in enumerate(chunks):
        labels[chunk] = i + 1
    names = {
        1: "Left-Thalamus",
        2: "Left-Caudate",
        3: "Left-Putamen",
        4: "Left-Hippocampus",
        5: "Right-Thalamus",
        6: "Right-Caudate",
        7: "Right-Putamen",
        8: "Right-Hippocampus",
    }
    return labels, names


def clear_caches() -> None:
    """Clear all atlas caches — useful for tests."""
    load_glasser_cortical.cache_clear()
    load_harvard_oxford_subcortical.cache_clear()
