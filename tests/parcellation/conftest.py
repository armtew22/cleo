"""Parcellation-specific fixtures.

Provides a synthetic atlas labeling that aligns with the planted hot spots
in the top-level conftest.make_synthetic factory:
  cortical [0:100]      -> "Amygdala"
  cortical [5000:5200]  -> "FFC"
  cortical [10242:10500] -> "V1"

The synthetic atlas pads the rest of the cortical surface with deterministic
filler parcels of equal size so tests can rely on exactly 360 cortical
parcel names — matching the real Glasser-360 atlas count.
"""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    SUBCORTICAL_VOXELS,
)


@pytest.fixture
def synthetic_cortical_labels() -> np.ndarray:
    """A (20484,) int array partitioning the cortex into 360 parcels.

    Parcel ids are 1-based (0 reserved for "unlabeled" / medial wall).
    Reserved ids:
      1 -> Amygdala  (vertices 0:100)
      2 -> FFC       (vertices 5000:5200)
      3 -> V1        (vertices 10242:10500)
    All other vertices are partitioned into 357 contiguous filler parcels
    (ids 4..360) of approximately equal size.
    """
    labels = np.zeros(CORTICAL_VERTICES, dtype=np.int32)
    labels[0:100] = 1  # Amygdala
    labels[5000:5200] = 2  # FFC
    labels[10242:10500] = 3  # V1

    # Fill remaining vertices with parcels 4..360 in contiguous chunks.
    remaining = np.where(labels == 0)[0]
    n_remaining_parcels = 360 - 3
    chunks = np.array_split(remaining, n_remaining_parcels)
    for i, chunk in enumerate(chunks):
        labels[chunk] = 4 + i
    return labels


@pytest.fixture
def synthetic_cortical_names() -> dict[int, str]:
    names = {1: "Amygdala", 2: "FFC", 3: "V1"}
    for i in range(4, 361):
        names[i] = f"P{i:03d}"
    return names


@pytest.fixture
def synthetic_subcortical_labels() -> np.ndarray:
    """A (8802,) int array partitioning the subcortical mask into 8 parcels."""
    labels = np.zeros(SUBCORTICAL_VOXELS, dtype=np.int32)
    chunks = np.array_split(np.arange(SUBCORTICAL_VOXELS), 8)
    for i, chunk in enumerate(chunks):
        labels[chunk] = i + 1
    return labels


@pytest.fixture
def synthetic_subcortical_names() -> dict[int, str]:
    return {
        1: "Left-Thalamus",
        2: "Left-Caudate",
        3: "Left-Putamen",
        4: "Left-Hippocampus",
        5: "Right-Thalamus",
        6: "Right-Caudate",
        7: "Right-Putamen",
        8: "Right-Hippocampus",
    }
