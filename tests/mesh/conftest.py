"""Shared fixtures for mesh tests — keep heavy fsaverage5 loads warm."""
from __future__ import annotations

import pytest

from tests.conftest import make_synthetic
from tribe_backend.mesh.exporter import BrainMeshExporter


@pytest.fixture(scope="module")
def exporter() -> BrainMeshExporter:
    return BrainMeshExporter()


@pytest.fixture(scope="module")
def small_output():
    # T=4 keeps animation tests quick.
    return make_synthetic(T=4, seed=7)
