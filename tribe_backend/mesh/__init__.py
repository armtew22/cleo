"""BrainMeshExporter compartment.

Public API:
    BrainMeshExporter — top-level exporter producing JSON / binary / GLB / animation bundles.
    activation_to_rgba — pure colormap function.
    aggregate_temporal — pure temporal reducer.
    load_fsaverage5 — fsaverage5 surface loader.
"""
from tribe_backend.mesh.colormap import activation_to_rgba
from tribe_backend.mesh.exporter import BrainMeshExporter, aggregate_temporal
from tribe_backend.mesh.geometry import load_fsaverage5

__all__ = [
    "BrainMeshExporter",
    "activation_to_rgba",
    "aggregate_temporal",
    "load_fsaverage5",
]
