"""ControlUnit compartment — orchestrator (top of backend).

Public API:
    ControlUnit, Bundle, ArtifactSink, DiskArtifactSink,
    MockLivePoller, NoopPoller, FilePoller, WindowBuffer.

The DI default factory (wiring real GlasserParcellationUnit + BrainMeshExporter
+ GpuTribeInference) is intentionally NOT defined here yet — it lands in a
follow-up commit after Phases 1, 2, and 4 merge to main.
"""
from __future__ import annotations

from tribe_backend.control.dispatcher import ArtifactSink, Bundle, DiskArtifactSink
from tribe_backend.control.poller import FilePoller, MockLivePoller, NoopPoller
from tribe_backend.control.unit import ControlUnit, MeshLike, ParcellationLike
from tribe_backend.control.window_buffer import WindowBuffer

__all__ = [
    "ArtifactSink",
    "Bundle",
    "ControlUnit",
    "DiskArtifactSink",
    "FilePoller",
    "MeshLike",
    "MockLivePoller",
    "NoopPoller",
    "ParcellationLike",
    "WindowBuffer",
]
