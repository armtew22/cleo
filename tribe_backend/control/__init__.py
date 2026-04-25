"""ControlUnit compartment — orchestrator (top of backend).

Public API:
    ControlUnit, Bundle, ArtifactSink, DiskArtifactSink,
    MockLivePoller, NoopPoller, FilePoller, WindowBuffer,
    default_control_unit (composition root — wires the real compartments).
"""
from __future__ import annotations

from tribe_backend.control.dispatcher import ArtifactSink, Bundle, DiskArtifactSink
from tribe_backend.control.factory import default_control_unit
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
    "default_control_unit",
]
