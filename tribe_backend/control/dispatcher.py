"""Bundle dataclass + ArtifactSink protocol/impl.

ArtifactSink decouples ControlUnit from on-disk layout — tests use an in-memory
or tmp_path-backed sink, production uses DiskArtifactSink rooted at a configured
output directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tribe_backend.contracts import StimulusWindow


@dataclass(frozen=True)
class Bundle:
    """The artifact a single 30s window produces.

    `report` is whatever the parcellation unit returned (kept loosely typed so
    this compartment doesn't pull in parcellation/). `mesh_dir` is the directory
    the mesh exporter wrote into, or None if mesh export failed.
    """

    window_id: str
    report: Any
    mesh_dir: Path | None


@runtime_checkable
class ArtifactSink(Protocol):
    """Resolves a per-window output directory."""

    def path_for(self, window: StimulusWindow) -> Path: ...


class DiskArtifactSink:
    """Concrete sink: `<root>/<window_id>/`."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, window: StimulusWindow) -> Path:
        p = self._root / window.window_id
        p.mkdir(parents=True, exist_ok=True)
        return p
