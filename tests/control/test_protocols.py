"""Tests for Bundle, ArtifactSink, and DiskArtifactSink."""
from __future__ import annotations

from pathlib import Path

import pytest

from tribe_backend.contracts import StimulusWindow
from tribe_backend.control import (
    ArtifactSink,
    Bundle,
    DiskArtifactSink,
)
from tests.conftest import make_stimulus_window
from tests.control.fakes import FakeReport


pytestmark = pytest.mark.unit


def test_bundle_holds_window_id_report_and_mesh_dir(tmp_path: Path) -> None:
    bundle = Bundle(
        window_id="w-1",
        report=FakeReport(window_id="w-1", text="hi"),
        mesh_dir=tmp_path,
    )
    assert bundle.window_id == "w-1"
    assert bundle.report.window_id == "w-1"
    assert bundle.mesh_dir == tmp_path


def test_disk_artifact_sink_returns_per_window_dir(tmp_path: Path) -> None:
    sink = DiskArtifactSink(tmp_path)
    window = make_stimulus_window(window_id="w-42")
    p = sink.path_for(window)
    assert p == tmp_path / "w-42"
    assert p.exists() and p.is_dir()


def test_disk_artifact_sink_satisfies_protocol(tmp_path: Path) -> None:
    sink: ArtifactSink = DiskArtifactSink(tmp_path)  # structural check via annotation
    window = make_stimulus_window(window_id="abc")
    assert isinstance(sink.path_for(window), Path)


def test_disk_artifact_sink_idempotent(tmp_path: Path) -> None:
    sink = DiskArtifactSink(tmp_path)
    w = make_stimulus_window(window_id="repeat")
    p1 = sink.path_for(w)
    p2 = sink.path_for(w)
    assert p1 == p2
    assert p1.exists()
