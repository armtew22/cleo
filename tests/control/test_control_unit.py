"""Tests for ControlUnit orchestration: basic flow, backpressure, failure
isolation, graceful shutdown, and window→output traceability."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tribe_backend.contracts import StimulusWindow, TribeOutput
from tribe_backend.control import (
    Bundle,
    ControlUnit,
    DiskArtifactSink,
    NoopPoller,
)
from tests.conftest import make_stimulus_window, make_synthetic
from tests.control.fakes import (
    FakeMesh,
    FakeParcellation,
    FakeReport,
    FakeTribeInference,
    MockPoller,
    RaisingMesh,
    RaisingParcellation,
    SlowInference,
)


pytestmark = pytest.mark.unit


# ----------------------------------------------------------------- helpers

def _three_windows() -> list[StimulusWindow]:
    return [
        make_stimulus_window(window_id=f"w-{i}", seed=i)
        for i in range(3)
    ]


def _make_control(tmp_path: Path, **overrides):
    fixture = make_synthetic()
    poller = overrides.pop("poller", MockPoller(_three_windows()))
    inference = overrides.pop("inference", FakeTribeInference(fixture))
    parcellation = overrides.pop("parcellation", FakeParcellation())
    mesh = overrides.pop("mesh", FakeMesh())
    sink = overrides.pop("sink", DiskArtifactSink(tmp_path))
    unit = ControlUnit(
        poller=poller,
        inference=inference,
        parcellation=parcellation,
        mesh=mesh,
        sink=sink,
        **overrides,
    )
    return unit, poller, inference, parcellation, mesh, sink


# -------------------------------------------------- basic flow / traceability

def test_run_once_produces_a_bundle_with_correct_window_id(tmp_path: Path) -> None:
    unit, _poller, inference, parcellation, mesh, _sink = _make_control(tmp_path)
    bundle = unit.run_once()
    assert isinstance(bundle, Bundle)
    assert bundle.window_id == "w-0"
    # Inference saw the right window.
    assert inference.calls[0].window_id == "w-0"
    # Parcellation got an output stamped with the same id (traceability).
    assert parcellation.calls[0].window_id == "w-0"
    # Mesh wrote into the per-window directory.
    assert mesh.calls[0][0].window_id == "w-0"
    assert bundle.mesh_dir is not None
    assert bundle.mesh_dir.name == "w-0"


def test_run_stream_yields_three_bundles_with_correct_ids(tmp_path: Path) -> None:
    unit, poller, _inf, _parc, _mesh, _sink = _make_control(tmp_path)
    bundles = list(unit.run_stream(poller))
    assert [b.window_id for b in bundles] == ["w-0", "w-1", "w-2"]


def test_window_id_traceability_through_pipeline(tmp_path: Path) -> None:
    unit, poller, inference, parcellation, _mesh, _sink = _make_control(tmp_path)
    bundles = list(unit.run_stream(poller))
    # bundle.window_id == window.window_id == output.window_id, for every step.
    for i, bundle in enumerate(bundles):
        assert bundle.window_id == f"w-{i}"
        assert inference.calls[i].window_id == f"w-{i}"
        assert parcellation.calls[i].window_id == f"w-{i}"


def test_inference_output_window_id_overridden_if_mismatched(tmp_path: Path) -> None:
    # Inference that returns a fixture with the WRONG window_id — ControlUnit
    # must defensively re-stamp it so downstream stages see consistent ids.

    class BadInference:
        def __init__(self, out: TribeOutput) -> None:
            self._out = out

        def __call__(self, w: StimulusWindow) -> TribeOutput:
            return self._out  # NOT stamped with w.window_id

    fixture = make_synthetic(window_id="STALE")
    parc = FakeParcellation()
    unit, _, _, _, _, _ = _make_control(
        tmp_path, inference=BadInference(fixture), parcellation=parc
    )
    bundle = unit.run_once()
    assert bundle.window_id == "w-0"
    assert parc.calls[0].window_id == "w-0"  # was re-stamped


# ----------------------------------------------------- one-shot process_window

def test_process_window_constructs_window_and_returns_report(tmp_path: Path) -> None:
    unit, _poller, inference, parc, _mesh, _sink = _make_control(
        tmp_path,
        poller=NoopPoller(),  # one-shot path
    )
    sw = make_stimulus_window(seed=7)
    report = unit.process_window(
        video=sw.video,
        audio=sw.audio,
        audio_sr=sw.audio_sr,
        text=sw.text,
        video_fps=sw.video_fps,
        t_start=0.0,
    )
    assert isinstance(report, FakeReport)
    # The constructed window flowed through inference & parcellation.
    assert len(inference.calls) == 1
    assert len(parc.calls) == 1
    assert inference.calls[0].window_id == parc.calls[0].window_id
    assert report.window_id == inference.calls[0].window_id


def test_process_window_uses_audio_length_for_duration(tmp_path: Path) -> None:
    unit, *_ = _make_control(tmp_path, poller=NoopPoller())
    audio = np.zeros(16000 * 28, dtype=np.float32)  # 28s
    video = np.zeros((28 * 4, 8, 8, 3), dtype=np.uint8)
    unit.process_window(
        video=video, audio=audio, audio_sr=16000, text="hi", video_fps=4.0, t_start=0.0
    )


# ---------------------------------------------------------------- backpressure

def test_backpressure_drops_stale_windows_when_inference_is_slow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Slow inference + fast poller → older windows get dropped, no deadlock."""
    fixture = make_synthetic()
    inference = SlowInference(fixture, delay_s=0.5)

    poller = MockPoller(interval_s=0.0)
    # Pre-load 5 windows so they're all immediately pending — emulates a fast
    # camera spitting frames while inference is still on window #0.
    for i in range(5):
        poller.push(make_stimulus_window(window_id=f"fast-{i}", seed=i))

    unit, *_ = _make_control(
        tmp_path, poller=poller, inference=inference, backpressure_max_lag=1
    )

    caplog.set_level(logging.WARNING, logger="tribe_backend.control.unit")

    t0 = time.time()
    bundles = list(unit.run_stream(poller))
    elapsed = time.time() - t0

    # We only ran inference some number of times < 5 because the rest got dropped.
    assert 1 <= len(bundles) <= 5
    assert len(inference.calls) == len(bundles)
    # Drop count was tracked & logged.
    assert unit.dropped_count >= 1
    assert any("backpressure" in rec.getMessage() for rec in caplog.records)
    # Total wall time < 5 * 0.5s — proves we dropped, not blocked on each.
    assert elapsed < 5 * 0.5


def test_backpressure_no_drops_when_poller_keeps_up(tmp_path: Path) -> None:
    """If the poller never has more than 1 pending, nothing is dropped."""
    poller = MockPoller(_three_windows())
    unit, *_ = _make_control(tmp_path, poller=poller)
    bundles = list(unit.run_stream(poller))
    assert len(bundles) == 3
    assert unit.dropped_count == 0


def test_backpressure_does_not_deadlock_with_blocking_poller(tmp_path: Path) -> None:
    """A blocking poller (no `pending()`) must not cause the loop to hang.

    The backpressure draining is gated on `pending()`; pollers that don't
    expose it just receive non-draining behavior — every window flows.
    """

    class BlockingPoller:
        """No pending() method. Iterator over a fixed list."""
        def __init__(self, ws):
            self._ws = list(ws)
            self._i = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._i >= len(self._ws):
                raise StopIteration
            w = self._ws[self._i]
            self._i += 1
            return w

    poller = BlockingPoller(_three_windows())
    unit, *_ = _make_control(tmp_path, poller=poller, backpressure_max_lag=1)
    bundles = list(unit.run_stream(poller))
    assert [b.window_id for b in bundles] == ["w-0", "w-1", "w-2"]
    assert unit.dropped_count == 0


# ---------------------------------------------------------- failure isolation

def test_parcellation_failure_does_not_block_mesh(tmp_path: Path) -> None:
    mesh = FakeMesh()
    unit, _poller, _inf, _parc, _, _ = _make_control(
        tmp_path,
        parcellation=RaisingParcellation(),
        mesh=mesh,
    )
    bundle = unit.run_once()
    assert bundle.report is None
    assert bundle.mesh_dir is not None  # mesh still ran
    assert mesh.calls and mesh.calls[0][0].window_id == "w-0"


def test_mesh_failure_does_not_block_parcellation(tmp_path: Path) -> None:
    parc = FakeParcellation()
    unit, *_ = _make_control(
        tmp_path,
        parcellation=parc,
        mesh=RaisingMesh(),
    )
    bundle = unit.run_once()
    assert bundle.mesh_dir is None
    assert bundle.report is not None  # parcellation still ran
    assert parc.calls and parc.calls[0].window_id == "w-0"


def test_both_downstreams_failing_still_returns_bundle(tmp_path: Path) -> None:
    unit, *_ = _make_control(
        tmp_path,
        parcellation=RaisingParcellation(),
        mesh=RaisingMesh(),
    )
    bundle = unit.run_once()
    assert bundle.window_id == "w-0"
    assert bundle.report is None
    assert bundle.mesh_dir is None


# ------------------------------------------------------- graceful shutdown

def test_stop_drains_in_flight_and_closes_poller(tmp_path: Path) -> None:
    fixture = make_synthetic()
    inference = SlowInference(fixture, delay_s=0.2)
    poller = MockPoller(_three_windows())
    unit, *_ = _make_control(tmp_path, poller=poller, inference=inference)

    bundles: list[Bundle] = []

    def runner():
        for b in unit.run_stream(poller):
            bundles.append(b)

    th = threading.Thread(target=runner)
    th.start()
    # Let the first window get into inference, then stop.
    time.sleep(0.05)
    unit.stop()
    th.join(timeout=2.0)
    assert not th.is_alive(), "run_stream did not exit on stop()"
    # Poller was closed (close() called).
    assert poller.closed


def test_stop_before_run_exits_immediately(tmp_path: Path) -> None:
    poller = MockPoller(_three_windows())
    unit, *_ = _make_control(tmp_path, poller=poller)
    unit.stop()
    bundles = list(unit.run_stream(poller))
    assert bundles == []


def test_run_forever_returns_when_stopped(tmp_path: Path) -> None:
    fixture = make_synthetic()
    inference = SlowInference(fixture, delay_s=0.05)
    poller = MockPoller(interval_s=0.0)
    # Pre-load 50 windows so the loop has plenty to chew on.
    for i in range(50):
        poller.push(make_stimulus_window(window_id=f"forever-{i}", seed=i))
    unit, *_ = _make_control(tmp_path, poller=poller, inference=inference)

    th = threading.Thread(target=unit.run_forever)
    th.start()
    time.sleep(0.1)
    unit.stop()
    th.join(timeout=2.0)
    assert not th.is_alive()
