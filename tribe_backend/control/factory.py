"""Composition root: the ONLY place all four compartments are imported together.

`default_control_unit` wires the real GlasserParcellationUnit + BrainMeshExporter
plus an inference implementation (default: FakeTribeInference for tests/dev;
swap to GpuTribeInference for real GPU). This is the single line that changes
when Phase 5 lands a real GPU adapter.
"""
from __future__ import annotations

from pathlib import Path

from tribe_backend.contracts import TribeInference, TribeOutput
from tribe_backend.control.dispatcher import DiskArtifactSink
from tribe_backend.control.poller import NoopPoller
from tribe_backend.control.unit import ControlUnit
from tribe_backend.inference import FakeTribeInference
from tribe_backend.inference.protocol import InferenceFailure
from tribe_backend.mesh import BrainMeshExporter
from tribe_backend.parcellation import GlasserParcellationUnit

__all__ = [
    "default_control_unit",
    "build_inference",
    "InferenceFailure",
]


def build_inference(
    name: str,
    *,
    fake_fixture: TribeOutput | None = None,
) -> TribeInference:
    """Pick a TribeInference implementation by name.

    Exists so the API layer can select fake-vs-gpu without itself importing
    `tribe_backend.inference` (preserves the api -> inference import-linter rule).

    Parameters
    ----------
    name
        "fake" or "gpu".
    fake_fixture
        Required when ``name == "fake"``; the deterministic TribeOutput the
        FakeTribeInference returns for every window.
    """
    n = name.strip().lower()
    if n == "fake":
        if fake_fixture is None:
            raise ValueError("build_inference('fake') requires fake_fixture")
        return FakeTribeInference(fake_fixture)
    if n == "gpu":
        # Lazy import — keeps torch out of `import tribe_backend.control.factory`
        # for non-GPU consumers (tests, the fake API path).
        from tribe_backend.inference import GpuTribeInference  # type: ignore[attr-defined]
        return GpuTribeInference()
    raise ValueError(f"unknown inference name {name!r}; expected 'fake' or 'gpu'")


def default_control_unit(
    *,
    out_dir: str | Path,
    inference: TribeInference | None = None,
    fake_fixture: TribeOutput | None = None,
    parcellation: GlasserParcellationUnit | None = None,
    mesh: BrainMeshExporter | None = None,
    backpressure_max_lag: int | None = None,
) -> ControlUnit:
    """Wire a ControlUnit with real parcellation + mesh + an inference impl.

    Parameters
    ----------
    out_dir
        Directory where per-window artifacts are written.
    inference
        Concrete TribeInference. If None, a FakeTribeInference backed by
        `fake_fixture` is constructed (one of the two must be provided).
    fake_fixture
        TribeOutput used by the auto-constructed FakeTribeInference. Ignored
        when `inference` is supplied.
    parcellation, mesh
        Optional pre-built compartments — primarily for tests that want to
        inject a custom atlas or surface. Defaults construct fresh instances.
    backpressure_max_lag
        Forwarded to ControlUnit. Defaults to None (off) for one-shot users.
    """
    if inference is None:
        if fake_fixture is None:
            raise ValueError(
                "default_control_unit requires either `inference` or "
                "`fake_fixture` (to build a FakeTribeInference)"
            )
        inference = FakeTribeInference(fake_fixture)

    parcellation = parcellation or GlasserParcellationUnit()
    mesh = mesh or BrainMeshExporter()
    sink = DiskArtifactSink(out_dir)

    return ControlUnit(
        poller=NoopPoller(),
        inference=inference,
        parcellation=parcellation,
        mesh=mesh,
        sink=sink,
        backpressure_max_lag=backpressure_max_lag,
    )
