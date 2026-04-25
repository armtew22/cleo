"""Inference compartment — GPU model host.

Public API:
    TribeInference        — Protocol (re-exported from contracts)
    InferenceFailure      — exception raised by adapters on failure
    FakeTribeInference    — test double that returns a fixture
    GpuTribeInference     — Phase 5 adapter wrapping upstream tribev2 (lazy import)
    WeightsNotFoundError  — raised at GpuTribeInference.__init__ on load failure
"""
from tribe_backend.inference.protocol import InferenceFailure, TribeInference

__all__ = [
    "TribeInference",
    "InferenceFailure",
    "FakeTribeInference",
    "GpuTribeInference",
    "WeightsNotFoundError",
]


def __getattr__(name: str):
    # Lazy re-export so importing the package doesn't pull in torch/tribev2.
    if name == "FakeTribeInference":
        from tribe_backend.inference.stub import FakeTribeInference
        return FakeTribeInference
    if name in ("GpuTribeInference", "WeightsNotFoundError"):
        from tribe_backend.inference.gpu_runner import (
            GpuTribeInference,
            WeightsNotFoundError,
        )
        return {"GpuTribeInference": GpuTribeInference, "WeightsNotFoundError": WeightsNotFoundError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
