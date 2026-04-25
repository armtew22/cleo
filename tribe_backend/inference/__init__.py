"""Inference compartment — GPU model host.

Public API:
    TribeInference   — Protocol (re-exported from contracts)
    InferenceFailure — exception raised by adapters on failure
    FakeTribeInference — test double that returns a fixture

The real GPU adapter (GpuTribeInference) lands in Phase 5.
"""
from tribe_backend.inference.protocol import InferenceFailure, TribeInference

__all__ = ["TribeInference", "InferenceFailure", "FakeTribeInference"]


def __getattr__(name: str):
    # Lazy re-export so importing the package doesn't pull in stub deps.
    if name == "FakeTribeInference":
        from tribe_backend.inference.stub import FakeTribeInference
        return FakeTribeInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
