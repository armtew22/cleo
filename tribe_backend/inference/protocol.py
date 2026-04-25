"""Inference protocol — single source of truth lives in contracts.py.

This module re-exports :class:`TribeInference` from :mod:`tribe_backend.contracts`
so consumers in the inference compartment can import it from a local namespace
without bypassing the contract. It also defines :class:`InferenceFailure`, the
exception adapters raise to signal a recoverable per-window failure (e.g. CUDA
OOM, non-finite outputs) so that ``ControlUnit``'s failure-isolation policy can
keep the loop alive.
"""
from __future__ import annotations

from tribe_backend.contracts import TribeInference

__all__ = ["TribeInference", "InferenceFailure"]


class InferenceFailure(Exception):
    """Raised by a TribeInference adapter when a single window fails.

    The control loop catches this, logs the offending ``window_id``, and
    proceeds with the next window. Use it for transient/per-window failures
    (CUDA OOM mid-window, non-finite outputs, padding-policy violations).
    Permanent setup errors (missing weights, wrong device) should raise at
    ``__init__`` time with a more specific exception, not this one.
    """
