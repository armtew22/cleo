"""Phase 4 — protocol.py: re-exports TribeInference and adds InferenceFailure."""
from __future__ import annotations

import pytest

from tribe_backend.contracts import TribeInference as ContractsTribeInference
from tribe_backend.inference.protocol import InferenceFailure, TribeInference


pytestmark = pytest.mark.unit


def test_protocol_reexports_contracts_tribeinference() -> None:
    # Single source of truth: the protocol module re-exports the same object.
    assert TribeInference is ContractsTribeInference


def test_inference_failure_is_exception_subclass() -> None:
    assert issubclass(InferenceFailure, Exception)


def test_inference_failure_carries_message() -> None:
    exc = InferenceFailure("CUDA OOM on window-42")
    assert "CUDA OOM" in str(exc)


def test_inference_failure_can_be_raised_and_caught() -> None:
    with pytest.raises(InferenceFailure):
        raise InferenceFailure("boom")
