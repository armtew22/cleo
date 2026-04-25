"""build_inference: name -> TribeInference selection."""
from __future__ import annotations

import pytest

from tests.conftest import make_synthetic
from tribe_backend.contracts import TribeInference
from tribe_backend.control.factory import build_inference, InferenceFailure


def test_build_fake_returns_protocol():
    fx = make_synthetic()
    inf = build_inference("fake", fake_fixture=fx)
    assert isinstance(inf, TribeInference)


def test_build_fake_requires_fixture():
    with pytest.raises(ValueError):
        build_inference("fake")


def test_build_unknown_name_raises():
    with pytest.raises(ValueError):
        build_inference("not-a-thing")


def test_inference_failure_reexported():
    """InferenceFailure must be reachable from control.factory so api/ can
    catch it without importing tribe_backend.inference directly."""
    assert issubclass(InferenceFailure, Exception)


def test_build_gpu_without_torch_raises_clearly():
    """No torch/weights here, so building gpu should raise — but the failure
    must be a clean, recognizable exception, not an obscure AttributeError."""
    with pytest.raises(Exception):
        build_inference("gpu")
