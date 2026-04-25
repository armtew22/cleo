"""Phase 4 — FakeTribeInference: returns a fixture, stamps window_id."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_stimulus_window, make_synthetic
from tribe_backend.contracts import StimulusWindow, TribeInference, TribeOutput
from tribe_backend.inference.stub import FakeTribeInference


pytestmark = pytest.mark.unit


def test_stub_returns_fixture_cortical_and_subcortical() -> None:
    fixture = make_synthetic(window_id="orig")
    stub = FakeTribeInference(fixture)
    w = make_stimulus_window(window_id="W-001")
    out = stub(w)
    assert isinstance(out, TribeOutput)
    np.testing.assert_array_equal(out.cortical, fixture.cortical)
    np.testing.assert_array_equal(out.subcortical, fixture.subcortical)


def test_stub_stamps_window_id_from_stimulus_window() -> None:
    fixture = make_synthetic(window_id="ignored-original")
    stub = FakeTribeInference(fixture)
    w = make_stimulus_window(window_id="W-stamp-42")
    out = stub(w)
    assert out.window_id == "W-stamp-42"


def test_stub_does_not_mutate_input_fixture() -> None:
    fixture = make_synthetic(window_id="orig")
    original_id = fixture.window_id
    stub = FakeTribeInference(fixture)
    w = make_stimulus_window(window_id="other")
    stub(w)
    # frozen dataclass guarantees this; double-check explicitly.
    assert fixture.window_id == original_id


def test_stub_satisfies_tribeinference_protocol() -> None:
    fixture = make_synthetic()
    stub = FakeTribeInference(fixture)
    assert isinstance(stub, TribeInference)


def test_stub_returns_independent_outputs_per_call() -> None:
    fixture = make_synthetic(window_id="orig")
    stub = FakeTribeInference(fixture)
    w1 = make_stimulus_window(window_id="A")
    w2 = make_stimulus_window(window_id="B")
    out1 = stub(w1)
    out2 = stub(w2)
    assert out1.window_id == "A"
    assert out2.window_id == "B"


def test_stub_preserves_metadata_fields() -> None:
    fixture = make_synthetic()
    stub = FakeTribeInference(fixture)
    out = stub(make_stimulus_window(window_id="X"))
    assert out.fps == fixture.fps
    assert out.surface == fixture.surface
    assert out.subcortical_atlas == fixture.subcortical_atlas
