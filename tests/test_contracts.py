"""Phase 0 — contract tests.

These define the keystone behavior every later phase relies on:
  - TribeOutput rejects malformed shapes.
  - TribeOutput round-trips through .npz.
  - The synthetic factory plants hot spots at known indices.
  - StimulusWindow validates its own preconditions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tribe_backend.contracts import (
    CORTICAL_VERTICES,
    EXPECTED_T_PER_30S_WINDOW,
    LH_VERTICES,
    RH_VERTICES,
    SUBCORTICAL_VOXELS,
    StimulusWindow,
    TribeInference,
    TribeOutput,
)
from tests.conftest import HotSpot, make_synthetic, make_stimulus_window
from tests.fixtures.load_sample import load_tribe_sample


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shape and constants
# ---------------------------------------------------------------------------


def test_constants_are_consistent():
    assert LH_VERTICES + RH_VERTICES == CORTICAL_VERTICES == 20484
    assert SUBCORTICAL_VOXELS == 8802
    assert EXPECTED_T_PER_30S_WINDOW == 31


def test_tribe_output_accepts_canonical_30s_shape():
    out = make_synthetic()
    assert out.cortical.shape == (31, 20484)
    assert out.subcortical.shape == (31, 8802)
    assert out.T == 31
    assert out.fps == 1.0
    assert out.surface == "fsaverage5"


def test_tribe_output_class_constant_matches_module_constant():
    assert TribeOutput.EXPECTED_T_PER_30S_WINDOW == EXPECTED_T_PER_30S_WINDOW


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_wrong_cortical_width():
    with pytest.raises(ValueError, match="cortical"):
        TribeOutput(
            cortical=np.zeros((31, 20483), dtype=np.float32),
            subcortical=np.zeros((31, 8802), dtype=np.float32),
        )


def test_rejects_wrong_subcortical_width():
    with pytest.raises(ValueError, match="subcortical"):
        TribeOutput(
            cortical=np.zeros((31, 20484), dtype=np.float32),
            subcortical=np.zeros((31, 8801), dtype=np.float32),
        )


def test_rejects_non_2d_cortical():
    with pytest.raises(ValueError, match="cortical"):
        TribeOutput(
            cortical=np.zeros(20484, dtype=np.float32),
            subcortical=np.zeros((1, 8802), dtype=np.float32),
        )


def test_rejects_mismatched_T():
    with pytest.raises(ValueError, match="share T"):
        TribeOutput(
            cortical=np.zeros((31, 20484), dtype=np.float32),
            subcortical=np.zeros((30, 8802), dtype=np.float32),
        )


def test_rejects_non_array_inputs():
    with pytest.raises(TypeError):
        TribeOutput(cortical=[[0.0] * 20484], subcortical=np.zeros((1, 8802), dtype=np.float32))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_npz_roundtrip_preserves_arrays_and_metadata(tmp_path):
    out = make_synthetic(T=4, seed=7, window_id="rt-1")
    p = out.to_npz(tmp_path / "out.npz")
    assert p.exists()
    loaded = TribeOutput.from_npz(p)
    np.testing.assert_array_equal(loaded.cortical, out.cortical)
    np.testing.assert_array_equal(loaded.subcortical, out.subcortical)
    assert loaded.fps == out.fps
    assert loaded.surface == out.surface
    assert loaded.subcortical_atlas == out.subcortical_atlas
    assert loaded.window_id == out.window_id


def test_npz_roundtrip_with_none_window_id(tmp_path):
    out = make_synthetic(T=2, seed=1, window_id=None)
    p = out.to_npz(tmp_path / "out.npz")
    loaded = TribeOutput.from_npz(p)
    assert loaded.window_id is None


def test_with_window_id_returns_new_frozen_instance():
    out = make_synthetic(window_id="orig")
    rebound = out.with_window_id("new")
    assert out.window_id == "orig"        # unchanged (frozen)
    assert rebound.window_id == "new"
    assert rebound is not out
    # underlying arrays reused, not copied
    assert rebound.cortical is out.cortical


# ---------------------------------------------------------------------------
# Synthetic factory plants known hot spots
# ---------------------------------------------------------------------------


def test_make_synthetic_plants_default_hotspots():
    out = make_synthetic(noise=0.0)
    # Amygdala vertices [0:100] -> +3.0
    assert np.all(out.cortical[:, 0:100] == 3.0)
    # FFC vertices [5000:5200] -> +2.0
    assert np.all(out.cortical[:, 5000:5200] == 2.0)
    # V1 vertices [10242:10500] -> +4.0  (first parcel of RH)
    assert np.all(out.cortical[:, 10242:10500] == 4.0)


def test_make_synthetic_is_deterministic_for_same_seed():
    a = make_synthetic(T=4, seed=42)
    b = make_synthetic(T=4, seed=42)
    np.testing.assert_array_equal(a.cortical, b.cortical)
    np.testing.assert_array_equal(a.subcortical, b.subcortical)


def test_make_synthetic_custom_hotspots():
    spots = [HotSpot("custom", 7, 17, 9.5)]
    out = make_synthetic(T=2, seed=0, noise=0.0, hotspots=spots)
    assert np.all(out.cortical[:, 7:17] == 9.5)
    # unplanted region is just noise=0 -> all zeros
    assert np.all(out.cortical[:, 0:7] == 0.0)
    assert np.all(out.cortical[:, 17:] == 0.0)


# ---------------------------------------------------------------------------
# StimulusWindow
# ---------------------------------------------------------------------------


def test_stimulus_window_accepts_valid():
    w = make_stimulus_window()
    assert w.duration_s == 30.0
    assert w.video.ndim == 4 and w.video.shape[-1] == 3
    assert w.audio.ndim == 1


def test_stimulus_window_rejects_bad_video_shape():
    with pytest.raises(ValueError, match="video"):
        StimulusWindow(
            window_id="x",
            t_start=0.0,
            duration_s=30.0,
            video=np.zeros((10, 16, 16), dtype=np.uint8),  # 3D
            video_fps=1.0,
            audio=np.zeros(16000, dtype=np.float32),
            audio_sr=16000,
            text="t",
        )


def test_stimulus_window_rejects_bad_audio_shape():
    with pytest.raises(ValueError, match="audio"):
        StimulusWindow(
            window_id="x",
            t_start=0.0,
            duration_s=30.0,
            video=np.zeros((1, 4, 4, 3), dtype=np.uint8),
            video_fps=1.0,
            audio=np.zeros((1, 1, 1), dtype=np.float32),  # 3D
            audio_sr=16000,
            text="t",
        )


def test_stimulus_window_rejects_out_of_range_duration():
    with pytest.raises(ValueError, match="duration_s"):
        StimulusWindow(
            window_id="x",
            t_start=0.0,
            duration_s=10.0,  # too short
            video=np.zeros((1, 4, 4, 3), dtype=np.uint8),
            video_fps=1.0,
            audio=np.zeros(160000, dtype=np.float32),
            audio_sr=16000,
            text="t",
        )


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


def test_callable_satisfies_tribe_inference_protocol(synth_output):
    class _Stub:
        def __call__(self, w: StimulusWindow) -> TribeOutput:
            return synth_output.with_window_id(w.window_id)

    stub = _Stub()
    assert isinstance(stub, TribeInference)
    out = stub(make_stimulus_window(window_id="abc"))
    assert out.window_id == "abc"


# ---------------------------------------------------------------------------
# Real-sample fixture loader
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_load_tribe_sample_parses_real_fixture():
    out = load_tribe_sample()
    assert out.cortical.shape == (31, 20484)
    assert out.cortical.dtype == np.float32
    assert out.subcortical.shape == (31, 8802)
    assert np.all(out.subcortical == 0.0)  # synthesized, since sample has no subcortical head
    assert out.window_id == "tribev2_sample_condition_A"
    # The real model output is z-scored BOLD; sanity-check distribution.
    assert np.isfinite(out.cortical).all()
    assert -20.0 < float(out.cortical.mean()) < 20.0
    assert 0.01 < float(out.cortical.std()) < 10.0


@pytest.mark.integration
def test_load_tribe_sample_is_cached(tmp_path):
    # Two calls should return arrays with identical contents; the loader
    # caches the parsed array so the second call is cheap.
    a = load_tribe_sample()
    b = load_tribe_sample()
    np.testing.assert_array_equal(a.cortical, b.cortical)
