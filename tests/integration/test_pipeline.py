"""End-to-end tests where all four compartments coexist.

This is the FIRST module that imports parcellation, mesh, inference, and
control together — the cross-module invariants live here.

Phase 2.5 invariant: for the same TribeOutput, the parcel that
GlasserParcellationUnit ranks #1 (by |window-mean z-score|) must also be the
spatial region with the highest |mean color intensity| in BrainMeshExporter's
output. Catches drift between the two modules' interpretation of vertex
ordering ([0:10242]=LH, [10242:20484]=RH).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.conftest import HotSpot, make_synthetic, make_stimulus_window
from tribe_backend.contracts import CORTICAL_VERTICES
from tribe_backend.control import default_control_unit
from tribe_backend.inference import FakeTribeInference
from tribe_backend.mesh import BrainMeshExporter, aggregate_temporal
from tribe_backend.parcellation import GlasserParcellationUnit


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Cross-module invariant (plan §3 Phase 2 item 6)
# ---------------------------------------------------------------------------


def test_top_parcel_matches_brightest_mesh_region():
    """Plant a strong activation in a parcel; verify both modules see it.

    We pick a single Glasser parcel from the live atlas, plant a large positive
    activation across all of its vertices, and confirm:
      - parcellation ranks that parcel #1 (positive direction).
      - the mesh exporter's per-vertex mean color (RGB norm) is highest at
        those same vertex indices, on average, vs. any other parcel.
    """
    parcellation = GlasserParcellationUnit()
    cort_names = parcellation.cortical_names

    # Pick the largest parcel — robust whether the atlas is real (Glasser-360,
    # parcels ~50-200 vertices) or the deterministic synthetic fallback
    # (parcels ~57 vertices each).
    target_name = max(cort_names, key=lambda n: int(parcellation._cort_masks[n].sum()))
    target_mask = parcellation._cort_masks[target_name]
    assert int(target_mask.sum()) >= 20, (
        f"largest parcel {target_name!r} has {int(target_mask.sum())} vertices — "
        "atlas seems degenerate"
    )

    # Build a synthetic output where the target parcel's vertices are +5.0
    # (well above z_threshold=1.5) and all other vertices are tiny noise.
    rng = np.random.default_rng(0)
    cortical = rng.standard_normal((31, CORTICAL_VERTICES), dtype=np.float32) * 0.05
    cortical[:, target_mask] = 5.0
    out = make_synthetic(T=31, seed=0, noise=0.0, hotspots=[])
    # Replace cortical only — keep the contract-valid subcortical from the factory.
    from dataclasses import replace
    out = replace(out, cortical=cortical, window_id="invariant-1")

    # --- Parcellation side ---
    report = parcellation.generate_report(out, z_threshold=1.5)
    assert len(report.top_regions) >= 1
    top = report.top_regions[0]
    assert top.name == target_name, (
        f"parcellation ranked {top.name!r} #1; expected {target_name!r}"
    )
    assert top.direction == "activation"

    # --- Mesh side ---
    mesh = BrainMeshExporter()
    per_vertex = aggregate_temporal(out.cortical, method="mean")          # (20484,)
    rgba = mesh._colors_for(per_vertex)                                   # (20484, 4) uint8
    rgb = rgba[:, :3].astype(np.float32)
    # The neutral color of the colormap (whatever it is — for RdBu_r centered
    # at vmid=0 it is ~white, not gray). Compute it by colorizing a single
    # zero-activation vertex through the same path.
    neutral = mesh._colors_for(np.zeros(1, dtype=np.float32))[0, :3].astype(np.float32)
    intensity = np.linalg.norm(rgb - neutral, axis=1)                     # (20484,)

    target_intensity = float(intensity[target_mask].mean())
    other_intensity = float(intensity[~target_mask].mean())
    assert target_intensity > other_intensity, (
        f"mesh disagrees: target parcel mean intensity {target_intensity:.1f} "
        f"<= other {other_intensity:.1f}"
    )


# ---------------------------------------------------------------------------
# default_control_unit composition root
# ---------------------------------------------------------------------------


def test_default_control_unit_process_window_produces_report_and_mesh(tmp_path):
    """Smoke-test the full composition root with a FakeTribeInference."""
    fixture = make_synthetic(T=31, seed=7, window_id="will-be-overwritten")
    control = default_control_unit(out_dir=tmp_path, fake_fixture=fixture)

    win = make_stimulus_window(window_id="ctrl-1", duration_s=30.0)
    report = control.process_window(
        video=win.video,
        audio=win.audio,
        audio_sr=win.audio_sr,
        text=win.text,
        video_fps=win.video_fps,
    )

    # Report is the parcellation Report and traces back to the window
    assert hasattr(report, "top_regions")
    assert hasattr(report, "window_id")
    # ControlUnit synthesizes a window_id; FakeTribeInference stamps it onto the
    # output, and parcellation propagates it.
    assert report.window_id is not None
    # JSON round-trip via the Report API.
    blob = report.to_json()
    json.loads(blob)  # parses

    # Mesh artifacts written under <out_dir>/<window_id>/
    out_subdir = tmp_path / report.window_id
    assert out_subdir.exists()
    assert (out_subdir / "brain_meta.json").exists()
    assert (out_subdir / "brain_colors.bin").exists()
    meta = json.loads((out_subdir / "brain_meta.json").read_text())
    assert meta["vertex_count"] == CORTICAL_VERTICES


def test_default_control_unit_requires_inference_or_fixture(tmp_path):
    with pytest.raises(ValueError, match="inference.*fake_fixture"):
        default_control_unit(out_dir=tmp_path)


def test_default_control_unit_accepts_explicit_inference(tmp_path):
    fixture = make_synthetic(T=31, seed=1)
    inf = FakeTribeInference(fixture)
    control = default_control_unit(out_dir=tmp_path, inference=inf)
    win = make_stimulus_window(window_id="ctrl-2")
    report = control.process_window(
        video=win.video,
        audio=win.audio,
        audio_sr=win.audio_sr,
        text=win.text,
        video_fps=win.video_fps,
    )
    assert report.window_id is not None
