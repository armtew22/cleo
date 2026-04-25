"""Tests for the Glasser parcellation unit.

No GPU, no atlas files, no network. All atlas index maps are synthetic.
Run: python -m unittest test_glasser -v
"""

from __future__ import annotations

import unittest

import numpy as np

from cleo.glasser import (
    aggregate,
    generate_template_report,
    parcellate,
    rank_and_threshold,
    z_score_scores,
)

# ── synthetic data ─────────────────────────────────────────────────────────────

T = 6          # time steps
N_CORTICAL = 20484

# Three non-overlapping fake parcels at different activation levels
PARCEL_HIGH   = np.arange(0, 100, dtype=np.int32)
PARCEL_MED    = np.arange(100, 300, dtype=np.int32)
PARCEL_LOW    = np.arange(300, 500, dtype=np.int32)

FAKE_CORTICAL_IDX = {
    "FFC":         PARCEL_HIGH,   # fusiform face complex — high activation
    "A1":          PARCEL_MED,    # primary auditory — medium
    "V1":          PARCEL_LOW,    # primary visual — low
}

# Subcortical: same pattern
N_SUBCORTICAL = 8802
AMYG_IDX   = np.arange(0, 50, dtype=np.int32)
HIPPO_IDX  = np.arange(50, 120, dtype=np.int32)

FAKE_SUBCORTICAL_IDX = {
    "Amygdala":    AMYG_IDX,
    "Hippocampus": HIPPO_IDX,
}


def _make_cortical(
    high_val: float = 3.0,
    med_val: float = 1.5,
    low_val: float = 0.2,
    bg_val: float = 0.5,
) -> np.ndarray:
    preds = np.full((T, N_CORTICAL), bg_val, dtype=np.float32)
    preds[:, PARCEL_HIGH] = high_val
    preds[:, PARCEL_MED]  = med_val
    preds[:, PARCEL_LOW]  = low_val
    return preds


def _make_subcortical(
    amyg_val: float = 2.5,
    hippo_val: float = 1.0,
    bg_val: float = 0.3,
) -> np.ndarray:
    sub = np.full((T, N_SUBCORTICAL), bg_val, dtype=np.float32)
    sub[:, AMYG_IDX]  = amyg_val
    sub[:, HIPPO_IDX] = hippo_val
    return sub


# ── parcellate() ──────────────────────────────────────────────────────────────

class TestParcellate(unittest.TestCase):
    def test_output_keys_match_index_map(self) -> None:
        ts = parcellate(_make_cortical(), FAKE_CORTICAL_IDX)
        self.assertSetEqual(set(ts.keys()), set(FAKE_CORTICAL_IDX.keys()))

    def test_output_shape_is_T(self) -> None:
        ts = parcellate(_make_cortical(), FAKE_CORTICAL_IDX)
        for name, arr in ts.items():
            self.assertEqual(arr.shape, (T,), f"wrong shape for {name!r}")

    def test_values_match_input(self) -> None:
        ts = parcellate(_make_cortical(high_val=3.0, med_val=1.5, low_val=0.2), FAKE_CORTICAL_IDX)
        self.assertAlmostEqual(ts["FFC"].mean(), 3.0, places=4)
        self.assertAlmostEqual(ts["A1"].mean(),  1.5, places=4)
        self.assertAlmostEqual(ts["V1"].mean(),  0.2, places=4)

    def test_constant_input_constant_output(self) -> None:
        preds = np.full((T, N_CORTICAL), 1.23, dtype=np.float32)
        ts = parcellate(preds, FAKE_CORTICAL_IDX)
        for arr in ts.values():
            self.assertAlmostEqual(float(arr.std()), 0.0, places=5)

    def test_raises_on_1d_input(self) -> None:
        with self.assertRaises(ValueError):
            parcellate(np.zeros(N_CORTICAL), FAKE_CORTICAL_IDX)

    def test_subcortical_parcellation(self) -> None:
        ts = parcellate(_make_subcortical(), FAKE_SUBCORTICAL_IDX)
        self.assertSetEqual(set(ts.keys()), {"Amygdala", "Hippocampus"})
        self.assertAlmostEqual(ts["Amygdala"].mean(), 2.5, places=4)


# ── aggregate() ───────────────────────────────────────────────────────────────

class TestAggregate(unittest.TestCase):
    def _ts(self) -> dict[str, np.ndarray]:
        return parcellate(_make_cortical(), FAKE_CORTICAL_IDX)

    def test_mean_returns_float_per_roi(self) -> None:
        scores = aggregate(self._ts(), method="mean")
        for v in scores.values():
            self.assertIsInstance(v, float)

    def test_mean_ordering(self) -> None:
        scores = aggregate(self._ts(), method="mean")
        self.assertGreater(scores["FFC"], scores["A1"])
        self.assertGreater(scores["A1"], scores["V1"])

    def test_peak_ge_mean(self) -> None:
        ts = self._ts()
        mean_scores = aggregate(ts, method="mean")
        peak_scores = aggregate(ts, method="peak")
        for k in mean_scores:
            self.assertGreaterEqual(peak_scores[k], mean_scores[k])

    def test_peak_window_between_mean_and_peak(self) -> None:
        ts = self._ts()
        mean_sc = aggregate(ts, method="mean")
        peak_sc = aggregate(ts, method="peak")
        pw_sc   = aggregate(ts, method="peak_window")
        for k in mean_sc:
            self.assertGreaterEqual(pw_sc[k] + 1e-6, mean_sc[k])
            self.assertGreaterEqual(peak_sc[k] + 1e-6, pw_sc[k])

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            aggregate(self._ts(), method="bogus")

    def test_constant_timeseries_mean_equals_value(self) -> None:
        ts = {"X": np.full(T, 2.5)}
        scores = aggregate(ts, method="mean")
        self.assertAlmostEqual(scores["X"], 2.5, places=5)

    def test_empty_dict_returns_empty(self) -> None:
        self.assertEqual(aggregate({}), {})


# ── z_score_scores() ──────────────────────────────────────────────────────────

class TestZScoreScores(unittest.TestCase):
    def test_zero_mean_unit_std(self) -> None:
        scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        z = z_score_scores(scores)
        vals = list(z.values())
        self.assertAlmostEqual(np.mean(vals), 0.0, places=5)
        self.assertAlmostEqual(np.std(vals),  1.0, places=5)

    def test_preserves_ordering(self) -> None:
        scores = {"high": 5.0, "med": 2.0, "low": 0.5}
        z = z_score_scores(scores)
        self.assertGreater(z["high"], z["med"])
        self.assertGreater(z["med"], z["low"])

    def test_constant_input_all_zero(self) -> None:
        scores = {"A": 1.0, "B": 1.0, "C": 1.0}
        z = z_score_scores(scores)
        for v in z.values():
            self.assertAlmostEqual(v, 0.0, places=5)


# ── rank_and_threshold() ──────────────────────────────────────────────────────

class TestRankAndThreshold(unittest.TestCase):
    def _scores(self) -> dict[str, float]:
        return {"FFC": 2.5, "A1": 1.2, "V1": 0.3, "RSC": -0.5}

    def test_threshold_excludes_low(self) -> None:
        result = rank_and_threshold(self._scores(), z_threshold=1.0)
        names = [r[0] for r in result]
        self.assertIn("FFC", names)
        self.assertIn("A1", names)
        self.assertNotIn("V1", names)
        self.assertNotIn("RSC", names)

    def test_sorted_descending(self) -> None:
        result = rank_and_threshold(self._scores(), z_threshold=0.0)
        zvals = [r[1] for r in result]
        self.assertEqual(zvals, sorted(zvals, reverse=True))

    def test_top_k_limit(self) -> None:
        scores = {f"roi_{i}": float(i) for i in range(20)}
        result = rank_and_threshold(scores, top_k=5, z_threshold=0.0)
        self.assertLessEqual(len(result), 5)

    def test_empty_when_all_below_threshold(self) -> None:
        result = rank_and_threshold({"A": -1.0, "B": -2.0}, z_threshold=1.0)
        self.assertEqual(result, [])


# ── generate_template_report() ────────────────────────────────────────────────

class TestGenerateTemplateReport(unittest.TestCase):
    """Tests the report builder against the real glasser_descriptions.json."""

    def _top_regions(self) -> list[tuple[str, float]]:
        return [("Amygdala", 3.5), ("FFC", 2.1), ("A1", 1.3), ("UnknownParcel99", 1.1)]

    def test_report_is_string(self) -> None:
        self.assertIsInstance(generate_template_report(self._top_regions()), str)

    def test_report_contains_known_region_names(self) -> None:
        report = generate_template_report(self._top_regions())
        self.assertIn("Amygdala", report)
        self.assertIn("FFC", report)

    def test_report_handles_missing_parcel_gracefully(self) -> None:
        report = generate_template_report(self._top_regions())
        self.assertIn("UnknownParcel99", report)
        self.assertIn("No detailed description", report)

    def test_empty_top_regions(self) -> None:
        report = generate_template_report([])
        self.assertIn("No regions exceeded", report)

    def test_high_intensity_label_for_z_ge_2(self) -> None:
        report = generate_template_report([("Amygdala", 2.5)])
        self.assertIn("high", report)

    def test_moderate_intensity_for_z_between_1_and_2(self) -> None:
        report = generate_template_report([("Amygdala", 1.5)])
        self.assertIn("moderate", report)

    def test_report_contains_disclaimer(self) -> None:
        report = generate_template_report([("V1", 1.5)])
        self.assertIn("TRIBE v2", report)


# ── GlasserParcellationUnit integration ───────────────────────────────────────

class TestGlasserParcellationUnit(unittest.TestCase):
    """Tests the full unit run() method with synthetic index maps."""

    def _unit(self):
        from cleo.glasser import GlasserParcellationUnit
        return GlasserParcellationUnit(
            cortical_index_map=FAKE_CORTICAL_IDX,
            subcortical_index_map=FAKE_SUBCORTICAL_IDX,
        )

    def test_run_returns_expected_keys(self) -> None:
        unit = self._unit()
        out = unit.run(_make_cortical(), _make_subcortical())
        for key in ("cortical_scores", "subcortical_scores", "all_scores",
                    "z_scores", "top_regions", "template_report"):
            self.assertIn(key, out)

    def test_run_top_regions_sorted_descending(self) -> None:
        unit = self._unit()
        out = unit.run(_make_cortical(), _make_subcortical(), top_k=10, z_threshold=0.0)
        zvals = [z for _, z in out["top_regions"]]
        self.assertEqual(zvals, sorted(zvals, reverse=True))

    def test_run_without_subcortical(self) -> None:
        unit = self._unit()
        out = unit.run(_make_cortical(), subcortical_output=None)
        self.assertEqual(out["subcortical_scores"], {})

    def test_run_raises_without_cortical_atlas(self) -> None:
        from cleo.glasser import GlasserParcellationUnit
        unit = GlasserParcellationUnit()  # no atlas provided
        with self.assertRaises(RuntimeError):
            unit.run(_make_cortical())


if __name__ == "__main__":
    unittest.main()
