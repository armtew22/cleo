"""Tests for Glasser → Destrieux group mapping.

Run: ``python -m unittest test_glasser_groups``
"""

from __future__ import annotations

import unittest

from cleo.glasser_groups import (
    aggregate_top_regions,
    group_for_region,
    normalize_glasser_name,
)


class TestNormalizeName(unittest.TestCase):
    def test_strips_hemisphere_prefix(self) -> None:
        self.assertEqual(normalize_glasser_name("L_FFC_ROI"), "FFC")
        self.assertEqual(normalize_glasser_name("R_V1_ROI"), "V1")
        self.assertEqual(normalize_glasser_name("lh_A1_ROI"), "A1")
        self.assertEqual(normalize_glasser_name("rh_A1_ROI"), "A1")

    def test_passes_through_bare_name(self) -> None:
        self.assertEqual(normalize_glasser_name("FFC"), "FFC")
        self.assertEqual(normalize_glasser_name("V1"), "V1")

    def test_strips_only_roi_suffix(self) -> None:
        self.assertEqual(normalize_glasser_name("FFC_ROI"), "FFC")
        # Don't strip something that isn't _ROI
        self.assertEqual(normalize_glasser_name("FFC_extra"), "FFC_extra")


class TestGroupForRegion(unittest.TestCase):
    def test_auditory(self) -> None:
        self.assertEqual(group_for_region("L_A1_ROI"), "auditory")
        self.assertEqual(group_for_region("R_LBelt_ROI"), "auditory")
        self.assertEqual(group_for_region("L_STSva_ROI"), "auditory")

    def test_visual(self) -> None:
        self.assertEqual(group_for_region("L_V1_ROI"), "visual")
        self.assertEqual(group_for_region("R_FFC_ROI"), "visual")
        self.assertEqual(group_for_region("L_MT_ROI"), "visual")
        self.assertEqual(group_for_region("R_PHA1_ROI"), "visual")

    def test_limbic(self) -> None:
        self.assertEqual(group_for_region("L_EC_ROI"), "limbic_adjacent")
        self.assertEqual(group_for_region("R_TGd_ROI"), "limbic_adjacent")
        self.assertEqual(group_for_region("L_AAIC_ROI"), "limbic_adjacent")
        self.assertEqual(group_for_region("R_OFC_ROI"), "limbic_adjacent")
        self.assertEqual(group_for_region("L_25_ROI"), "limbic_adjacent")

    def test_unmapped_returns_none(self) -> None:
        # Somatomotor / dorsal-attention / fronto-parietal parcels are not mapped.
        self.assertIsNone(group_for_region("L_4_ROI"))            # primary motor
        self.assertIsNone(group_for_region("R_MadeUpRegion_ROI"))


class TestAggregateTopRegions(unittest.TestCase):
    def test_buckets_correctly(self) -> None:
        top = [
            {"name": "R_FFC_ROI",   "z_score": 3.42, "direction": "activation"},
            {"name": "L_V1_ROI",    "z_score": 2.10, "direction": "activation"},
            {"name": "L_A1_ROI",    "z_score": 1.80, "direction": "activation"},
            {"name": "R_STSva_ROI", "z_score": 2.40, "direction": "activation"},
            {"name": "L_EC_ROI",    "z_score": 1.55, "direction": "activation"},
            {"name": "R_4_ROI",     "z_score": 1.70, "direction": "activation"},  # unmapped
        ]
        out = aggregate_top_regions(top)

        # Sets of three required groups, all present
        self.assertSetEqual(set(out.keys()), {"auditory", "visual", "limbic_adjacent"})

        # Visual: FFC (3.42), V1 (2.10) → peak 3.42, sustained 2.76
        self.assertEqual(out["visual"]["n_vertices"], 2)
        self.assertAlmostEqual(out["visual"]["z_peak"], 3.42)
        self.assertAlmostEqual(out["visual"]["z_sustained"], (3.42 + 2.10) / 2)

        # Auditory: A1 (1.80), STSva (2.40) → peak 2.40, sustained 2.10
        self.assertEqual(out["auditory"]["n_vertices"], 2)
        self.assertAlmostEqual(out["auditory"]["z_peak"], 2.40)
        self.assertAlmostEqual(out["auditory"]["z_sustained"], 2.10)

        # Limbic: EC (1.55) → peak 1.55, sustained 1.55
        self.assertEqual(out["limbic_adjacent"]["n_vertices"], 1)
        self.assertAlmostEqual(out["limbic_adjacent"]["z_peak"], 1.55)

    def test_empty_input_yields_zero_groups(self) -> None:
        out = aggregate_top_regions([])
        for g in ("auditory", "visual", "limbic_adjacent"):
            self.assertIn(g, out)
            self.assertEqual(out[g]["n_vertices"], 0)
            self.assertEqual(out[g]["z_peak"], 0.0)
            self.assertEqual(out[g]["z_sustained"], 0.0)

    def test_skips_entries_without_z_score(self) -> None:
        top = [
            {"name": "R_FFC_ROI", "z_score": 2.0},
            {"name": "L_V1_ROI"},  # missing z_score → skip
        ]
        out = aggregate_top_regions(top)
        self.assertEqual(out["visual"]["n_vertices"], 1)


if __name__ == "__main__":
    unittest.main()
