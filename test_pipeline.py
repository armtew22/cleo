"""Tests for pipeline.rank_groups (pure ranking; no GPU, no network).

The old ``aggregate_rois`` math has been retired now that the upstream TRIBE
backend returns Glasser top-regions directly — see ``test_glasser_groups.py``
for the replacement (``aggregate_top_regions``).

Run: ``python -m unittest test_pipeline``
"""

from __future__ import annotations

import unittest


class TestRankGroups(unittest.TestCase):
    def _stats(self, z_auditory: float, z_visual: float, z_limbic: float) -> dict:
        def _entry(z: float) -> dict:
            return {"n_vertices": 100, "peak": 0.5, "sustained": 0.3,
                    "z_sustained": z, "z_peak": 0.4}
        return {
            "auditory":        _entry(z_auditory),
            "visual":          _entry(z_visual),
            "limbic_adjacent": _entry(z_limbic),
        }

    def test_rank_descending(self) -> None:
        from cleo.pipeline import rank_groups
        stats = self._stats(z_auditory=1.5, z_visual=0.3, z_limbic=-0.5)
        self.assertEqual(rank_groups(stats), ["auditory", "visual", "limbic_adjacent"])

    def test_rank_visual_top(self) -> None:
        from cleo.pipeline import rank_groups
        stats = self._stats(z_auditory=0.1, z_visual=2.0, z_limbic=0.5)
        self.assertEqual(rank_groups(stats)[0], "visual")

    def test_rank_single_group(self) -> None:
        from cleo.pipeline import rank_groups
        self.assertEqual(rank_groups({"auditory": {"z_sustained": 1.0}}), ["auditory"])

    def test_rank_stable_for_equal_z(self) -> None:
        from cleo.pipeline import rank_groups
        stats = self._stats(z_auditory=1.0, z_visual=1.0, z_limbic=1.0)
        self.assertEqual(set(rank_groups(stats)), {"auditory", "visual", "limbic_adjacent"})


if __name__ == "__main__":
    unittest.main()
