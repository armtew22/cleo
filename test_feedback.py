"""Tests for caregiver feedback: deterministic score + synthetic ROI stats.

No GPU or TRIBE; optional mocked Claude call. Run: ``python -m unittest test_feedback``
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from cleo.feedback import compute_load_score, generate_feedback, score_icon


# Minimal shape matching aggregate_rois (values arbitrary for z-score math)
SYNTHETIC_ROI_STATS: dict[str, dict[str, float]] = {
    "auditory": {
        "n_vertices": 100,
        "peak": 0.5,
        "sustained": 0.1,
        "z_sustained": 1.0,
        "z_peak": 0.5,
    },
    "visual": {
        "n_vertices": 200,
        "peak": 0.6,
        "sustained": 0.15,
        "z_sustained": 1.2,
        "z_peak": 0.6,
    },
    "limbic_adjacent": {
        "n_vertices": 80,
        "peak": 0.4,
        "sustained": 0.2,
        "z_sustained": 0.3,
        "z_peak": 0.3,
    },
}

SYNTHETIC_RANKING = ["visual", "auditory", "limbic_adjacent"]


class TestComputeLoadScore(unittest.TestCase):
    def test_top_z_maps_to_score(self) -> None:
        # top z_sustained = 1.2 → 5 + 2*1.2 = 7.4 → round 7
        s = compute_load_score(SYNTHETIC_ROI_STATS)
        self.assertEqual(s, 7)

    def test_clamped_bounds(self) -> None:
        low = {
            g: {**v, "z_sustained": -2.0}
            for g, v in SYNTHETIC_ROI_STATS.items()
        }
        self.assertEqual(compute_load_score(low), 1)
        high = {
            g: {**v, "z_sustained": 2.0}
            for g, v in SYNTHETIC_ROI_STATS.items()
        }
        self.assertEqual(compute_load_score(high), 9)
        self.assertEqual(
            compute_load_score(
                {g: {**v, "z_sustained": 2.5} for g, v in SYNTHETIC_ROI_STATS.items()}
            ),
            10,
        )


class TestScoreIcon(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(score_icon(1), "🟢")
        self.assertEqual(score_icon(3), "🟢")
        self.assertEqual(score_icon(4), "🟡")
        self.assertEqual(score_icon(6), "🟡")
        self.assertEqual(score_icon(7), "🔴")
        self.assertEqual(score_icon(10), "🔴")


class TestSubcorticalScoreInfluence(unittest.TestCase):
    def test_high_amygdala_drives_score_above_cortical(self) -> None:
        # cortical top = 1.2, amygdala = 2.3 → score uses 2.3 → 5 + 2*2.3 = 9.6 → 10
        sub = {"Amygdala": 2.3, "Hippocampus": 0.1, "Thalamus": -0.2}
        self.assertEqual(compute_load_score(SYNTHETIC_ROI_STATS, sub), 10)

    def test_low_subcortical_does_not_lower_score(self) -> None:
        # cortical top still 1.2 → 7; subcortical max = -0.5 must not win
        sub = {"Amygdala": -0.5, "Hippocampus": -1.0, "Thalamus": -2.0}
        self.assertEqual(compute_load_score(SYNTHETIC_ROI_STATS, sub), 7)

    def test_no_subcortical_falls_back_to_cortical(self) -> None:
        self.assertEqual(
            compute_load_score(SYNTHETIC_ROI_STATS, None),
            compute_load_score(SYNTHETIC_ROI_STATS),
        )


class TestGenerateFeedbackMocked(unittest.TestCase):
    def _stub_anthropic(self) -> tuple[MagicMock, Any]:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(type="text", text="## test markdown")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        fake_pkg = MagicMock()
        fake_pkg.Anthropic = MagicMock(return_value=mock_client)
        return mock_client, fake_pkg

    def test_returns_dict_with_markdown(self) -> None:
        """Stub ``anthropic`` via ``sys.modules`` so this runs without the SDK."""
        mock_client, fake_pkg = self._stub_anthropic()

        with patch.dict(sys.modules, {"anthropic": fake_pkg}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                out = generate_feedback(
                    "A quiet room with carpet.",
                    SYNTHETIC_ROI_STATS,
                    SYNTHETIC_RANKING,
                )
        self.assertEqual(out["score"], 7)
        self.assertEqual(out["icon"], "🔴")  # 7 is above the 4–6 yellow band
        self.assertEqual(out["dominant_roi"], "visual")
        self.assertEqual(out["dominant_label"], "visual cortex")
        self.assertIn("markdown", out)
        self.assertIn("test markdown", out["markdown"])

    def test_subcortical_takes_dominant_when_higher(self) -> None:
        mock_client, fake_pkg = self._stub_anthropic()
        sub = {"Amygdala": 2.4, "Hippocampus": 0.1, "Thalamus": 0.0}

        with patch.dict(sys.modules, {"anthropic": fake_pkg}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                out = generate_feedback(
                    "A busy entrance hall with PA announcements.",
                    SYNTHETIC_ROI_STATS,
                    SYNTHETIC_RANKING,
                    subcortical_z=sub,
                )
        self.assertEqual(out["score"], 10)
        self.assertEqual(out["dominant_roi"], "Amygdala")
        self.assertIn("amygdala", out["dominant_label"].lower())

        # The subcortical numbers must reach the LLM
        sent = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("amygdala", sent.lower())
        self.assertIn("+2.40", sent)


if __name__ == "__main__":
    unittest.main()
