"""End-to-end pipeline: clip in → ROI forecast out.

  demux audio  →  generate objective caption  →  TRIBE inference
                                                       │
                                                       ▼
                              aggregate per-ROI peak / sustained / z-score
                                                       │
                                                       ▼
                                              forecast dict (JSON-ready)

TRIBE is imported lazily and cached at module scope so a long-running process
(e.g. a server, or batch over many clips) doesn't reload the model weights
each call.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cleo import rois
from cleo.captioning import caption_clip

_MODEL = None
_MODEL_NAME = "facebook/tribev2"


def _get_model():
    """Load TRIBE v2 once and cache."""
    global _MODEL
    if _MODEL is None:
        from tribev2 import TribeModel  # heavy; only available on GPU box

        _MODEL = TribeModel.from_pretrained(_MODEL_NAME, cache_folder="./cache")
    return _MODEL


def extract_audio(video_path: Path, out_wav: Path) -> Path:
    """Demux audio to a 16 kHz mono wav."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-loglevel", "error",
            str(out_wav),
        ],
        check=True,
    )
    return out_wav


def aggregate_rois(preds: np.ndarray) -> dict[str, dict[str, float]]:
    """For each ROI group, compute peak, sustained, and z-score vs whole cortex.

    `preds` has shape (T, n_vertices).
    Returns: { group: { peak, sustained, n_vertices, z_sustained, z_peak } }
    """
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D (T, n_vertices); got {preds.shape}")
    if preds.shape[1] != rois.TOTAL_VERTICES:
        # Don't fail hard — TRIBE could conceivably grow; just warn and proceed.
        print(
            f"[warn] preds has {preds.shape[1]} vertices, expected "
            f"{rois.TOTAL_VERTICES}. ROI lookup assumes fsaverage5 ordering."
        )

    global_mean = float(preds.mean())
    global_std = float(preds.std()) or 1.0

    out: dict[str, dict[str, float]] = {}
    for group, idx in rois.all_group_vertex_indices().items():
        group_ts = preds[:, idx].mean(axis=1)  # (T,)
        peak = float(group_ts.max())
        sustained = float(group_ts.mean())
        out[group] = {
            "n_vertices": int(idx.size),
            "peak": peak,
            "sustained": sustained,
            "z_sustained": (sustained - global_mean) / global_std,
            "z_peak": (peak - global_mean) / global_std,
        }
    return out


def rank_groups(roi_stats: dict[str, dict[str, float]]) -> list[str]:
    """Rank groups by sustained z-score, descending."""
    return sorted(roi_stats, key=lambda g: roi_stats[g]["z_sustained"], reverse=True)


def run_forecast(video_path: Path) -> dict[str, Any]:
    """Run the full pipeline on a single clip and return a forecast dict."""
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav_path = extract_audio(video_path, td_path / "audio.wav")

        caption = caption_clip(video_path)

        text_path = td_path / "caption.txt"
        text_path.write_text(caption, encoding="utf-8")

        model = _get_model()
        events = model.get_events_dataframe(
            video_path=str(video_path),
            audio_path=str(wav_path),
            text_path=str(text_path),
        )
        preds, segments = model.predict(events=events)

        preds = np.asarray(preds)
        roi_stats = aggregate_rois(preds)
        ranking = rank_groups(roi_stats)

        return {
            "video": str(video_path),
            "caption": caption,
            "preds_shape": list(preds.shape),
            "n_segments": int(getattr(segments, "shape", [0])[0]) if segments is not None else None,
            "roi_stats": roi_stats,
            "ranking_by_sustained_z": ranking,
            "notes": (
                "TRIBE v2 predicts cortical surface only (fsaverage5). "
                "limbic_adjacent values are cortical proxies "
                "(parahippocampal, entorhinal-adjacent, anterior insula, "
                "temporal pole) — not direct hippocampus/amygdala readout."
            ),
        }
