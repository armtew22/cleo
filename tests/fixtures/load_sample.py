"""Loader for the real TRIBE v2 sample output committed at agent/tribev2_sample_output.txt.

The sample is a header + verbatim NumPy `repr` of a (31, 20484) float32 array.
We parse it lazily and cache the result so repeated calls are cheap.

The sample only carries the cortical head; we synthesize a zero-filled
subcortical of shape (31, 8802) to satisfy the TribeOutput contract.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from tribe_backend.contracts import (
    EXPECTED_T_PER_30S_WINDOW,
    SUBCORTICAL_VOXELS,
    TribeOutput,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_PATH = _REPO_ROOT / "agent" / "tribev2_sample_output.txt"
SAMPLE_WINDOW_ID = "tribev2_sample_condition_A"


def _parse_sample_file(path: Path) -> np.ndarray:
    text = path.read_text()
    marker = "=== preds (verbatim, full array) ==="
    if marker not in text:
        raise ValueError(f"sample file missing marker '{marker}': {path}")
    body = text.split(marker, 1)[1]

    # Trim to the array literal: starts at first '[', ends at the matching ']]'.
    start = body.find("[[")
    end = body.rfind("]]")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"could not find array literal in {path}")
    literal = body[start : end + 2]

    # `np.array(...)` repr uses whitespace-separated floats; convert to a flat
    # list and reshape. Strip brackets and split on whitespace.
    flat = re.sub(r"[\[\]]", " ", literal)
    tokens = flat.split()
    arr = np.fromiter((float(t) for t in tokens), dtype=np.float32, count=len(tokens))

    expected = EXPECTED_T_PER_30S_WINDOW * 20484
    if arr.size != expected:
        raise ValueError(
            f"parsed {arr.size} floats; expected {expected} "
            f"(={EXPECTED_T_PER_30S_WINDOW}*20484)"
        )
    return arr.reshape(EXPECTED_T_PER_30S_WINDOW, 20484)


@lru_cache(maxsize=4)
def _cached_cortical(path_str: str) -> np.ndarray:
    return _parse_sample_file(Path(path_str))


def load_tribe_sample(
    path: str | Path | None = None,
    *,
    window_id: str = SAMPLE_WINDOW_ID,
) -> TribeOutput:
    """Parse the real sample fixture into a TribeOutput.

    The cortical tensor is the verbatim sample; subcortical is zero-filled
    with shape (31, 8802) since the sample only carries the cortical head.
    """
    p = Path(path) if path is not None else DEFAULT_SAMPLE_PATH
    cortical = _cached_cortical(str(p)).copy()
    subcortical = np.zeros((cortical.shape[0], SUBCORTICAL_VOXELS), dtype=np.float32)
    return TribeOutput(
        cortical=cortical,
        subcortical=subcortical,
        fps=1.0,
        surface="fsaverage5",
        subcortical_atlas="harvard_oxford_2mm",
        window_id=window_id,
    )
