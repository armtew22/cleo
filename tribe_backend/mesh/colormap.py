"""Pure colormap functions for the mesh exporter.

`activation_to_rgba` maps z-scored activations to per-vertex RGBA bytes,
optionally darkening RGB by a per-vertex sulcal depth.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib import colormaps


def activation_to_rgba(
    activations: np.ndarray,
    *,
    vmin: float = -3.0,
    vmax: float = 3.0,
    cmap: str = "RdBu_r",
    sulcal_depth: Optional[np.ndarray] = None,
    sulcal_strength: float = 0.4,
) -> np.ndarray:
    """Map activations to RGBA uint8 in [0, 255].

    Parameters
    ----------
    activations : array, any shape ending in V (vertices). Values are clipped
                  to [vmin, vmax] before colormapping.
    vmin, vmax  : clipping range. Symmetric around 0 by default so 0 maps to
                  the colormap midpoint.
    cmap        : matplotlib colormap name. "RdBu_r" puts white at center.
    sulcal_depth: optional same-shape-as-activations array in [0, 1]; 1.0
                  darkens RGB toward black, 0.0 leaves it unchanged.
    sulcal_strength : maximum fractional darkening at sulcal_depth==1.0.

    Returns
    -------
    rgba : same shape as activations + (4,), dtype uint8.
    """
    a = np.asarray(activations)
    a_clipped = np.clip(a, vmin, vmax)
    # Normalize to [0, 1] for the colormap. Guard against vmin == vmax.
    span = vmax - vmin
    if span <= 0:
        raise ValueError(f"vmax ({vmax}) must be strictly greater than vmin ({vmin})")
    norm = (a_clipped - vmin) / span

    cmap_obj = colormaps.get_cmap(cmap)
    rgba_float = cmap_obj(norm)  # shape (..., 4) float in [0, 1]

    if sulcal_depth is not None:
        sd = np.asarray(sulcal_depth, dtype=np.float32)
        if sd.shape != a.shape:
            raise ValueError(
                f"sulcal_depth shape {sd.shape} must match activations shape {a.shape}"
            )
        sd = np.clip(sd, 0.0, 1.0)
        factor = 1.0 - sulcal_strength * sd  # in [1 - strength, 1.0]
        rgba_float[..., :3] *= factor[..., None]

    rgba_float = np.clip(rgba_float, 0.0, 1.0)
    return (rgba_float * 255.0 + 0.5).astype(np.uint8)
