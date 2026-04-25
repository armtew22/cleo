"""GlasserParcellationUnit — ROI interpretation for TRIBE v2 cortical/subcortical output.

Public API:
    GlasserParcellationUnit  — generate_report(output) -> Report
    Report                   — top regions + narrative text + window_id
    WindowSizeWarning        — emitted when T != EXPECTED_T_PER_30S_WINDOW
"""
from __future__ import annotations

from .unit import (
    GlasserParcellationUnit,
    Report,
    RegionActivation,
    WindowSizeWarning,
)

__all__ = [
    "GlasserParcellationUnit",
    "Report",
    "RegionActivation",
    "WindowSizeWarning",
]
