"""Test double for the TribeInference protocol.

Used by every downstream compartment (control, parcellation, mesh) to drive
the pipeline without a GPU. The real GPU adapter lands in Phase 5.
"""
from __future__ import annotations

from dataclasses import replace

from tribe_backend.contracts import StimulusWindow, TribeOutput

__all__ = ["FakeTribeInference"]


class FakeTribeInference:
    """Returns a pre-built TribeOutput, stamped with the incoming window_id.

    Satisfies the :class:`tribe_backend.contracts.TribeInference` protocol.
    """

    def __init__(self, fixture: TribeOutput) -> None:
        self._fixture = fixture

    def __call__(self, window: StimulusWindow) -> TribeOutput:
        return replace(self._fixture, window_id=window.window_id)
