"""Phase 4 — placeholder for the real GPU adapter.

The actual ``GpuTribeInference`` (a.k.a. ``tribev2_adapter.py`` /
``gpu_runner.py``) lands in Phase 5 once weights are downloaded and a CUDA
runner is available. This test exists so the suite documents the boundary
and so CI surfaces the gap as a SKIPPED reason rather than silence.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.gpu


@pytest.mark.skip(reason="Phase 5 — needs CUDA + tribev2 weights")
def test_gpu_runner_smoke_forward_pass() -> None:
    """Single forward pass on a tiny synthetic 30s window.

    Phase-5 acceptance: returns a TribeOutput with cortical (31, 20484)
    and subcortical (31, 8802), all values finite, window_id stamped.
    """
    raise NotImplementedError
