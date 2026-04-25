"""HTTP API package — thin transport over ControlUnit.

Public surface:
    get_app   — FastAPI app factory (uvicorn --factory entrypoint)
    app       — module-level lazy app (built on first attribute access)

This package may import only `tribe_backend.contracts` and
`tribe_backend.control` from the rest of the backend. Inference, parcellation,
and mesh are reached exclusively through `default_control_unit()`.
"""
from __future__ import annotations

from tribe_backend.api.app import get_app

__all__ = ["get_app", "app"]


def __getattr__(name: str):
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
