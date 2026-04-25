"""FastAPI dependency providers.

A single ControlUnit is built at app startup and stashed on `app.state`. These
helpers pull it back out for handlers via `Depends(...)`.
"""
from __future__ import annotations

from fastapi import Request

from tribe_backend.control.unit import ControlUnit


def get_control_unit(request: Request) -> ControlUnit:
    cu = getattr(request.app.state, "control_unit", None)
    if cu is None:
        raise RuntimeError(
            "control_unit not initialized on app.state — did you skip the lifespan?"
        )
    return cu


def get_settings(request: Request):
    return request.app.state.settings
