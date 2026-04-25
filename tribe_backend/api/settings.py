"""Environment-driven Settings for the API.

Single source of truth for the few env vars Phase A reads:

    TRIBE_INFERENCE        "fake" (default) | "gpu"
    OUT_DIR                directory for ControlUnit artifacts (default: ./out)
    CORS_ALLOW_ORIGINS     comma-separated origin list (default: "*")

A pydantic-free implementation keeps the dependency surface tight — all we
need is os.environ + a tiny dataclass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


_VALID_INFERENCE = ("fake", "gpu")


@dataclass(frozen=True)
class Settings:
    inference: Literal["fake", "gpu"] = field(default="fake")
    out_dir: str = field(default="./out")
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])
    queue_depth: int = field(default=4)
    drain_timeout_s: float = field(default=5.0)

    def __init__(self) -> None:  # type: ignore[override]
        # Read env at construction so tests can monkeypatch reliably.
        inference = os.environ.get("TRIBE_INFERENCE", "fake").strip().lower()
        if inference not in _VALID_INFERENCE:
            raise ValueError(
                f"TRIBE_INFERENCE must be one of {_VALID_INFERENCE!r}; got {inference!r}"
            )
        out_dir = os.environ.get("OUT_DIR", "./out")
        cors_raw = os.environ.get("CORS_ALLOW_ORIGINS", "*")
        cors = [o.strip() for o in cors_raw.split(",") if o.strip()]
        if not cors:
            cors = ["*"]
        try:
            queue_depth = int(os.environ.get("TRIBE_QUEUE_DEPTH", "4"))
        except ValueError:
            raise ValueError("TRIBE_QUEUE_DEPTH must be an integer")
        if queue_depth < 1:
            raise ValueError("TRIBE_QUEUE_DEPTH must be >= 1")
        try:
            drain_timeout = float(os.environ.get("TRIBE_DRAIN_TIMEOUT_S", "5.0"))
        except ValueError:
            raise ValueError("TRIBE_DRAIN_TIMEOUT_S must be a float")
        # frozen=True dataclass: bypass setattr restriction via object.__setattr__
        object.__setattr__(self, "inference", inference)
        object.__setattr__(self, "out_dir", out_dir)
        object.__setattr__(self, "cors_allow_origins", cors)
        object.__setattr__(self, "queue_depth", queue_depth)
        object.__setattr__(self, "drain_timeout_s", drain_timeout)
