"""Settings: env-driven configuration for the API."""
from __future__ import annotations

import os

from tribe_backend.api.settings import Settings


def test_defaults(monkeypatch):
    for k in ("TRIBE_INFERENCE", "OUT_DIR", "CORS_ALLOW_ORIGINS"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.inference == "fake"
    assert s.out_dir == "./out"
    assert s.cors_allow_origins == ["*"]


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIBE_INFERENCE", "gpu")
    monkeypatch.setenv("OUT_DIR", str(tmp_path))
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:3000")
    s = Settings()
    assert s.inference == "gpu"
    assert s.out_dir == str(tmp_path)
    assert s.cors_allow_origins == ["http://localhost:5173", "http://localhost:3000"]


def test_invalid_inference_rejected(monkeypatch):
    monkeypatch.setenv("TRIBE_INFERENCE", "bogus")
    import pytest
    with pytest.raises(ValueError):
        Settings()
