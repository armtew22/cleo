"""Unit tests for tribe_backend.api.artifacts pure helpers.

Phase C: artifact path resolution + name allowlist + job_id sanitization.

These helpers are pure (no FastAPI, no JobStore) — they live in their own
module so the routing layer stays a thin shim that just plumbs them in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tribe_backend.api import artifacts as art


# ----------------------------------------------------------------- allowlist

class TestArtifactNameAllowlist:
    def test_known_names_pass(self) -> None:
        for name in ("brain_meta.json", "brain_colors.bin",
                     "brain_vertices.bin", "brain_faces.bin"):
            assert art.is_allowed_artifact_name(name) is True

    def test_unknown_names_rejected(self) -> None:
        for name in ("anything_else.bin", "../etc/passwd", "brain_meta.json/x",
                     "BRAIN_META.JSON", "brain_meta", ""):
            assert art.is_allowed_artifact_name(name) is False


# ------------------------------------------------------------ id sanitization

class TestSanitizeJobId:
    def test_accepts_iso_timestamp_with_suffix(self) -> None:
        # window_id format: ISO timestamp + "-" + 6-char uuid hex suffix
        wid = "2026-04-25T12:34:56Z-abc123"
        assert art.is_safe_job_id(wid) is True

    def test_accepts_uuid_hex(self) -> None:
        # job_id from JobStore.submit is uuid4().hex (32 lowercase hex chars).
        assert art.is_safe_job_id("a" * 32) is True

    @pytest.mark.parametrize("bad", [
        "../etc/passwd",
        "..",
        ".",
        "/etc/passwd",
        "foo/bar",
        "foo\\bar",
        "foo bar",     # space
        "foo\x00bar",  # null
        "",
        "x" * 200,     # absurdly long
    ])
    def test_rejects_traversal_and_garbage(self, bad: str) -> None:
        assert art.is_safe_job_id(bad) is False


# ----------------------------------------------------------------- path build

class TestArtifactPath:
    def test_path_is_under_out_dir(self, tmp_path: Path) -> None:
        wid = "2026-04-25T12:34:56Z-abc123"
        p = art.artifact_path(tmp_path, wid, "brain_meta.json")
        assert p == tmp_path / wid / "brain_meta.json"

    def test_unsafe_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(art.UnsafeArtifactRequest):
            art.artifact_path(tmp_path, "../escape", "brain_meta.json")

    def test_unknown_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(art.UnsafeArtifactRequest):
            art.artifact_path(tmp_path, "abc123", "evil.bin")

    def test_resolved_path_must_stay_inside_out_dir(self, tmp_path: Path) -> None:
        # Even if id passes the regex but resolves outside (symlink etc.), guard
        # via realpath comparison. We can't easily simulate that without OS
        # tricks, so we just verify the helper resolves under out_dir for valid
        # input.
        wid = "2026-04-25T12:34:56Z-abc123"
        p = art.artifact_path(tmp_path, wid, "brain_colors.bin")
        assert str(p.resolve()).startswith(str(tmp_path.resolve()))
