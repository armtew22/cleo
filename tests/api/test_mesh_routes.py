"""Tests for Phase 6b mesh/inference endpoints.

Covers:
    GET  /v1/mesh/static          — tar contents, ETag, 304 round-trip
    POST /v1/inference/colors     — binary out, headers, 422 on missing image
    POST /v1/inference/full       — envelope + cached colors GET
    GET  /v1/inference/colors/{window_id} — 404 for unknown id
"""
from __future__ import annotations

import io
import struct
import tarfile

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures (reuse the test_client fixture from conftest.py)
# ---------------------------------------------------------------------------

def _make_jpeg_1px() -> bytes:
    """Minimal valid JPEG: 1x1 white pixel."""
    # This is a known-good 1x1 white JPEG (631 bytes).
    import base64
    DATA = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"A\tBb\x1f0L\x14&Yq6>E\x00\x00\x00 !\x15\x1e\x1e!\x1b\x1c\x1c!!\x00\x00"
    )
    # Use a real minimal JPEG via PIL if available, else raw bytes.
    try:
        from PIL import Image as _Image
        img = _Image.new("RGB", (1, 1), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: minimal hardcoded 1x1 JPEG
    import struct as _s
    # Construct a minimal JFIF JPEG (SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI).
    # This is generated once and known-good.
    MINIMAL_JPEG = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10,  # SOI + APP0 marker + length
        0x4A, 0x46, 0x49, 0x46, 0x00,         # "JFIF\0"
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,  # version, density
        0xFF, 0xDB, 0x00, 0x43, 0x00,         # DQT marker + length + table id
        *([16] * 64),                         # quantization table
        0xFF, 0xC0, 0x00, 0x0B,               # SOF0 + length
        0x08, 0x00, 0x01, 0x00, 0x01,         # precision, height, width
        0x01,                                 # components
        0x01, 0x11, 0x00,                     # component spec
        0xFF, 0xC4, 0x00, 0x1F,               # DHT marker + length
        0x00,                                 # DC table 0
        *([0] * 16),                          # code lengths (all zero except 1)
        0x00,                                 # one code
        0xFF, 0xDA, 0x00, 0x08,               # SOS + length
        0x01,                                 # components
        0x01, 0x00,                           # component + table
        0x00, 0x3F, 0x00,                     # params
        0x7F, 0xFF,                           # compressed data
        0xFF, 0xD9,                           # EOI
    ])
    return bytes(MINIMAL_JPEG)


def _jpeg_bytes() -> bytes:
    return _make_jpeg_1px()


# ---------------------------------------------------------------------------
# GET /v1/mesh/static
# ---------------------------------------------------------------------------

class TestMeshStatic:
    def test_returns_200_with_tar_body(self, test_client: TestClient):
        r = test_client.get("/v1/mesh/static")
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/octet-stream")

    def test_tar_contains_four_expected_files(self, test_client: TestClient):
        r = test_client.get("/v1/mesh/static")
        assert r.status_code == 200
        buf = io.BytesIO(r.content)
        with tarfile.open(fileobj=buf, mode="r") as tf:
            names = set(tf.getnames())
        assert "brain_vertices.bin" in names
        assert "brain_normals.bin" in names
        assert "brain_faces.bin" in names
        assert "brain_meta.json" in names

    def test_tar_meta_json_is_valid(self, test_client: TestClient):
        import json
        r = test_client.get("/v1/mesh/static")
        buf = io.BytesIO(r.content)
        with tarfile.open(fileobj=buf, mode="r") as tf:
            meta_bytes = tf.extractfile("brain_meta.json").read()
        meta = json.loads(meta_bytes)
        assert meta["vertex_count"] == 20484
        assert meta["face_count"] > 0
        assert meta["surface_type"] == "pial"

    def test_etag_header_present(self, test_client: TestClient):
        r = test_client.get("/v1/mesh/static")
        assert "etag" in r.headers
        assert "fsaverage5-pial-v1" in r.headers["etag"]

    def test_cache_control_immutable(self, test_client: TestClient):
        r = test_client.get("/v1/mesh/static")
        cc = r.headers.get("cache-control", "")
        assert "immutable" in cc
        assert "max-age=31536000" in cc

    def test_if_none_match_returns_304(self, test_client: TestClient):
        # First request to get ETag.
        r1 = test_client.get("/v1/mesh/static")
        etag = r1.headers["etag"]
        # Second request with If-None-Match should yield 304.
        r2 = test_client.get("/v1/mesh/static", headers={"If-None-Match": etag})
        assert r2.status_code == 304

    def test_surface_param_changes_etag(self, test_client: TestClient):
        r_pial = test_client.get("/v1/mesh/static?surface=pial")
        r_infl = test_client.get("/v1/mesh/static?surface=infl")
        assert r_pial.headers["etag"] != r_infl.headers["etag"]


# ---------------------------------------------------------------------------
# POST /v1/inference/colors
# ---------------------------------------------------------------------------

class TestInferenceColors:
    def test_returns_200_with_binary(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_body_length_exactly_81936(self, test_client: TestClient):
        """Single frame: 20484 vertices × 4 bytes RGBA = 81936."""
        r = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        assert len(r.content) == 81936

    def test_x_headers_populated(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert "x-window-id" in r.headers
        assert "x-tribe-method" in r.headers
        assert "x-vmin" in r.headers
        assert "x-vmax" in r.headers
        assert "x-frame-count" in r.headers
        assert r.headers["x-frame-count"] == "1"

    def test_x_vmin_vmax_defaults(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert float(r.headers["x-vmin"]) == -3.0
        assert float(r.headers["x-vmax"]) == 3.0

    def test_missing_image_returns_422(self, test_client: TestClient):
        """Multipart with no image field → FastAPI 422 Unprocessable Entity."""
        r = test_client.post(
            "/v1/inference/colors",
            data={"method": "window_mean"},
        )
        assert r.status_code == 422

    def test_method_param_echoed(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"method": "peak"},
        )
        assert r.headers["x-tribe-method"] == "peak"

    def test_vmin_vmax_overrides_echoed(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"vmin": "-5.0", "vmax": "5.0"},
        )
        assert float(r.headers["x-vmin"]) == -5.0
        assert float(r.headers["x-vmax"]) == 5.0

    def test_deterministic_same_input(self, test_client: TestClient):
        """Same image bytes → same output bytes (stub is deterministic)."""
        img = _jpeg_bytes()
        r1 = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", img, "image/jpeg")},
        )
        r2 = test_client.post(
            "/v1/inference/colors",
            files={"image": ("test.jpg", img, "image/jpeg")},
        )
        assert r1.content == r2.content


# ---------------------------------------------------------------------------
# POST /v1/inference/full + GET /v1/inference/colors/{window_id}
# ---------------------------------------------------------------------------

class TestInferenceFull:
    def test_returns_json_envelope(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/full",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "window_id" in body
        assert "colors_url" in body
        assert "report" in body
        assert "vmin" in body
        assert "vmax" in body
        assert "method" in body

    def test_colors_url_format(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/full",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        body = r.json()
        wid = body["window_id"]
        assert body["colors_url"] == f"/v1/inference/colors/{wid}"

    def test_report_has_expected_shape(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/full",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        report = r.json()["report"]
        assert "top_regions" in report
        assert "text" in report
        assert "method" in report
        assert "z_threshold" in report
        assert isinstance(report["top_regions"], list)

    def test_get_cached_colors_returns_80kb_body(self, test_client: TestClient):
        r_full = test_client.post(
            "/v1/inference/full",
            files={"image": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r_full.status_code == 200
        wid = r_full.json()["window_id"]

        r_colors = test_client.get(f"/v1/inference/colors/{wid}")
        assert r_colors.status_code == 200
        # Should be exactly 81936 bytes (20484 × 4 RGBA).
        assert len(r_colors.content) == 81936

    def test_invalid_window_id_returns_404(self, test_client: TestClient):
        r = test_client.get("/v1/inference/colors/nonexistent-window-id-xyz")
        assert r.status_code == 404

    def test_missing_image_full_returns_422(self, test_client: TestClient):
        r = test_client.post(
            "/v1/inference/full",
            data={"method": "window_mean"},
        )
        assert r.status_code == 422
