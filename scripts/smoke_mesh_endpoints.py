#!/usr/bin/env python3
"""
smoke_mesh_endpoints.py — Linux-runnable smoke test for the FastAPI mesh routes.

Exercises the same code paths a Swift BrainMeshClient would follow:
  1. GET /v1/mesh/static  → 200 + tar containing expected files
  2. POST /v1/inference/colors with a 3-byte stub image → 200 + 81936-byte body + X-* headers
  3. POST /v1/inference/full  → JSON envelope shape + colors_url GET → 81936 bytes
  4. GET /v1/mesh/static with If-None-Match → 304

Usage:
    python scripts/smoke_mesh_endpoints.py [BASE_URL]

    BASE_URL defaults to http://127.0.0.1:18000
"""
from __future__ import annotations

import io
import sys
import tarfile

import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:18000"

COLORS_FRAME_BYTES = 20484 * 4  # 81936

_PASS = "PASS"
_FAIL = "FAIL"
_results: list[tuple[str, str]] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = _PASS if condition else _FAIL
    _results.append((name, status))
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    if not condition:
        raise SystemExit(f"\nSmoke failed at: {name}")


# ---------------------------------------------------------------------------
# 1. GET /v1/mesh/static
# ---------------------------------------------------------------------------
print("\n--- 1. GET /v1/mesh/static ---")
r = requests.get(f"{BASE_URL}/v1/mesh/static")
_check("status 200", r.status_code == 200, f"got {r.status_code}")
_check("ETag header present", "etag" in {k.lower() for k in r.headers})
_check("Cache-Control: immutable", "immutable" in r.headers.get("Cache-Control", ""))

# Unpack tar and verify expected members.
with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:*") as tf:
    names = tf.getnames()
_check(
    "tar contains brain_vertices.bin",
    "brain_vertices.bin" in names,
    str(names),
)
_check("tar contains brain_normals.bin", "brain_normals.bin" in names, str(names))
_check("tar contains brain_faces.bin", "brain_faces.bin" in names, str(names))
_check("tar contains brain_meta.json", "brain_meta.json" in names, str(names))

etag = r.headers.get("ETag") or r.headers.get("etag")

# ---------------------------------------------------------------------------
# 2. POST /v1/inference/colors
# ---------------------------------------------------------------------------
print("\n--- 2. POST /v1/inference/colors ---")
stub_image = b"\xff\xd8\xff"   # minimal 3-byte JPEG-looking stub
files = {"image": ("frame.jpg", io.BytesIO(stub_image), "image/jpeg")}
r2 = requests.post(f"{BASE_URL}/v1/inference/colors", files=files)

_check("status 200", r2.status_code == 200, f"got {r2.status_code}")
_check(
    "body exactly 81936 bytes",
    len(r2.content) == COLORS_FRAME_BYTES,
    f"got {len(r2.content)}",
)
_check("X-Window-Id present", "X-Window-Id" in r2.headers)
_check("X-Tribe-Method present", "X-Tribe-Method" in r2.headers)
_check("X-Vmin present", "X-Vmin" in r2.headers)
_check("X-Vmax present", "X-Vmax" in r2.headers)
_check("X-Frame-Count present", "X-Frame-Count" in r2.headers)
_check(
    "X-Frame-Count == 1",
    r2.headers.get("X-Frame-Count") == "1",
    f"got {r2.headers.get('X-Frame-Count')}",
)

# ---------------------------------------------------------------------------
# 3. POST /v1/inference/full + colors_url GET
# ---------------------------------------------------------------------------
print("\n--- 3. POST /v1/inference/full ---")
files3 = {"image": ("frame.jpg", io.BytesIO(stub_image), "image/jpeg")}
r3 = requests.post(f"{BASE_URL}/v1/inference/full", files=files3)
_check("status 200", r3.status_code == 200, f"got {r3.status_code}")

envelope = r3.json()
_check("envelope has window_id", "window_id" in envelope)
_check("envelope has colors_url", "colors_url" in envelope)
_check("envelope has report", "report" in envelope)
_check("envelope has vmin", "vmin" in envelope)
_check("envelope has vmax", "vmax" in envelope)
_check("envelope has method", "method" in envelope)
_check(
    "report has top_regions",
    "top_regions" in envelope.get("report", {}),
    str(list(envelope.get("report", {}).keys())),
)

colors_path = envelope["colors_url"]
colors_abs = f"{BASE_URL}{colors_path}" if colors_path.startswith("/") else colors_path
r3c = requests.get(colors_abs)
_check(
    "colors_url GET 200",
    r3c.status_code == 200,
    f"got {r3c.status_code}",
)
_check(
    "colors_url body 81936 bytes",
    len(r3c.content) == COLORS_FRAME_BYTES,
    f"got {len(r3c.content)}",
)

# ---------------------------------------------------------------------------
# 4. GET /v1/mesh/static with If-None-Match → 304
# ---------------------------------------------------------------------------
print("\n--- 4. GET /v1/mesh/static with If-None-Match ---")
r4 = requests.get(f"{BASE_URL}/v1/mesh/static", headers={"If-None-Match": etag})
_check("304 on matching ETag", r4.status_code == 304, f"got {r4.status_code}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Smoke summary ===")
passed = sum(1 for _, s in _results if s == _PASS)
failed = sum(1 for _, s in _results if s == _FAIL)
print(f"Total checks : {len(_results)}")
print(f"Passed       : {passed}")
print(f"Failed       : {failed}")
if failed == 0:
    print("Result       : ALL PASS")
else:
    print("Result       : FAILED")
    raise SystemExit(1)
