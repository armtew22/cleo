"""Error response envelope shape."""
from __future__ import annotations

from tribe_backend.api.errors import error_body


def test_error_body_shape():
    body = error_body(code="decode_error", message="not an mp4")
    assert body == {"error": {"code": "decode_error", "message": "not an mp4"}}
