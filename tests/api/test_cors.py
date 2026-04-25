"""CORS preflight is allowed for the configured origins."""
from __future__ import annotations


def test_options_preflight_includes_allow_origin(test_client):
    r = test_client.options(
        "/v1/health",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette's CORSMiddleware returns 200 for valid preflights.
    assert r.status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}
