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


def test_options_preflight_for_mesh_endpoints(test_client):
    """Phase C: the mesh artifact GETs need CORS for the browser renderer.

    We don't need a real job_id for an OPTIONS preflight — Starlette's
    CORSMiddleware short-circuits before the handler runs. This just
    verifies the same global middleware permits these paths.
    """
    for path in (
        "/v1/runs/abc/mesh",
        "/v1/runs/abc/mesh/meta",
        "/v1/runs/abc/mesh/colors",
        "/v1/runs/abc/mesh/vertices",
        "/v1/runs/abc/mesh/faces",
    ):
        r = test_client.options(
            path,
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code == 200, (path, r.status_code)
        assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}
