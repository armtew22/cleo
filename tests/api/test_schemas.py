"""Pydantic schemas for the API surface."""
from __future__ import annotations

from tribe_backend.api.schemas import ReportResponse, RegionActivationResponse
from tribe_backend.parcellation.unit import Report, RegionActivation


def test_report_response_round_trips_dataclass():
    """ReportResponse must accept Report.to_json() and round-trip equal."""
    import json

    r = Report(
        top_regions=[
            RegionActivation(name="L_V1", z_score=3.42, direction="activation",
                             description="Primary visual cortex."),
            RegionActivation(name="R_FFC", z_score=-2.1, direction="deactivation",
                             description=""),
        ],
        text="Window summary: ...",
        method="window_mean",
        z_threshold=1.5,
        window_id="abc-123",
    )
    payload = json.loads(r.to_json())
    rr = ReportResponse.model_validate(payload)
    # Re-dump and confirm shape match
    redumped = rr.model_dump()
    assert redumped["window_id"] == "abc-123"
    assert redumped["method"] == "window_mean"
    assert redumped["z_threshold"] == 1.5
    assert redumped["text"] == "Window summary: ..."
    assert len(redumped["top_regions"]) == 2
    assert redumped["top_regions"][0]["name"] == "L_V1"
    assert redumped["top_regions"][0]["z_score"] == 3.42
    assert redumped["top_regions"][0]["direction"] == "activation"


def test_report_response_empty_regions():
    rr = ReportResponse(
        top_regions=[],
        text="None.",
        method="window_mean",
        z_threshold=1.5,
        window_id=None,
    )
    assert rr.top_regions == []
    assert rr.window_id is None


def test_region_activation_minimal():
    ra = RegionActivationResponse(name="x", z_score=1.0, direction="activation")
    # description defaults to ""
    assert ra.description == ""
