"""CLI: run the Cleo sensory-load forecast on a single clip.

Usage
─────
  cleo-forecast --video clip.mp4 --out forecast.json

Writes:
  forecast.json   — full pipeline output (caption + Glasser top regions + feedback)
  forecast.md     — caregiver markdown sidecar (when feedback succeeds)

Backend selection: by default we POST to the remote TRIBE service at
``$TRIBE_BACKEND_URL`` (default ellis-compute-02:8004). Use ``--tribe-url`` or
the env var to point at a different host. A real GPU forward pass takes ~5 min.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from cleo.pipeline import run_forecast
from cleo.tribe_client import DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT

load_dotenv()


def _print_stage(stage: str, info: dict) -> None:
    extra = ""
    if "video" in info:
        extra = f" — {Path(info['video']).name}"
    elif "top_group" in info:
        extra = f" — top group: {info['top_group']}"
    elif "reason" in info:
        extra = f" — reason: {info['reason']}"
    print(f"[stage] {stage}{extra}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleo TRIBE v2 sensory-load forecast")
    parser.add_argument("--video", required=True, type=Path, help="path to a ≈30 s mp4 clip")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("forecast.json"),
        help="JSON output path (default: ./forecast.json)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="caregiver markdown sidecar (default: <out>.md)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="skip Claude caregiver brief",
    )
    parser.add_argument(
        "--tribe-url",
        type=str,
        default=None,
        help="override TRIBE backend URL (default reads $TRIBE_BACKEND_URL)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"seconds between TRIBE polls (default {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"per-job timeout in seconds (default {DEFAULT_TIMEOUT})",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress stage updates")
    args = parser.parse_args()

    forecast = run_forecast(
        args.video,
        include_feedback=not args.no_feedback,
        tribe_url=args.tribe_url,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        on_status=None if args.quiet else _print_stage,
    )
    args.out.write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    md_path = args.md_out if args.md_out is not None else args.out.with_suffix(".md")
    fb = forecast.get("feedback")
    if fb and isinstance(fb, dict) and fb.get("markdown"):
        md_path.write_text(fb["markdown"], encoding="utf-8")
        print(f"wrote {md_path}")
    elif "feedback_error" in forecast:
        print(f"feedback failed: {forecast['feedback_error']}", file=sys.stderr)

    ranking = forecast.get("ranking_by_sustained_z") or []
    if ranking:
        top = ranking[0]
        z = forecast["roi_stats"][top]["z_sustained"]
        print(f"top group: {top} (z={z:+.2f})")
    n_regions = len(forecast.get("top_regions") or [])
    print(f"upstream regions returned: {n_regions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
