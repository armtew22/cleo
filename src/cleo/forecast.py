"""CLI: run the Cleo sensory-load forecast on a single clip.

Usage:
    python -m cleo.forecast --video clip.mp4 --out forecast.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cleo.pipeline import run_forecast


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleo TRIBE v2 sensory-load forecast")
    parser.add_argument("--video", required=True, type=Path, help="path to a 30s mp4 clip")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("forecast.json"),
        help="where to write the JSON forecast (default: ./forecast.json)",
    )
    args = parser.parse_args()

    forecast = run_forecast(args.video)
    args.out.write_text(json.dumps(forecast, indent=2), encoding="utf-8")

    ranking = forecast["ranking_by_sustained_z"]
    top = ranking[0]
    z = forecast["roi_stats"][top]["z_sustained"]
    print(f"wrote {args.out}")
    print(f"top ROI by sustained z: {top} (z={z:+.2f})")
    print(f"caption: {forecast['caption']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
