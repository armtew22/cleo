"""Quick test: sample frames from sample_video.mp4 and produce one holistic caption.

Usage:
    python test_captions.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from cleo.captioning import caption_clip

load_dotenv()

VIDEO = Path(__file__).parent / "sample_video.mp4"
OUT_DIR = Path(__file__).parent / "out" / "test_frames"


def main() -> None:
    if not VIDEO.exists():
        raise SystemExit(f"missing video: {VIDEO}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"captioning {VIDEO.name} ...")
    caption = caption_clip(VIDEO)

    print("\n--- caption ---")
    print(caption)

    log_path = OUT_DIR / "caption.md"
    log_path.write_text(f"# caption for {VIDEO.name}\n\n{caption}\n")
    print(f"\nwrote caption -> {log_path}")


if __name__ == "__main__":
    main()
