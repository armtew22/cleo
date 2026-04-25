"""Objective scene captioning from a video clip.

Samples N evenly spaced frames via ffmpeg and asks Claude (vision) for a single
factual paragraph describing the space. The output paragraph is the *third*
TRIBE input alongside the raw video and audio — it acts as a scene-description
track. We deliberately strip emotional/judgmental language so the text encoder
sees factual content only.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

CAPTION_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You write strictly factual, objective scene descriptions of physical spaces.

Rules:
- Describe only what is visible or audible: counts, materials, lighting, colors, density of people, audible sounds, motion.
- Use neutral concrete nouns. No adjectives that imply emotion or judgment ("overwhelming", "cozy", "chaotic", "pleasant", "loud and stressful").
- Quantify when possible ("approximately 15 people", "two ceiling-mounted speakers").
- One paragraph, 3-6 sentences. No headings, no lists, no bullet points.
- Do not speculate about purpose, mood, or how the space "feels". Stick to observables.
"""

USER_INSTRUCTION = (
    "These frames are sampled from a 30-second clip of a physical space. "
    "Write one paragraph describing the space using only objective observables. "
    "Cover lighting, surfaces, density and arrangement of people, visible motion, "
    "and any audio cues you can infer from visual evidence (e.g., visible speakers, "
    "musicians, machinery)."
)


def probe_duration_seconds(video_path: Path) -> float:
    """Use ffprobe to get clip duration in seconds."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]
    )
    return float(json.loads(out)["format"]["duration"])


def sample_frames(video_path: Path, n: int, out_dir: Path) -> list[Path]:
    """Extract n evenly spaced JPEG frames from the clip into out_dir."""
    duration = probe_duration_seconds(video_path)
    if duration <= 0:
        raise RuntimeError(f"clip {video_path} has zero duration")

    # avoid the very last frame (some encoders return blank); sample at midpoints
    timestamps = [duration * (i + 0.5) / n for i in range(n)]

    paths: list[Path] = []
    for i, ts in enumerate(timestamps):
        frame_path = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",                # overwrite
                "-ss", f"{ts:.3f}",  # seek before input — fast, ~accurate
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "3",         # decent quality, small file
                "-loglevel", "error",
                str(frame_path),
            ],
            check=True,
        )
        paths.append(frame_path)
    return paths


def caption_clip(video_path: Path, n_frames: int = 5) -> str:
    """Return a single objective paragraph describing the scene."""
    import anthropic  # lazy: keeps pipeline math importable without the SDK

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        frame_paths = sample_frames(video_path, n_frames, td_path)

        image_blocks = []
        for fp in frame_paths:
            data = base64.standard_b64encode(fp.read_bytes()).decode("ascii")
            image_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": data,
                },
            })

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=CAPTION_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [*image_blocks, {"type": "text", "text": USER_INSTRUCTION}],
                }
            ],
        )

    text_blocks = [b.text for b in msg.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"no text in caption response: {msg}")
    return "\n".join(text_blocks).strip()
