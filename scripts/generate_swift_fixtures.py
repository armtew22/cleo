"""Generate deterministic synthetic TRIBE fixtures for Swift loader tests.

Usage:
    python scripts/generate_swift_fixtures.py [--out-dir PATH]

Defaults to: swift/TribeBrainView/Tests/Fixtures

Seed: 42 (np.random.default_rng)
Frames: 31  (cortical data shape: 31 x 20484 float32)

Output layout:
    <out>/static/          -- export_binary output (1 frame aggregated)
    <out>/animation/       -- export_animation_bundle output (31 frames)
    <out>/MANIFEST.json    -- sorted by relpath, SHA-256 + byte size
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on the path when run from any cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tribe_backend.contracts import TribeOutput, CORTICAL_VERTICES, SUBCORTICAL_VOXELS
from tribe_backend.mesh.exporter import BrainMeshExporter


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _build_output(rng: np.random.Generator) -> TribeOutput:
    T = 31
    cortical = rng.standard_normal((T, CORTICAL_VERTICES)).astype(np.float32)
    subcortical = np.zeros((T, SUBCORTICAL_VOXELS), dtype=np.float32)
    return TribeOutput(
        cortical=cortical,
        subcortical=subcortical,
        fps=10.0,
        surface="fsaverage5",
        window_id="fixture-seed42",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Swift fixture files.")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "swift" / "TribeBrainView" / "Tests" / "Fixtures"),
        help="Output directory (default: swift/TribeBrainView/Tests/Fixtures)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()

    # Idempotent: wipe and recreate
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    output = _build_output(rng)

    exporter = BrainMeshExporter(surface_type="pial", cmap="RdBu_r", vmin=-3.0, vmax=3.0)

    static_dir = out_dir / "static"
    exporter.export_binary(output, static_dir)

    anim_dir = out_dir / "animation"
    exporter.export_animation_bundle(output.cortical, anim_dir)

    # Build MANIFEST.json
    manifest: list[dict] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            relpath = path.relative_to(out_dir).as_posix()
            manifest.append(
                {
                    "relpath": relpath,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Fixtures written to: {out_dir}")
    print(f"Files: {len(manifest)}")
    total_bytes = sum(e["bytes"] for e in manifest)
    print(f"Total size: {total_bytes / 1_048_576:.2f} MB")

    # Quick sanity check
    meta_path = static_dir / "brain_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["vertex_count"] == CORTICAL_VERTICES, (
        f"vertex_count mismatch: {meta['vertex_count']} != {CORTICAL_VERTICES}"
    )
    print(f"brain_meta.json vertex_count={meta['vertex_count']} OK")


if __name__ == "__main__":
    main()
