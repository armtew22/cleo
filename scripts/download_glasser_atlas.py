#!/usr/bin/env python3
"""Download the standard Glasser-360 (HCP-MMP1.0) atlas projected onto fsaverage,
then subsample to fsaverage5 and write the .annot files the parcellation loader
expects at <repo>/data/atlases/glasser/.

Source: figshare "HCP-MMP1.0 projected on fsaverage" (Mills, 2016).
- LH: https://ndownloader.figshare.com/files/5528816
- RH: https://ndownloader.figshare.com/files/5528819

The icosahedron hierarchy guarantees fsaverage5 vertex i corresponds to
fsaverage vertex i for i < 10242, so subsampling is just labels[:10242].

Idempotent: skips download if .fsaverage source files already exist; always
re-writes the fsaverage5 .annot files.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "atlases" / "glasser"
SOURCES = {
    "lh": "https://ndownloader.figshare.com/files/5528816",
    "rh": "https://ndownloader.figshare.com/files/5528819",
}
FS5_VERTS_PER_HEMI = 10242


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for hemi, url in SOURCES.items():
        src = OUT_DIR / f"{hemi}.HCP-MMP1.annot.fsaverage"
        dst = OUT_DIR / f"{hemi}.HCP-MMP1.annot"
        if not src.exists():
            print(f"[{hemi}] downloading {url} -> {src}")
            urllib.request.urlretrieve(url, src)
        else:
            print(f"[{hemi}] using cached source {src}")

        labels, ctab, names = nib.freesurfer.read_annot(str(src))
        if labels.shape[0] != 163842:
            raise RuntimeError(
                f"[{hemi}] expected fsaverage hemi (163842 vertices); got {labels.shape[0]}"
            )
        fs5_labels = np.asarray(labels[:FS5_VERTS_PER_HEMI], dtype=np.int32)
        nib.freesurfer.write_annot(str(dst), fs5_labels, ctab, names)
        print(
            f"[{hemi}] wrote {dst}  shape={fs5_labels.shape}  "
            f"unique_parcels={len(np.unique(fs5_labels))}  max_id={int(fs5_labels.max())}"
        )

    print("\nDone. The parcellation loader will pick these up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
