"""BrainMeshExporter — export TribeOutput as 3D mesh artifacts.

Pure functions:
    aggregate_temporal(cortical, method) -> (V,) float32

Class:
    BrainMeshExporter
        export_json(output, out_dir)        -> Path
        export_binary(output, out_dir)      -> Path  (writes brain_meta.json,
                                                       brain_vertices.bin,
                                                       brain_faces.bin,
                                                       brain_colors.bin)
        export_glb(output, out_dir)         -> Path  (requires trimesh)
        export_animation_bundle(cortical,
                                out_dir)    -> Path  (T per-frame color files
                                                       + geometry once)

I/O lives only in this file. All math (colormap, aggregation, normals) is
delegated to pure helpers in colormap.py / geometry.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from tribe_backend.contracts import CORTICAL_VERTICES, TribeOutput
from tribe_backend.mesh.colormap import activation_to_rgba
from tribe_backend.mesh.geometry import _compute_vertex_normals, load_fsaverage5

MESH_FORMAT_VERSION = "tribe_brain_mesh_v1"

AggregateMethod = Literal["mean", "peak", "max"]


def aggregate_temporal(cortical: np.ndarray, method: AggregateMethod = "mean") -> np.ndarray:
    """Reduce a (T, V) cortical tensor to a (V,) per-vertex summary.

    Methods:
        "mean" — np.mean over T (matches output.cortical.mean(axis=0)).
        "peak" — value at the timestep with largest absolute value, per vertex.
        "max"  — np.max over T.
    """
    arr = np.asarray(cortical)
    if arr.ndim != 2 or arr.shape[1] != CORTICAL_VERTICES:
        raise ValueError(
            f"cortical must be 2D with shape (T, {CORTICAL_VERTICES}); got {arr.shape}"
        )
    if method == "mean":
        return arr.mean(axis=0).astype(np.float32)
    if method == "peak":
        peak_idx = np.argmax(np.abs(arr), axis=0)  # (V,)
        cols = np.arange(arr.shape[1])
        return arr[peak_idx, cols].astype(np.float32)
    if method == "max":
        return arr.max(axis=0).astype(np.float32)
    raise ValueError(
        f"unknown method {method!r}; expected one of 'mean', 'peak', 'max'"
    )


@dataclass
class BrainMeshExporter:
    """Export TribeOutput → 3D artifacts.

    Parameters
    ----------
    surface_type : fsaverage5 surface variant ("pial", "infl", ...).
    cmap         : matplotlib colormap name.
    vmin, vmax   : symmetric clipping range for activations.
    method       : default temporal aggregation method for single-frame exports.
    """

    surface_type: str = "pial"
    cmap: str = "RdBu_r"
    vmin: float = -3.0
    vmax: float = 3.0
    method: AggregateMethod = "mean"

    # ---- internal -----------------------------------------------------------

    def _geometry(self) -> tuple[np.ndarray, np.ndarray]:
        return load_fsaverage5(self.surface_type)

    def _colors_for(self, per_vertex: np.ndarray) -> np.ndarray:
        return activation_to_rgba(
            per_vertex, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
        )

    # ---- exports ------------------------------------------------------------

    def export_json(self, output: TribeOutput, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        coords, faces = self._geometry()
        per_vertex = aggregate_temporal(output.cortical, method=self.method)
        rgba = self._colors_for(per_vertex)
        path = out_dir / "brain_mesh.json"
        payload = {
            "format": MESH_FORMAT_VERSION,
            "surface": output.surface,
            "surface_type": self.surface_type,
            "vertex_count": int(coords.shape[0]),
            "face_count": int(faces.shape[0]),
            "vertices": coords.astype(np.float32).tolist(),
            "faces": faces.astype(np.int32).tolist(),
            "colors": rgba.astype(np.uint8).tolist(),
            "window_id": output.window_id,
        }
        path.write_text(json.dumps(payload))
        return path

    def export_binary(self, output: TribeOutput, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        coords, faces = self._geometry()
        per_vertex = aggregate_temporal(output.cortical, method=self.method)
        rgba = self._colors_for(per_vertex)

        verts_bin = out_dir / "brain_vertices.bin"
        faces_bin = out_dir / "brain_faces.bin"
        colors_bin = out_dir / "brain_colors.bin"

        coords.astype(np.float32, copy=False).tofile(verts_bin)
        faces.astype(np.int32, copy=False).tofile(faces_bin)
        rgba.astype(np.uint8, copy=False).tofile(colors_bin)

        meta = {
            "format": MESH_FORMAT_VERSION,
            "surface": output.surface,
            "surface_type": self.surface_type,
            "vertex_count": int(coords.shape[0]),
            "face_count": int(faces.shape[0]),
            "bytes_per_vertex": 12,  # 3 * float32
            "bytes_per_face": 12,    # 3 * int32
            "bytes_per_color": 4,    # rgba uint8
            "files": {
                "vertices": verts_bin.name,
                "faces": faces_bin.name,
                "colors": colors_bin.name,
            },
            "window_id": output.window_id,
        }
        meta_path = out_dir / "brain_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        return out_dir

    def export_animation_bundle(
        self,
        cortical: np.ndarray,
        out_dir: str | Path,
        *,
        prefix: str = "brain_colors_t",
    ) -> Path:
        """Write per-frame color bin files + geometry once.

        Filenames: ``{prefix}{frame_index:04d}.bin`` for each timestep.
        Geometry (vertices + faces) is written exactly once alongside.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(cortical)
        if arr.ndim != 2 or arr.shape[1] != CORTICAL_VERTICES:
            raise ValueError(
                f"cortical must be (T, {CORTICAL_VERTICES}); got {arr.shape}"
            )
        coords, faces = self._geometry()
        coords.astype(np.float32, copy=False).tofile(out_dir / "brain_vertices.bin")
        faces.astype(np.int32, copy=False).tofile(out_dir / "brain_faces.bin")

        T = arr.shape[0]
        frame_files: list[str] = []
        for t in range(T):
            rgba = self._colors_for(arr[t])
            name = f"{prefix}{t:04d}.bin"
            (out_dir / name).write_bytes(rgba.astype(np.uint8, copy=False).tobytes())
            frame_files.append(name)

        meta = {
            "format": MESH_FORMAT_VERSION,
            "kind": "animation_bundle",
            "vertex_count": int(coords.shape[0]),
            "frame_count": T,
            "bytes_per_color": 4,
            "frames": frame_files,
        }
        (out_dir / "animation_meta.json").write_text(json.dumps(meta, indent=2))
        return out_dir

    def export_glb(self, output: TribeOutput, out_dir: str | Path) -> Path:
        try:
            import trimesh  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("trimesh is required for GLB export") from exc
        import trimesh

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        coords, faces = self._geometry()
        per_vertex = aggregate_temporal(output.cortical, method=self.method)
        rgba = self._colors_for(per_vertex)
        mesh = trimesh.Trimesh(
            vertices=coords.astype(np.float32),
            faces=faces.astype(np.int32),
            vertex_colors=rgba.astype(np.uint8),
            vertex_normals=_compute_vertex_normals(coords, faces),
            process=False,
        )
        path = out_dir / "brain_mesh.glb"
        mesh.export(path, file_type="glb")
        return path
