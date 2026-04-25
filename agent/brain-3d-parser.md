# 3D Brain Model Parser Unit + Swift UI Integration

## Implementation Plan for TRIBE v2 Cortical Surface Rendering

**Objective:** Build a Python-side parser that takes the raw `(T, 20484)` cortical output from TRIBE v2, maps it onto the fsaverage5 3D brain mesh with activation coloring, and exports a lightweight format that a Swift UI frontend can render in real time.

---

## 1. The 3D Brain Surface: What It Is and Where It Comes From

### 1.1 The fsaverage5 Mesh (From the Paper & Code)

TRIBE v2 predictions live on the **fsaverage5** cortical surface — a standardized brain mesh from FreeSurfer (Fischl et al.). This is the same mesh used in the codebase's `tribev2/plotting/` module (which wraps PyVista and nilearn).

The mesh has:

| Property | Left Hemisphere | Right Hemisphere | Total |
|---|---|---|---|
| **Vertices** | 10,242 | 10,242 | 20,484 |
| **Faces (triangles)** | ~20,480 | ~20,480 | ~40,960 |

Each **vertex** is a 3D point `(x, y, z)` in RAS (Right-Anterior-Superior) coordinate space, representing a location on the folded cortical surface.

Each **face** is a triangle defined by three vertex indices `(v0, v1, v2)`.

The mesh comes in multiple surface types:
- **`pial`** — the outer (gray matter) surface. This is what you see in all the paper's brain figures.
- **`inflated`** — the same mesh but "inflated" to reveal sulci. Useful for visualization but less anatomically realistic.
- **`white`** — the inner (white matter) boundary.

### 1.2 How the Codebase Loads the Mesh

From the TRIBE v2 repository structure:

```
tribev2/
├── plotting/            # Brain visualization (PyVista & Nilearn backends)
├── utils_fmri.py        # Surface projection (MNI / fsaverage) and ROI analysis
├── demo_utils.py        # TribeModel — includes PlotBrain initialization
```

**Reference:** `https://github.com/facebookresearch/tribev2`

The demo notebook initializes a `PlotBrain` object for 3D surface visualization using the fsaverage5 mesh. Under the hood, this calls:

```python
# How nilearn loads the fsaverage5 mesh:
from nilearn.datasets import fetch_surf_fsaverage

fsaverage = fetch_surf_fsaverage('fsaverage5')

# fsaverage is a dict with keys:
# 'pial_left'    → path to left hemisphere pial surface file
# 'pial_right'   → path to right hemisphere pial surface file
# 'infl_left'    → path to left hemisphere inflated surface
# 'infl_right'   → path to right hemisphere inflated surface
# 'white_left'   → path to left hemisphere white matter surface
# 'white_right'  → path to right hemisphere white matter surface
# 'sulc_left'    → path to left sulcal depth map
# 'sulc_right'   → path to right sulcal depth map
```

Each surface file is in **FreeSurfer binary format**, which nibabel can parse:

```python
import nibabel as nib

coords, faces = nib.freesurfer.read_geometry(fsaverage['pial_left'])
# coords: np.ndarray of shape (10242, 3) — vertex positions in mm
# faces:  np.ndarray of shape (~20480, 3) — triangle indices (int32)
```

### 1.3 How Activation Values Map to the Mesh

TRIBE v2's output is `(T, 20484)` where:
- Indices `[0:10242]` → **left hemisphere** vertices
- Indices `[10242:20484]` → **right hemisphere** vertices

This is a 1:1 mapping: output index `i` is the predicted BOLD z-score at vertex `i` of the fsaverage5 mesh. The codebase's `PlotBrain` assigns these as **per-vertex scalar data**, which PyVista/nilearn then color-maps onto the mesh surface.

### 1.4 The Visualization Pipeline in the Codebase

The `tribev2/plotting/` module (PyVista backend) does roughly:

```python
import pyvista as pv

# 1. Build PyVista mesh from FreeSurfer geometry
mesh_lh = pv.PolyData(coords_lh, np.column_stack([
    np.full(len(faces_lh), 3), faces_lh
]))

# 2. Assign per-vertex activation scalars
mesh_lh.point_data['activation'] = activations[0:10242]

# 3. Render with colormap
plotter = pv.Plotter()
plotter.add_mesh(mesh_lh, scalars='activation', cmap='RdBu_r',
                 clim=[-3, 3], smooth_shading=True)
```

The nilearn backend uses `nilearn.plotting.plot_surf_stat_map()` which does the same thing but renders to matplotlib.

---

## 2. The 3D Model Parser Unit

This unit sits between TRIBE v2 inference and the Swift frontend. It:
1. Loads the fsaverage5 mesh geometry (once, cached)
2. Takes a `(T, 20484)` activation tensor
3. Applies temporal aggregation (mean, peak, or per-frame)
4. Maps activation values to per-vertex RGBA colors
5. Exports as a format Swift can consume

### 2.1 Export Format: GLB (recommended)

**Why GLB over USDZ:**
- GLB (binary glTF) supports **per-vertex colors** natively
- Can be generated from Python without Apple tooling
- SceneKit can load GLB via `SCNScene(named:)` or ModelI/O
- Lightweight — the fsaverage5 brain is ~2MB as GLB
- Can be updated frame-by-frame for animation by swapping vertex color buffers

**Alternative: JSON mesh bundle** (for maximum control):
- Export vertices, faces, and colors as raw arrays
- Swift side builds `SCNGeometry` from buffers directly
- More work but gives you full control over rendering

### 2.2 Module: `brain_mesh_exporter.py`

```python
"""
Brain Mesh Exporter for TRIBE v2 → Swift UI

Loads fsaverage5 mesh, maps TRIBE v2 activations to vertex colors,
and exports as GLB or JSON for SceneKit/RealityKit consumption.

References:
  - TRIBE v2 codebase: https://github.com/facebookresearch/tribev2
    - tribev2/plotting/ (PyVista & nilearn backends)
    - tribev2/utils_fmri.py (surface projection utilities)
    - tribev2/demo_utils.py (PlotBrain initialization)
  - Paper: d'Ascoli et al., 2026, Section 5.1 (approach overview)
    - Output is 20,484 vertices on fsaverage5 cortical surface
    - Surface extracted via nilearn.surface.vol_to_surf
  - nilearn: https://nilearn.github.io/stable/modules/generated/
             nilearn.datasets.fetch_surf_fsaverage.html
"""

import numpy as np
import nibabel as nib
import json
import struct
from pathlib import Path
from nilearn.datasets import fetch_surf_fsaverage
from matplotlib import cm
from matplotlib.colors import Normalize


class BrainMeshExporter:
    """Converts TRIBE v2 cortical output to 3D renderable formats."""

    def __init__(self, surface_type="pial", cache_dir="./cache"):
        """
        Args:
            surface_type: 'pial' (anatomical), 'inflated' (spread out),
                          or 'white' (inner boundary)
            cache_dir: where to cache the fsaverage5 mesh files
        """
        self.fsaverage = fetch_surf_fsaverage('fsaverage5',
                                              data_dir=cache_dir)

        # Load left hemisphere geometry
        self.coords_lh, self.faces_lh = nib.freesurfer.read_geometry(
            self.fsaverage[f'{surface_type}_left']
        )
        # Load right hemisphere geometry
        self.coords_rh, self.faces_rh = nib.freesurfer.read_geometry(
            self.fsaverage[f'{surface_type}_right']
        )

        # Load sulcal depth for shading (darkens sulci for realism)
        self.sulc_lh = nib.freesurfer.read_morph_data(
            self.fsaverage['sulc_left']
        )
        self.sulc_rh = nib.freesurfer.read_morph_data(
            self.fsaverage['sulc_right']
        )

        # Combined mesh for full brain
        n_lh = len(self.coords_lh)
        self.coords_full = np.vstack([self.coords_lh, self.coords_rh])
        self.faces_full = np.vstack([
            self.faces_lh,
            self.faces_rh + n_lh  # offset RH face indices
        ])
        self.sulc_full = np.concatenate([self.sulc_lh, self.sulc_rh])

    def activation_to_rgba(self, activations, cmap_name='RdBu_r',
                           vmin=-3.0, vmax=3.0, sulc_darken=0.3):
        """
        Map scalar activations to RGBA vertex colors.

        Args:
            activations: (20484,) array of z-scored BOLD predictions
            cmap_name: matplotlib colormap name
                'RdBu_r' = blue (negative) → white (zero) → red (positive)
                'hot' = black → red → yellow → white (unipolar)
            vmin, vmax: clipping range for color normalization
            sulc_darken: how much to darken sulci (0=none, 1=full)

        Returns:
            colors: (20484, 4) uint8 RGBA array
        """
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        colormap = cm.get_cmap(cmap_name)

        # Map activations → RGBA float [0,1]
        normalized = norm(activations)
        rgba = colormap(normalized)  # (20484, 4) float

        # Apply sulcal depth shading for anatomical realism
        # Sulcal depth > 0 = sulcus (should be darker)
        sulc_factor = 1.0 - sulc_darken * np.clip(self.sulc_full, 0, 1)
        rgba[:, :3] *= sulc_factor[:, np.newaxis]

        # Convert to uint8
        return (rgba * 255).astype(np.uint8)

    def aggregate_temporal(self, preds, method='mean', time_idx=None):
        """
        Collapse (T, 20484) → (20484,) for a single-frame export.

        Args:
            preds: (T, 20484) TRIBE v2 output
            method: 'mean', 'peak', 'rms', or 'frame'
            time_idx: specific frame index (used with method='frame')

        Returns:
            (20484,) activation vector
        """
        if method == 'mean':
            return preds.mean(axis=0)
        elif method == 'peak':
            return preds[preds.mean(axis=1).argmax()]
        elif method == 'rms':
            return np.sqrt((preds ** 2).mean(axis=0))
        elif method == 'frame':
            return preds[time_idx]
        else:
            raise ValueError(f"Unknown method: {method}")

    # ─────────────────────────────────────────────────────
    # EXPORT FORMAT 1: JSON Mesh Bundle (for SceneKit)
    # ─────────────────────────────────────────────────────

    def export_json(self, activations, output_path,
                    cmap='RdBu_r', vmin=-3, vmax=3,
                    hemispheres='both', separate=True):
        """
        Export brain mesh + activation colors as a JSON bundle.

        This is the simplest format for Swift to consume:
        - vertices: flat float32 array [x0,y0,z0, x1,y1,z1, ...]
        - normals: flat float32 array (per-vertex normals for lighting)
        - faces: flat uint32 array [i0,i1,i2, i3,i4,i5, ...]
        - colors: flat uint8 array [r0,g0,b0,a0, r1,g1,b1,a1, ...]

        Args:
            activations: (20484,) or (T, 20484)
            output_path: path to write JSON file
            hemispheres: 'both', 'left', or 'right'
            separate: if True, export LH and RH as separate meshes
        """
        if activations.ndim == 2:
            activations = self.aggregate_temporal(activations)

        colors = self.activation_to_rgba(activations, cmap, vmin, vmax)
        normals = self._compute_vertex_normals(
            self.coords_full, self.faces_full
        )

        def mesh_dict(coords, faces, cols, norms):
            return {
                'vertex_count': int(len(coords)),
                'face_count': int(len(faces)),
                'vertices': coords.astype(np.float32).flatten().tolist(),
                'normals': norms.astype(np.float32).flatten().tolist(),
                'faces': faces.astype(np.uint32).flatten().tolist(),
                'colors': cols.flatten().tolist(),  # uint8 RGBA
            }

        n_lh = len(self.coords_lh)
        bundle = {
            'format': 'tribe_brain_mesh_v1',
            'surface': 'fsaverage5',
            'vertex_total': 20484,
            'colormap': cmap,
            'vmin': vmin,
            'vmax': vmax,
        }

        if separate:
            bundle['left_hemisphere'] = mesh_dict(
                self.coords_lh, self.faces_lh,
                colors[:n_lh], normals[:n_lh]
            )
            bundle['right_hemisphere'] = mesh_dict(
                self.coords_rh, self.faces_rh,
                colors[n_lh:], normals[n_lh:]
            )
        else:
            bundle['mesh'] = mesh_dict(
                self.coords_full, self.faces_full,
                colors, normals
            )

        with open(output_path, 'w') as f:
            json.dump(bundle, f)

        return output_path

    # ─────────────────────────────────────────────────────
    # EXPORT FORMAT 2: Binary Buffers (compact, fast load)
    # ─────────────────────────────────────────────────────

    def export_binary(self, activations, output_dir,
                      cmap='RdBu_r', vmin=-3, vmax=3):
        """
        Export as raw binary files that Swift can memory-map.

        Files written:
          brain_vertices.bin  — float32, (N*3,) = [x0,y0,z0,...]
          brain_normals.bin   — float32, (N*3,)
          brain_faces.bin     — uint32,  (F*3,) = [i0,i1,i2,...]
          brain_colors.bin    — uint8,   (N*4,) = [r,g,b,a,...]
          brain_meta.json     — vertex/face counts, colormap info

        This is ~2MB total for the full brain. The mesh files
        (vertices, normals, faces) are static and only need to be
        shipped once. Only brain_colors.bin changes per prediction.
        """
        if activations.ndim == 2:
            activations = self.aggregate_temporal(activations)

        colors = self.activation_to_rgba(activations, cmap, vmin, vmax)
        normals = self._compute_vertex_normals(
            self.coords_full, self.faces_full
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Static mesh geometry (ship once with app)
        self.coords_full.astype(np.float32).tofile(
            out / 'brain_vertices.bin'
        )
        normals.astype(np.float32).tofile(out / 'brain_normals.bin')
        self.faces_full.astype(np.uint32).tofile(out / 'brain_faces.bin')

        # Dynamic activation colors (update per inference)
        colors.tofile(out / 'brain_colors.bin')

        # Metadata
        meta = {
            'vertex_count': int(len(self.coords_full)),
            'face_count': int(len(self.faces_full)),
            'lh_vertex_count': int(len(self.coords_lh)),
            'rh_vertex_count': int(len(self.coords_rh)),
            'bytes_per_vertex': 12,   # 3 × float32
            'bytes_per_normal': 12,   # 3 × float32
            'bytes_per_face': 12,     # 3 × uint32
            'bytes_per_color': 4,     # RGBA uint8
            'colormap': cmap,
            'vmin': vmin,
            'vmax': vmax,
        }
        with open(out / 'brain_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        return out

    # ─────────────────────────────────────────────────────
    # EXPORT FORMAT 3: GLB (binary glTF)
    # ─────────────────────────────────────────────────────

    def export_glb(self, activations, output_path,
                   cmap='RdBu_r', vmin=-3, vmax=3):
        """
        Export as GLB (binary glTF 2.0) for SceneKit / RealityKit.

        Requires: pip install trimesh
        GLB supports per-vertex colors natively and is loadable
        by SceneKit via ModelIO framework.
        """
        import trimesh

        if activations.ndim == 2:
            activations = self.aggregate_temporal(activations)

        colors = self.activation_to_rgba(activations, cmap, vmin, vmax)

        mesh = trimesh.Trimesh(
            vertices=self.coords_full,
            faces=self.faces_full,
            vertex_colors=colors,
            process=False  # keep vertex ordering intact
        )

        mesh.export(output_path, file_type='glb')
        return output_path

    # ─────────────────────────────────────────────────────
    # EXPORT FORMAT 4: Animation sequence (per-second frames)
    # ─────────────────────────────────────────────────────

    def export_animation_bundle(self, preds, output_dir,
                                 cmap='RdBu_r', vmin=-3, vmax=3):
        """
        Export a time series of color buffers for animation.

        The mesh geometry is exported once. Each timestep gets
        a separate color buffer file (brain_colors_t{i}.bin).

        This allows Swift to animate by swapping the vertex color
        source on each frame (at 1 Hz, matching fMRI temporal resolution).

        Args:
            preds: (T, 20484) full temporal output from TRIBE v2
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Export static geometry once
        normals = self._compute_vertex_normals(
            self.coords_full, self.faces_full
        )
        self.coords_full.astype(np.float32).tofile(
            out / 'brain_vertices.bin'
        )
        normals.astype(np.float32).tofile(out / 'brain_normals.bin')
        self.faces_full.astype(np.uint32).tofile(out / 'brain_faces.bin')

        # Export per-frame color buffers
        T = preds.shape[0]
        for t in range(T):
            colors = self.activation_to_rgba(
                preds[t], cmap, vmin, vmax
            )
            colors.tofile(out / f'brain_colors_t{t:04d}.bin')

        meta = {
            'vertex_count': int(len(self.coords_full)),
            'face_count': int(len(self.faces_full)),
            'frame_count': T,
            'fps': 1,  # fMRI temporal resolution
            'color_file_pattern': 'brain_colors_t{frame:04d}.bin',
            'bytes_per_color_frame': int(len(self.coords_full)) * 4,
        }
        with open(out / 'brain_anim_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        return out

    # ─────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_vertex_normals(vertices, faces):
        """Compute per-vertex normals from face normals (area-weighted)."""
        normals = np.zeros_like(vertices)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)

        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        return normals / lengths
```

---

## 3. Swift UI Integration

### 3.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  Python Backend (server or CLI)                          │
│                                                          │
│  TRIBE v2 inference → (T, 20484) predictions             │
│       ↓                                                  │
│  BrainMeshExporter.export_binary()                       │
│       ↓                                                  │
│  brain_vertices.bin (static, ~240KB)                     │
│  brain_normals.bin  (static, ~240KB)                     │
│  brain_faces.bin    (static, ~480KB)                     │
│  brain_colors.bin   (dynamic, ~80KB per frame)           │
│  brain_meta.json                                         │
│  report.json (from GlasserParcellationUnit)              │
└───────────────────────┬──────────────────────────────────┘
                        │  HTTP / file transfer / bundled
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Swift UI Frontend                                       │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  BrainMeshLoader                               │      │
│  │  - Reads binary buffers                        │      │
│  │  - Builds SCNGeometry from raw arrays          │      │
│  │  - Updates vertex colors for animation         │      │
│  └────────────────┬───────────────────────────────┘      │
│                   ↓                                       │
│  ┌────────────────────────────────────────────────┐      │
│  │  BrainSceneView (UIViewRepresentable)          │      │
│  │  - SCNView with orbit camera                   │      │
│  │  - Pinch to zoom, drag to rotate               │      │
│  │  - Tap vertex → show ROI label + report        │      │
│  └────────────────┬───────────────────────────────┘      │
│                   ↓                                       │
│  ┌────────────────────────────────────────────────┐      │
│  │  ReportOverlayView                             │      │
│  │  - Qualitative text from GlasserParcellation   │      │
│  │  - Top-N activated regions list                 │      │
│  │  - Timeline scrubber (for animation mode)      │      │
│  └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Swift: `BrainMeshLoader.swift`

```swift
import SceneKit
import Foundation

/// Loads TRIBE v2 brain mesh exports into SceneKit geometry.
///
/// The binary format matches BrainMeshExporter.export_binary():
///   - brain_vertices.bin: float32 × (N × 3) — vertex positions
///   - brain_normals.bin:  float32 × (N × 3) — vertex normals
///   - brain_faces.bin:    uint32  × (F × 3) — triangle indices
///   - brain_colors.bin:   uint8   × (N × 4) — RGBA per vertex
///   - brain_meta.json:    vertex/face counts
struct BrainMeshLoader {

    struct MeshMeta: Codable {
        let vertex_count: Int
        let face_count: Int
        let lh_vertex_count: Int
        let rh_vertex_count: Int
    }

    /// Build an SCNGeometry from the binary mesh files.
    static func loadBrainMesh(from directory: URL) -> SCNGeometry? {
        guard
            let metaData = try? Data(contentsOf:
                directory.appendingPathComponent("brain_meta.json")),
            let meta = try? JSONDecoder().decode(
                MeshMeta.self, from: metaData)
        else { return nil }

        let vertexData = try? Data(contentsOf:
            directory.appendingPathComponent("brain_vertices.bin"))
        let normalData = try? Data(contentsOf:
            directory.appendingPathComponent("brain_normals.bin"))
        let faceData = try? Data(contentsOf:
            directory.appendingPathComponent("brain_faces.bin"))
        let colorData = try? Data(contentsOf:
            directory.appendingPathComponent("brain_colors.bin"))

        guard let vd = vertexData, let nd = normalData,
              let fd = faceData, let cd = colorData
        else { return nil }

        // Vertex positions
        let vertexSource = SCNGeometrySource(
            data: vd,
            semantic: .vertex,
            vectorCount: meta.vertex_count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: 4,
            dataOffset: 0,
            dataStride: 12
        )

        // Vertex normals
        let normalSource = SCNGeometrySource(
            data: nd,
            semantic: .normal,
            vectorCount: meta.vertex_count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: 4,
            dataOffset: 0,
            dataStride: 12
        )

        // Per-vertex colors (RGBA uint8)
        let colorSource = SCNGeometrySource(
            data: cd,
            semantic: .color,
            vectorCount: meta.vertex_count,
            usesFloatComponents: false,
            componentsPerVector: 4,
            bytesPerComponent: 1,
            dataOffset: 0,
            dataStride: 4
        )

        // Triangle indices
        let element = SCNGeometryElement(
            data: fd,
            primitiveType: .triangles,
            primitiveCount: meta.face_count,
            bytesPerIndex: 4
        )

        let geometry = SCNGeometry(
            sources: [vertexSource, normalSource, colorSource],
            elements: [element]
        )

        // Material: use vertex colors, no texture
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.white
        material.lightingModel = .physicallyBased
        material.metalness.contents = 0.0
        material.roughness.contents = 0.6
        geometry.materials = [material]

        return geometry
    }

    /// Update only the vertex colors (for animation).
    /// This avoids rebuilding the entire geometry.
    static func updateColors(
        on node: SCNNode,
        from colorFileURL: URL,
        vertexCount: Int
    ) {
        guard let colorData = try? Data(contentsOf: colorFileURL)
        else { return }

        let colorSource = SCNGeometrySource(
            data: colorData,
            semantic: .color,
            vectorCount: vertexCount,
            usesFloatComponents: false,
            componentsPerVector: 4,
            bytesPerComponent: 1,
            dataOffset: 0,
            dataStride: 4
        )

        // Rebuild geometry with new colors but same vertices/faces
        if let geometry = node.geometry {
            let vertexSource = geometry.sources(
                for: .vertex).first!
            let normalSource = geometry.sources(
                for: .normal).first!
            let element = geometry.elements.first!

            node.geometry = SCNGeometry(
                sources: [vertexSource, normalSource, colorSource],
                elements: [element]
            )
            node.geometry?.materials = geometry.materials
        }
    }
}
```

### 3.3 Swift: `BrainSceneView.swift` (SwiftUI wrapper)

```swift
import SwiftUI
import SceneKit

/// A SwiftUI view that renders the TRIBE v2 brain mesh.
struct BrainSceneView: UIViewRepresentable {
    let meshDirectory: URL
    @Binding var currentFrame: Int
    let animationFrameCount: Int

    func makeUIView(context: Context) -> SCNView {
        let sceneView = SCNView()
        sceneView.scene = SCNScene()
        sceneView.allowsCameraControl = true  // orbit, zoom, pan
        sceneView.autoenablesDefaultLighting = true
        sceneView.backgroundColor = UIColor.systemBackground

        // Load brain mesh
        if let geometry = BrainMeshLoader.loadBrainMesh(
            from: meshDirectory
        ) {
            let brainNode = SCNNode(geometry: geometry)
            brainNode.name = "brain"

            // Center and scale the mesh
            // fsaverage5 coords are in mm, roughly ±80mm range
            brainNode.scale = SCNVector3(0.01, 0.01, 0.01)
            brainNode.position = SCNVector3(0, 0, 0)

            sceneView.scene?.rootNode.addChildNode(brainNode)
        }

        // Camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 45
        cameraNode.position = SCNVector3(0, 0, 2)
        sceneView.scene?.rootNode.addChildNode(cameraNode)
        sceneView.pointOfView = cameraNode

        return sceneView
    }

    func updateUIView(_ sceneView: SCNView, context: Context) {
        // Update colors when frame changes (animation)
        guard let brainNode = sceneView.scene?.rootNode
            .childNode(withName: "brain", recursively: false)
        else { return }

        let colorFile = meshDirectory.appendingPathComponent(
            String(format: "brain_colors_t%04d.bin", currentFrame)
        )

        BrainMeshLoader.updateColors(
            on: brainNode,
            from: colorFile,
            vertexCount: 20484
        )
    }
}
```

### 3.4 Swift: Main View with Report Overlay

```swift
struct BrainViewerScreen: View {
    @State private var currentFrame: Int = 0
    @State private var isAnimating: Bool = false
    @State private var report: BrainReport?

    let meshDirectory: URL
    let frameCount: Int

    var body: some View {
        ZStack {
            // 3D brain
            BrainSceneView(
                meshDirectory: meshDirectory,
                currentFrame: $currentFrame,
                animationFrameCount: frameCount
            )
            .edgesIgnoringSafeArea(.all)

            // Report overlay
            VStack {
                Spacer()

                if let report = report {
                    ReportCard(report: report)
                        .padding()
                }

                // Timeline scrubber
                HStack {
                    Button(action: { isAnimating.toggle() }) {
                        Image(systemName:
                            isAnimating ? "pause.fill" : "play.fill")
                    }
                    Slider(
                        value: Binding(
                            get: { Double(currentFrame) },
                            set: { currentFrame = Int($0) }
                        ),
                        in: 0...Double(max(frameCount - 1, 1))
                    )
                    Text("t=\(currentFrame)s")
                        .monospacedDigit()
                }
                .padding()
                .background(.ultraThinMaterial)
            }
        }
    }
}
```

---

## 4. Data Flow Summary

```
                    TRIBE v2 Model
                         │
          ┌──────────────┴──────────────┐
          │     (T, 20484) cortical     │
          │     (T, 8802)  subcortical  │
          └──────────────┬──────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  GlasserParcUnit   BrainMeshExporter   (both combined)
         │               │               │
    report.json    binary mesh files    Full package
    (qualitative   (vertices, faces,    for Swift app
     text)          normals, colors)
         │               │
         └───────┬───────┘
                 ▼
         Swift UI Frontend
         ├── 3D SceneKit brain (rotate, zoom)
         ├── Color-coded activations
         ├── Timeline animation (1 Hz)
         └── Report overlay (top regions + descriptions)
```

---

## 5. File Size Estimates

| File | Size | Ships With | Updates Per |
|---|---|---|---|
| `brain_vertices.bin` | 240 KB | App bundle (static) | Never |
| `brain_normals.bin` | 240 KB | App bundle (static) | Never |
| `brain_faces.bin` | 480 KB | App bundle (static) | Never |
| `brain_meta.json` | <1 KB | App bundle (static) | Never |
| `brain_colors.bin` | 80 KB | Per inference | Each prediction |
| `brain_colors_t{N}.bin` | 80 KB × T | Per animation | Each prediction |
| `report.json` | <5 KB | Per inference | Each prediction |
| **Total static** | **~960 KB** | | |
| **Per-frame dynamic** | **~85 KB** | | |

For a 60-second video stimulus, the animation bundle is 960 KB static + 60 × 80 KB colors ≈ **5.7 MB total**. Easily transferable over HTTP or bundled locally.

---

## 6. Dependencies

### Python Side

```
nilearn          # fsaverage5 mesh loading, surface utilities
nibabel          # FreeSurfer geometry I/O
numpy            # Array operations
matplotlib       # Colormaps (cm, Normalize)
trimesh          # GLB export (optional, for Format 3)
torch            # TRIBE v2 model inference
```

### Swift Side

```
SceneKit         # 3D rendering (built into iOS/macOS)
SwiftUI          # UI framework
Foundation       # File I/O, JSON decoding
# Optional:
RealityKit       # Alternative renderer (visionOS/AR)
ModelIO          # For GLB import path
```

---

## 7. References

- **TRIBE v2 Code:** `https://github.com/facebookresearch/tribev2`
  - `tribev2/plotting/` — PyVista & nilearn brain visualization backends
  - `tribev2/utils_fmri.py` — Surface projection, ROI analysis
  - `tribev2/demo_utils.py` — `TribeModel` class, `PlotBrain` initialization
  - `tribe_demo.ipynb` — Full walkthrough with brain visualizations
- **TRIBE v2 Paper:** d'Ascoli et al. (2026), Section 5.1 & 5.8
  - Output: 20,484 fsaverage5 vertices + 8,802 subcortical voxels
  - Surface extraction: `nilearn.surface.vol_to_surf` with "ball" sampling, 3mm radius
  - Plotting deps: `pip install -e ".[plotting]"` → nibabel, pyvista, nilearn, scipy
- **fsaverage5 mesh:** `nilearn.datasets.fetch_surf_fsaverage('fsaverage5')`
  - 10,242 nodes per hemisphere, FreeSurfer binary surface format
  - Nilearn docs: `https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_surf_fsaverage.html`
- **FreeSurfer surface format:** `http://www.grahamwideman.com/gw/brain/fs/surfacefileformats.htm`
- **Glasser Parcellation:** Glasser et al. (2016), Nature 536(7615), 171-178
- **Previous plan:** `glasser_parcellation_unit_plan.md` (ROI interpretation layer)
