# tribev2 Render Survey — Second Pass

**Date**: 2026-04-25  
**Scope**: All plotting/, utils_fmri.py, demo_utils.py, utils.py (ROI section), tribe_demo.ipynb, pyproject.toml  
**License**: CC BY-NC 4.0 (Meta Platforms, Inc.) — non-commercial only

---

## A. Reusable Mesh / Surface Plumbing

### A1. `BasePlotBrain.get_mesh()` — full quote

File: `tribev2/tribev2/plotting/base.py`, lines 127–172

```python
def get_mesh(self) -> dict:
    """Load mesh geometry and background maps for both hemispheres.

    Returns a dict with keys ``'left'``, ``'right'``, ``'both'``,
    each mapping to ``{'coords': array, 'faces': array, 'bg_map': array}``.
    The ``'both'`` entry has hemisphere_gap applied.
    """
    fs_out = cached_fetch_surf_fsaverage(self.mesh)

    out = {}
    for hemi in ("left", "right"):
        infl_out_xyz, _ = nib.load(getattr(fs_out, f"infl_{hemi}")).darrays
        pial_xyz, faces = nib.load(getattr(fs_out, f"pial_{hemi}")).darrays

        alpha = 0.5
        jr_xyz = infl_out_xyz.data * alpha + (1 - alpha) * pial_xyz.data
        if self.inflate == "half":
            coords = jr_xyz
        elif self.inflate is True:
            coords = infl_out_xyz.data
        elif self.inflate is False:
            coords = pial_xyz.data

        bg_key = "curv" if self.bg_map == "curvature" else "sulc"
        bg_map = nib.load(getattr(fs_out, f"{bg_key}_{hemi}")).darrays[0].data
        if self.bg_map == "thresholded":
            bg_map = 1.0 * (bg_map > -0.10)
            bg_map[-1] = -5
            bg_map[-2] = 2.0
        if hemi == "left":
            coords[:, 0] = coords[:, 0] - coords[:, 0].max() - self.hemisphere_gap
        else:
            coords[:, 0] = coords[:, 0] - coords[:, 0].min() + self.hemisphere_gap

        out[hemi] = dict(coords=coords, faces=faces.data, bg_map=bg_map)

    out["both"] = dict(
        coords=np.r_[out["left"]["coords"], out["right"]["coords"]],
        faces=np.r_[
            out["left"]["faces"],
            out["right"]["faces"] + out["left"]["faces"].max() + 1,
        ],
        bg_map=np.r_[out["left"]["bg_map"], out["right"]["bg_map"]],
    )

    return out
```

**Surface types supported**: The `inflate` field controls which geometry is loaded:
- `inflate=False` → **pial** surface (`pial_left` / `pial_right` .gii)
- `inflate=True` → **inflated** surface (`infl_left` / `infl_right`)
- `inflate="half"` (default) → **50/50 linear blend** of inflated + pial (the `jr_xyz` blend)

There is no `white` surface path here; the white surface appears only in `TribeSurfaceProjector.get_intermediate_mesh()` (see A10).

**Return format**: `coords` is `(N_vertices, 3)` float32, `faces` is `(N_faces, 3)` int32. The `"both"` dict concatenates left + right with face indices offset so they index into the merged vertex array. This is **directly shimmable** onto `BrainMeshExporter`: the `(coords, faces)` pair maps 1:1 to a `SCNGeometry` or a glTF `POSITION` + `indices` buffer. The `bg_map` (sulcal depth) is a `(N_vertices,)` float array ready to drive per-vertex shading.

### A2. `get_stat_map()` — kNN resampling between fsaverage resolutions

File: `tribev2/tribev2/plotting/base.py`, lines 178–215

```python
def get_stat_map(self, data: np.ndarray) -> dict:
    """Split vertex data into hemispheres, upsampling if needed.

    Returns ``{'left': array, 'right': array, 'both': array}``.
    """
    in_mesh = None
    for name, size in FSAVERAGE_SIZES.items():
        if data.shape[0] // 2 == size:
            in_mesh = name
            break
    if in_mesh is None:
        raise ValueError(f"Incoherent number of vertices: {data.shape[0]}")

    left = data[: len(data) // 2]
    right = data[len(data) // 2 :]

    if in_mesh != self.mesh:
        fs_in = cached_fetch_surf_fsaverage(in_mesh)
        fs_out = cached_fetch_surf_fsaverage(self.mesh)
        resampled = {}
        for hemi, values in (("left", left), ("right", right)):
            infl_in_xyz, _ = nib.load(getattr(fs_in, f"infl_{hemi}")).darrays
            infl_out_xyz, _ = nib.load(getattr(fs_out, f"infl_{hemi}")).darrays
            tree = cKDTree(infl_in_xyz.data)
            distances, indices = tree.query(infl_out_xyz.data, k=5)
            if "int" in data.dtype.name:
                # get most frequent
                resampled[hemi] = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), axis=1, arr=values[indices]
                )
            else:
                distances = np.where(distances == 0, 1e-12, distances)
                weights = 1 / distances
                weights = weights / weights.sum(axis=1, keepdims=True)
                resampled[hemi] = np.sum(values[indices] * weights, axis=1)
        left, right = resampled["left"], resampled["right"]

    return dict(left=left, right=right, both=np.r_[left, right])
```

**What it does**: Auto-detects the input resolution by matching `data.shape[0] // 2` against `FSAVERAGE_SIZES` (a dict from `neuralset` mapping `"fsaverage3"→642`, `"fsaverage4"→2562`, `"fsaverage5"→10242`, `"fsaverage6"→40962`, `"fsaverage7"→163842` per hemisphere). If the input resolution differs from `self.mesh`, it builds a `cKDTree` on the inflated-surface coordinates of the *source* mesh and queries the inflated coordinates of the *target* mesh. K=5 nearest neighbours; inverse-distance weighted for float data; majority-vote for integer (label) data.

**Applicability**: This is exactly the helper needed if a client sends `(T, 20484)` fsaverage5 predictions but you want to display at fsaverage7 (163842×2 vertices). The only dependency is `scipy.spatial.cKDTree` and `nibabel`/`nilearn` for coordinate loading. The algorithm itself (~15 lines) is short enough to reimplement from scratch without needing to depend on tribev2 (see Section E / reusability verdict).

### A3. ROI / Parcellation Lookup — HCP MMP (Glasser-360)

File: `tribev2/tribev2/utils.py`, lines 213–284

The full HCP MMP-1 (Glasser-360) atlas is exposed through three cached functions:

```python
@lru_cache
def get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both"):
    # Uses mne.datasets.fetch_hcp_mmp_parcellation() to download the annot
    # Returns dict: roi_name -> np.array of vertex indices (at target mesh resolution)
    # Projects fsaverage7 (163842 per hemi) indices down to the target mesh size
    ...

def get_hcp_roi_indices(rois: str | list[str], hemi="both", mesh="fsaverage5"):
    # Supports wildcard: "V*" matches V1, V2, V3A, etc.
    # Returns np.array of vertex indices
    ...

def summarize_by_roi(data: np.ndarray, hemi="both", mesh="fsaverage5"):
    # Returns per-ROI mean — shape (n_rois,)
    ...
```

The `get_hcp_labels` function at line 240 asserts that at fsaverage7 the total is 163842 vertices per hemisphere, then truncates to `expected_size = FSAVERAGE_SIZES[mesh]` vertices at the target resolution (lines 242–245). This is the **parcel→vertex mask** machinery.

**No Yeo or Schaefer atlas** parcellation-to-vertex code is in `utils.py`. Schaefer and DiFuMo appear only as **volume atlases** in `BasePlotBrain.get_atlas()` / `atlas_masker` (base.py lines 67–84) for the volumetric-input pathway; they are not projected to vertices directly, only used with `NiftiLabelsMasker` / `NiftiMapsMasker`.

### A4. Sulcal-depth / Curvature Loading

In `get_mesh()` (base.py lines 150–153):

```python
bg_key = "curv" if self.bg_map == "curvature" else "sulc"
bg_map = nib.load(getattr(fs_out, f"{bg_key}_{hemi}")).darrays[0].data
if self.bg_map == "thresholded":
    bg_map = 1.0 * (bg_map > -0.10)
    bg_map[-1] = -5
    bg_map[-2] = 2.0
```

Three modes:
- `"sulcal"` (default) — raw sulcal depth (`sulc_left`, `sulc_right`)
- `"curvature"` — mean curvature (`curv_left`, `curv_right`)
- `"thresholded"` — binarised sulc > -0.10 (gyrus=0, sulcus=1), with sentinel extreme values pinned at positions [-1] and [-2] (a normalisation hack for nilearn's colormap)

In `cortical_pv.py` the sulcal `bg_map` feeds the background darkening directly (lines 122–128):

```python
bg_map = mesh["bg_map"]
bg_norm = (bg_map - bg_map.min()) / (bg_map.max() - bg_map.min() + 1e-8)
bg_rgb = 1 - np.column_stack(
    [self.bg_darkness + bg_norm * (1 - self.bg_darkness)] * 3
)
colors = rgba[:, 3:4] * rgba[:, :3] + (1 - rgba[:, 3:4]) * bg_rgb
```

This is the **exact same darkening logic we use**: sulcal depth → greyscale background → alpha-blend with stat-map RGBA. The upstream constant `bg_darkness=0` (field on `PlotBrainPyvista`, line 44) means fully dark sulci; adjusting this value lightens the sulcal floor.

---

## B. Subcortical Handling

### B5. `plotting/subcortical.py` — Full Analysis

File: `tribev2/tribev2/plotting/subcortical.py`

**Backend**: PyVista exclusively (no nilearn surface backend).

**What it renders**: Individual subcortical nuclei (thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, brainstem, cerebellum) reconstructed via **marching cubes** from the Harvard-Oxford subcortical probabilistic atlas, smoothed, and plotted as closed meshes.

Key functions:

```python
@lru_cache()
def get_mesh(label: str, resolution: tp.Literal["1mm", "2mm"]):
    """Marching cubes on Harvard-Oxford atlas mask → PyVista PolyData."""
    nii_mask = get_mask(label, resolution)
    volume = gaussian_filter(nii_mask.get_fdata().astype(float), sigma=1)
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.9)
    affine = nii_mask.affine
    verts = nib.affines.apply_affine(affine, verts)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.smooth(n_iter=50, relaxation_factor=0.01)
    return mesh

def nii_to_mesh(nii, mesh, mask_img=None):
    """Project voxel scores onto a subcortical surface mesh via vol_to_surf."""
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    vertex_vals = vol_to_surf(
        nii,
        surf_mesh=(vertices, faces),
        mask_img=mask_img,
        kind="line",
        depth=np.linspace(-3, 0, 40),
        interpolation="linear",
    )
    return vertex_vals
```

**What it does NOT do**: It does not render the 8802-voxel subcortical output as a point cloud or volume. Instead it creates a **new anatomical mesh** from the atlas and then projects the voxel scores onto that mesh via `vol_to_surf`. The 8802-voxel array from the TribeModel output is **not directly consumed** here — the subcortical module is anatomy-driven, not prediction-array-driven in the same way as the cortical module.

The `plot_subcortical()` function (lines 179–290) accepts either:
- `colors: dict` — per-label RGBA, for a simple colour-coded view
- `voxel_scores: np.ndarray` shape `(N_subcortical_voxels,)` or `(3, N_subcortical_voxels)` — the scores are projected per-nucleus via `voxel_to_mesh()` → `nii_to_mesh()`

Camera is controlled directly on the PyVista plotter (no `VIEW_DICT`):

```python
plotter.camera.zoom(zoom)             # default 1.3
plotter.camera.azimuth = azimuth      # default 15 degrees
plotter.camera.elevation = elevation  # default -10 degrees
light = pv.Light(intensity=intensity) # default 1.5
light.set_headlight()
```

**Implication for our exporter**: The subcortical mesh pipeline (marching cubes + MNI affine warp + Laplacian smooth) is directly portable. The per-nucleus meshes in MNI space are standard `(verts, faces)` pairs convertible to glTF/SCN. The `nii_to_mesh` projection is the missing piece if we want to colour subcortical structures from model predictions; it requires nilearn's `vol_to_surf`.

---

## C. Render-Config Knowledge Worth Lifting

### C6. Camera Presets / View Dictionaries

**Nilearn backend** (`cortical.py`, lines 22–35) — `(elevation_deg, azimuth_deg)` tuples passed to matplotlib's `view_init`:

```python
VIEW_DICT = {
    "left": (0, 180),
    "right": (0, 0),
    "medial_left": (0, 0),
    "medial_right": (0, 180),
    "dorsal": (90, 0),
    "ventral": (-90, 0),
    "anterior": (0, 90),
    "posterior": (0, -90),
    "posterior_left": (0, -135),
    "posterior_right": (0, -45),
    "posterior_ventral": (-45, -90),
    "posterior_ventral_left": (-10, -135),
}
```

**PyVista backend** (`cortical_pv.py`, lines 27–38) — `(view_vector, up_vector)` tuples for `pl.view_vector(vec, viewup=up)`:

```python
VIEW_DICT = {
    "ventral": ([0, 0, -1], [1, 0, 0]),
    "dorsal": ([0, 0, 1], [0, 1, 0]),
    "left": ([-1, 0, 0], [0, 0, 1]),
    "right": ([1, 0, 0], [0, 0, 1]),
    "anterior": ([0, 1, 0], [0, 0, -1]),
    "posterior": ([0, -1, 0], [0, 0, 1]),
    "medial_left": ([1, 0, 0], [0, 0, 1]),
    "medial_right": ([-1, 0, 0], [0, 0, 1]),
    "posterior_left": ([-1, 0, 0], [0, 0, 1]),
    "posterior_right": ([-1, 0, 0], [0, 0, 1]),
}
```

The PyVista view vectors map directly to SceneKit camera conventions: `view_vector` is the direction the camera looks along (negate for the `SCNVector3` camera position offset from origin), and `up` is the `SCNNode` up direction.

**Lighting** (PyVista backend, `cortical_pv.py`):

```python
ambient: float = 0.3   # passed as ambient= to pl.add_mesh(... smooth_shading=True)
dpi: int = 3000        # offscreen render resolution
bg_darkness: float = 0 # 0 = fully dark sulci, 1 = white sulci
```

Subcortical uses a dedicated headlight at intensity 1.5 (`subcortical.py` line 281).

### C7. Colormap Defaults

The default colormap in all `plot_surf` calls is `cmap="hot"` (matplotlib), but the demo notebook and `plot_colorbar` default use **`colorcet.cm.fire`**:

```python
# plotting/utils.py line 278
def plot_colorbar(ax, sm=None, cmap=colorcet.cm.fire, vmin=0, vmax=1, ...):
```

Demo notebook cells (tribe_demo.ipynb):
```python
plotter.plot_timesteps(preds[:n_timesteps], ..., cmap="fire", norm_percentile=99,
                       vmin=.6, alpha_cmap=(0, .2), show_stimuli=True)
```

The `get_cmap` function in `plotting/utils.py` (lines 139–161) resolves cmap names from three namespaces in order: `matplotlib.cm`, `seaborn.cm`, then `colorcet.cm`. So `"fire"` resolves to `colorcet.cm.fire`.

**clim defaults**: No hardcoded global clim. `vmin`/`vmax` default to `None` and fall back to `np.nanmin`/`np.nanmax` inside `get_scalar_mappable()`. The demo uses `vmin=0.6` with `norm_percentile=99` — meaning data is first percentile-clipped to [1st, 99th] percentile, then the colormap starts at 60% of that range (effectively suppressing the bottom 60% of values as transparent/zero).

**Bipolar vs unipolar**: The `symmetric_cbar` boolean (default `False`) in `plot_surf` / `PlotBrainPyvista.plot_surf` flips sign: `vmin, vmax = -vmax, vmax`. The `threshold` parameter greys out values in `[-threshold, +threshold]` via `get_thresholded_sm()` (utils.py lines 55–74):

```python
def get_thresholded_sm(vmin, vmax, threshold=None, cmap=None):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap("hot")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if threshold is not None:
        istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
    our_cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=our_cmap, norm=norm)
    sm._A = []
    return sm
```

The `alpha_cmap` parameter adds a transparency ramp so low values fade to transparent (useful for overlaying on the grey background):

```python
def get_alpha_cmap(cmap, threshold: float = 0, scale: float = 1, symmetric=False):
    # Builds a 1024-point ListedColormap with alpha=0 below threshold,
    # linearly ramping to alpha=1 at threshold+scale.
    ...
```

### C8. Mosaic / Time-Series Layout and MP4 Export

**`plot_timesteps`** (base.py lines 235–365):

Layout uses `plt.subplot_mosaic`. The mosaic string is dynamically built as a 2D list:
- Rows: one row per neuro signal dict key (e.g. `["Brain response_0", "Brain response_1", ...]`)
- Optionally prepend: video row (`VIDEO_KEY_i`), audio row (`SOUND_KEY * n`), text row (`TEXT_KEY * n`)
- `height_ratios`: video=0.7, audio=0.3, text=0.3, brain=1.0 each
- Figure size: `(2.5 * n_timesteps, 2 * sum(height_ratios))`
- `wspace=0.0, hspace=0`

Column titles: `t={timestamps[i]}s` placed via `fig.text(0.5, -0.1, ...)` relative to each brain axis.

**`plot_timesteps_mp4`** (base.py lines 431–490):

Renders one PNG per frame at `dpi=300` to a temp dir, then invokes ffmpeg:

```python
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(1),          # 1 fps input (1 TR = 1 second)
    "-i", f"{str(tmp_dir)}/tmp_%05d.png",
]
if interpolated_fps is not None:
    cmd.append("-vf")
    cmd.append(f"minterpolate=fps={interpolated_fps}")  # motion interpolation
cmd.extend([
    "-c:v", "libx264",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    str(filepath),
])
subprocess.run(cmd)
```

Single frame size: `(3, 3)` inches at 300 dpi → 900×900 px per frame.

**For Swift UI**: The mosaic structure (video strip + brain surface strip + text strip) would map well to a SwiftUI `VStack` + `ScrollView` with a `TimelineView` controller. The 1-fps frame rate aligns with TR=1s. The `minterpolate` ffmpeg filter is the equivalent of `CADisplayLink`-driven keyframe interpolation in SceneKit.

---

## D. Reusability Surface

### D9. Functions Callable on Pure numpy Arrays

The following functions take only numpy arrays (no TribeModel, no segments, no PyTorch) and return figures or arrays:

| Function | Signature | Returns | File |
|---|---|---|---|
| `PlotBrainPyvista.plot_surf` | `(data: ndarray (2*N,), axes, views, ...)` | `ScalarMappable` | cortical_pv.py:80 |
| `PlotBrainNilearn.plot_surf` | `(signals: ndarray (2*N,), ...)` | `ScalarMappable` | cortical.py:59 |
| `PlotBrainPyvista.plot_surf_rgb` | `(signals: List[ndarray], ...)` | `ndarray (N,4)` vertex colours | cortical_pv.py:169 |
| `PlotBrainNilearn.plot_surf_rgb` | `(signals: List[ndarray], ...)` | `ndarray (N,4)` | cortical.py:161 |
| `BasePlotBrain.plot_timesteps` | `(neuro: ndarray (T,2*N), ...)` | `matplotlib.Figure` | base.py:235 |
| `BasePlotBrain.plot_timesteps_mp4` | `(neuro: ndarray (T,2*N), filepath, ...)` | None (writes mp4) | base.py:431 |
| `BasePlotBrain.get_mesh` | `()` | `dict{left,right,both: {coords,faces,bg_map}}` | base.py:127 |
| `BasePlotBrain.get_stat_map` | `(data: ndarray (2*N,))` | `dict{left,right,both: array}` | base.py:178 |
| `robust_normalize` | `(array, axis, percentile, ...)` | `ndarray` | utils.py:19 |
| `get_cmap` | `(cmap_name, alpha_cmap)` | `matplotlib.colors.Colormap` | utils.py:139 |
| `get_scalar_mappable` | `(data, cmap, vmin, vmax, ...)` | `ScalarMappable` | utils.py:38 |
| `plot_subcortical` | `(ax, voxel_scores=ndarray, ...)` | `ScalarMappable` | subcortical.py:179 |
| `get_hcp_roi_indices` | `(rois, hemi, mesh)` | `ndarray` of vertex indices | utils.py:268 |
| `summarize_by_roi` | `(data: ndarray (2*N,), hemi, mesh)` | `ndarray (n_rois,)` | utils.py:287 |
| `get_topk_rois` | `(data: ndarray, hemi, mesh, k)` | `list[str]` | utils.py:309 |

**Entanglement note**: `plot_timesteps` accepts an optional `segments` list (TribeModel output), but this is only used for the `show_stimuli=True` branch. When called with `show_stimuli=False` (the default), it has zero TribeModel dependency. All functions above can be called with only numpy arrays.

### D10. `utils_fmri.py` — Pre-processing Before Export

`TribeSurfaceProjector` (utils_fmri.py lines 129–248) is the correct upstream step for users who provide **MNI volumetric data** rather than fsaverage surface arrays. It wraps `nilearn.surface.vol_to_surf` and handles:

- 4D NIfTI → fsaverage5/6/7 surface projection (lines 203–226)
- Optional intermediate-depth mesh (pial/white blend, lines 175–198) via `center_depth` parameter
- 2D surface data passthrough with downsampling (lines 228–244)

```python
# Usage: project a (91, 109, 91, T) MNI volume to (20484, T) fsaverage5
projector = TribeSurfaceProjector(mesh="fsaverage5", radius=3.0, kind="ball")
surface_data = projector.apply(mni_volume_4d)  # shape: (20484, T)
```

`FmriTemplateSpace` enum (lines 22–67) provides the canonical MNI space identifiers with expected shapes — useful for input validation in our FastAPI endpoint.

`load_mni_mesh()` (lines 76–126) loads subject-specific pial/white .gii files from a FreeSurfer subjects directory, indexed by `FmriTemplateSpace` value. This is relevant only if users have individual FreeSurfer reconstructions; it is not needed for the standard fsaverage average-subject pipeline.

**Verdict**: `TribeSurfaceProjector.apply()` should be run **before** `BrainMeshExporter` when the user uploads an MNI volume. It is the correct entry point. The projector depends on `nibabel` + `nilearn` + `pydantic`, all already in our stack.

### D11. License: Reimplement vs Depend At Runtime

**Short / mechanical enough to reimplement from scratch** (no license concern):

1. **PyVista `VIEW_DICT`** (10 `(vec, up)` tuples) — pure constants, trivially translatable to a Swift enum or a JSON config.
2. **kNN resampling kernel** in `get_stat_map()` — 15 lines of scipy + numpy. The algorithm is standard inverse-distance-weighted kNN; no novel IP. We already have the fsaverage coordinate files in-repo.
3. **`robust_normalize`** — 10 lines, a standard percentile-clip normalisation.
4. **Sulcal background darkening** blend formula — 3 lines of numpy; trivially portable to a Metal shader or a GLSL fragment shader.
5. **`get_alpha_cmap` transparency ramp** — 15 lines; the same logic can be expressed as a Swift `CAGradientLayer` or a Metal texture lookup.
6. **ffmpeg invocation pattern** (`plot_timesteps_mp4`) — the command string is freely reimplementable.

**Should depend on at runtime** (non-trivial, not worth reimplementing):

1. **`nilearn.datasets.fetch_surf_fsaverage`** — network-fetching + caching of fsaverage `.gii` files across 5 resolutions. Reimplementing this means bundling or fetching the files ourselves; easier to depend on nilearn for the surface-loading plumbing in the Python server.
2. **`TribeSurfaceProjector` / `nilearn.surface.vol_to_surf`** — the `vol_to_surf` ball/line sampling from MNI volumes is non-trivial (involves nearest-voxel or trilinear interpolation within a radius ball around each vertex). Only relevant if we support MNI input in the FastAPI endpoint.
3. **`get_hcp_labels` / HCP MMP parcellation** — depends on `mne.datasets.fetch_hcp_mmp_parcellation` and the `.annot` file format parsing. The ROI→vertex index mapping is non-trivial to reproduce; depend on MNE+nibabel for this, or use the Glasser atlas already bundled in-repo.

---

## E. Anything We Missed

### E12. 3D/Render Libraries Beyond PyVista + Nilearn

Searching the full codebase:

- **`pyvista`**: present — primary 3D backend for `PlotBrainPyvista` and `plot_subcortical`. Used in `cortical_pv.py`, `subcortical.py`.
- **`vtk`**: **not imported directly**. PyVista wraps VTK internally but tribev2 never calls VTK APIs directly.
- **`imageio`**: **not present**.
- **`ffmpeg`**: present — invoked via `subprocess.run(["ffmpeg", ...])` in `plot_timesteps_mp4` (base.py line 468). Not a Python import; used as a CLI tool.
- **`mp4`**: produced by `plot_timesteps_mp4`.
- **`gif`**: produced by `PlotBrainNilearn.save_gif()` (cortical.py lines 273–288) using `matplotlib.animation.PillowWriter` — no external dep beyond Pillow.
- **`mlab`**: **not present** (no mayavi).
- **`surfplot`**: **not present**.
- **`brainspace`**: **not present**.
- **`fury`**: **not present**.
- **`vedo`**: **not present**.
- **`polyscope`**: **not present**.
- **`open3d`**: **not present**.
- **`scikit-image` (`skimage`)**: present — `skimage.measure.marching_cubes` is used in `subcortical.py` line 161 to extract subcortical meshes from volumetric atlas masks.
- **`scipy.ndimage.gaussian_filter`**: used in `subcortical.py` line 159 to smooth the volumetric mask before marching cubes.
- **`moviepy`**: present in core deps — used in `plotting/utils.py` (`get_clip`, `get_audio`) for video/audio segment extraction in the `show_stimuli` branch of `plot_timesteps`. Not a 3D rendering library.
- **`colorcet`**: present — source of the `fire` colormap.

**Summary**: The only 3D library is PyVista (VTK-backed). No point-cloud libs, no WebGL exporters, no OpenGL bindings. All output is rasterised to PNG via PyVista's offscreen renderer, then embedded in matplotlib axes.

### E13. `pyproject.toml` `[plotting]` Extra

File: `tribev2/pyproject.toml`, lines 40–49:

```toml
[project.optional-dependencies]
  plotting = [
    "nibabel",
    "matplotlib",
    "seaborn",
    "colorcet",
    "nilearn",
    "scipy",
    "pyvista",
    "scikit-image",
  ]
```

Eight packages. Notable absences from `plotting`: `mne` (required by `get_hcp_labels` in `utils.py` — pulled in transitively or is a hidden dep), `moviepy` (in core deps), `tqdm` (in core deps). The `scikit-image` dep exists solely for `skimage.measure.marching_cubes` in the subcortical pipeline.

### E14. `tribe_demo.ipynb` — All Visualization-Producing Cells

The notebook has exactly **two visualization-producing cells**. Both call the same function with the same parameters, differing only in input stimulus type:

**Cell 1** (cell_id `"5"`) — Video stimulus predictions:
```python
n_timesteps = 15
fig = plotter.plot_timesteps(
    preds[:n_timesteps],
    segments=segments[:n_timesteps],
    cmap="fire",
    norm_percentile=99,
    vmin=.6,
    alpha_cmap=(0, .2),
    show_stimuli=True
)
```
Shows 15 brain surface panels (1 per second of Sintel trailer), with a video-frame strip, an audio waveform strip, and a word-event strip above.

**Cell 2** (cell_id `"d2dcaa65"`) — Text-to-speech predictions:
```python
n_timesteps = 15
fig = plotter.plot_timesteps(
    preds[:n_timesteps],
    segments=segments[:n_timesteps],
    cmap="fire",
    norm_percentile=99,
    vmin=.6,
    alpha_cmap=(0, .2),
    show_stimuli=True
)
```
Identical call, but input was Shakespeare text converted via gTTS→WhisperX. Shows brain + audio waveform + word strip (no video row, since there are no video events in this run).

Both calls use `PlotBrain` which is the alias for `PlotBrainPyvista` (the PyVista backend). Both use:
- `cmap="fire"` (colorcet)
- `norm_percentile=99` (clip to [1st, 99th] percentile)
- `vmin=0.6` (show only the top 40% of the normalized range)
- `alpha_cmap=(0, 0.2)` — alpha=0 at 0% of colormap range, ramps to alpha=1 by 20%

No `plot_timesteps_mp4`, no `plot_surf_rgb`, no subcortical plots appear in the demo notebook.

---

## Reusability Verdict

### Top 3 to Incorporate

**1. `BasePlotBrain.get_mesh()` (base.py:127) — Direct lift for mesh loading pipeline.**  
Returns `(coords: N×3 float, faces: N_faces×3 int, bg_map: N float)` for fsaverage3–7, with pial/inflated/half-inflated modes and hemisphere concatenation with correct index offsetting. This is the **exact** geometry format our `BrainMeshExporter` needs. Adapt it by calling `nilearn.datasets.fetch_surf_fsaverage` once at server startup and caching the dict; the six lines of vertex/face extraction and the `bg_map` load are directly portable. Reimplementing from scratch using nibabel is also viable since the logic is simple once you know which `.gii` files to load.

**2. `get_stat_map()` kNN resampling algorithm (base.py:178) — Reimplement from scratch.**  
The 15-line inverse-distance-weighted kNN resampling between fsaverage resolutions should be reimplemented (it is short enough, and CC BY-NC blocks commercial use). The algorithm uses `scipy.spatial.cKDTree`, which is already in our stack. This directly solves the fsaverage5→fsaverage7 upsampling problem for high-resolution display, and the integer-label branch (majority vote) is also needed for projecting Glasser parcel IDs to higher-resolution meshes.

**3. PyVista `VIEW_DICT` camera presets (cortical_pv.py:27) — Copy the constants verbatim as config.**  
Ten `(view_vector, up_vector)` tuples covering all standard neuroimaging views (lateral L/R, medial L/R, dorsal, ventral, anterior, posterior). These are pure data with no algorithmic IP and convert directly to SceneKit `SCNVector3` camera positions. Freely reimplement as a Swift enum or a JSON config file.

### Top 3 to NOT Depend On

**1. `TribeModel` / `TribeExperiment` from `demo_utils.py` / `main.py`.**  
Has a deep dependency tree (PyTorch Lightning, neuralset, exca, wandb, huggingface_hub, spacy, julius, etc.) and is the fMRI prediction model itself — not a rendering tool. Our exporter is downstream of this; we receive the `(T, 20484)` array and do not call the model. Never import these into `tribe_backend/mesh/`.

**2. `PlotBrainPyvista` / `PlotBrainNilearn` as runtime rendering dependencies.**  
Both require a live display or offscreen VTK context (`pv.Plotter(off_screen=True)`). In a server context, PyVista's offscreen rendering needs a virtual framebuffer (Xvfb) on Linux, which is fragile in a container. Our exporter path — building mesh buffers + colour arrays and returning JSON/binary — does not need a plotter at all. Use the geometry and colour math (pure numpy) in isolation.

**3. `get_hcp_labels` / `summarize_by_roi` from `utils.py` (the MNE-backed version).**  
These depend on `mne.datasets.fetch_hcp_mmp_parcellation` which downloads ~50 MB of FreeSurfer subject data on first call and relies on MNE's dataset infrastructure. We already have the Glasser-360 atlas bundled as a static asset. Build our own `roi_indices(parcel_id, mesh)` lookup from the bundled `.annot` or `.label.gii` files rather than pulling in MNE at runtime.

---

## Sources

- `/home/md2292/tribev2/tribev2/plotting/base.py`
- `/home/md2292/tribev2/tribev2/plotting/cortical.py`
- `/home/md2292/tribev2/tribev2/plotting/cortical_pv.py`
- `/home/md2292/tribev2/tribev2/plotting/subcortical.py`
- `/home/md2292/tribev2/tribev2/plotting/utils.py`
- `/home/md2292/tribev2/tribev2/utils_fmri.py`
- `/home/md2292/tribev2/tribev2/utils.py`
- `/home/md2292/tribev2/tribev2/demo_utils.py`
- `/home/md2292/tribev2/tribe_demo.ipynb`
- `/home/md2292/tribev2/pyproject.toml`
- `/home/md2292/tribev2/README.md`
