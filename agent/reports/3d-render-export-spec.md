# 3D Render Export — Backend Output Spec & Consumer Implementation Guide

**Module under spec:** `tribe_backend.mesh` (`BrainMeshExporter`)
**Format header:** `tribe_brain_mesh_v1`
**Surface:** fsaverage5 (FreeSurfer standard cortical mesh, loaded via nilearn)
**Date:** 2026-04-25

---

## 1. Scope

This document specifies, byte-for-byte, what the `tribe_backend.mesh` compartment writes to disk and how a 3D consumer (Swift/SceneKit, Web/Three.js, native/Metal, etc.) must interpret those artifacts to faithfully render TRIBE v2 cortical predictions.

Producer entry point: `BrainMeshExporter` in `tribe_backend/mesh/exporter.py`.
Inputs to the producer: `tribe_backend.contracts.TribeOutput` (a frozen dataclass containing `cortical: (T, 20484) float32` and metadata).

Out of scope: subcortical voxel rendering, parcellation/ROI text reports (separate compartment), live-feed transport.

---

## 2. Invariants the Consumer May Rely On

These hold across all four export formats and all calls to `BrainMeshExporter`:

1. **Vertex count**: exactly **`V = 20,484`** vertices in the combined mesh.
2. **Face count**: exactly **`F = 40,960`** triangles.
3. **Hemisphere split**: `vertices[0:10242]` is the **left hemisphere**, `vertices[10242:20484]` is the **right hemisphere**. Right-hemisphere face indices have already been **offset by `+10242`** so the combined mesh is a single index space (consumers must NOT add an offset themselves).
4. **Coordinate frame**: RAS millimetres (FreeSurfer / fsaverage5 native). X≈Right+, Y≈Anterior+, Z≈Superior+. Range roughly ±80 mm. Origin near the AC. There is **no implicit unit scaling** — consumers should scale (e.g. ×0.01 for SceneKit metres) and centre as needed.
5. **Winding order**: triangle vertex order matches the FreeSurfer/GIFTI source (counter-clockwise when viewed from outside on `pial`). Cull back faces accordingly.
6. **Format version**: every metadata payload starts with `"format": "tribe_brain_mesh_v1"`. Consumers MUST verify and refuse unknown versions.
7. **Geometry is static**: for a fixed `surface_type`, vertex positions and face indices NEVER change between calls. Only colors change per inference. Consumers may ship geometry once with the app and stream colors only.
8. **Color encoding**: per-vertex RGBA, **uint8, 4 bytes per vertex, premultiply-free, sRGB-ish via matplotlib colormap**. Order is `[R, G, B, A, R, G, B, A, ...]`. Alpha is currently always `255` from `RdBu_r` but consumers SHOULD honour it (future cmaps may use alpha).
9. **Endianness**: little-endian (host order on the producer; cleo runs on x86_64/ARM64 little-endian). Binary files are written via `numpy.ndarray.tofile`, so they are raw, contiguous, no header, no padding.

If any of these are violated by an export the producer is buggy — please file an issue rather than working around it on the consumer side.

---

## 3. Source Data Pipeline (Producer Side)

For context, here is what generates the artifacts. Consumers do not need to reproduce this, but understanding it helps interpret the colors.

```
TribeOutput.cortical  (T, 20484) float32, z-scored BOLD
        │
        ▼  aggregate_temporal(method ∈ {"mean", "peak", "max"})
(20484,) float32  per-vertex scalar
        │
        ▼  activation_to_rgba(vmin=-3, vmax=3, cmap="RdBu_r")
(20484, 4) uint8  per-vertex RGBA
        │
        ▼  load_fsaverage5(surface_type ∈ {"pial","white","infl","sphere","flat"})
(20484, 3) float32 coords + (40960, 3) int32 faces
        │
        ▼  serialize as JSON | binary | GLB | animation bundle
artifacts on disk
```

**Default colormap semantics:** `"RdBu_r"` clipped to `[-3.0, +3.0]` → blue = strong negative deviation, white = baseline (z≈0), red = strong positive. Configurable via `BrainMeshExporter(cmap=..., vmin=..., vmax=...)`.

**Aggregation methods** (single-frame exports):
- `"mean"` — `cortical.mean(axis=0)` (default).
- `"peak"` — per vertex, the value at the timestep with largest `|value|`.
- `"max"` — `cortical.max(axis=0)`.

For animation, no aggregation occurs — each timestep produces its own color frame.

---

## 4. Output Format Reference

`BrainMeshExporter` exposes four export methods. Each is described below: the files written, byte layout, when to use, and a load recipe.

### 4.1 `export_json(output, out_dir) → Path`

Writes a single self-contained `brain_mesh.json`. Use for: prototyping, debugging, web consumers that already speak JSON, low-throughput tooling. Avoid for animation or large-volume streaming (slow to parse, ~6× the size of the binary form).

**File: `<out_dir>/brain_mesh.json`** — UTF-8 JSON object:

```json
{
  "format": "tribe_brain_mesh_v1",
  "surface": "fsaverage5",
  "surface_type": "pial",
  "vertex_count": 20484,
  "face_count": 40960,
  "vertices": [[x, y, z], ... 20484 entries ...],
  "faces":    [[i0, i1, i2], ... 40960 entries ...],
  "colors":   [[r, g, b, a], ... 20484 entries ...],
  "window_id": "<string-or-null>"
}
```

- `vertices`: float, mm (RAS).
- `faces`: int32-range integer indices into `vertices`.
- `colors`: integers in `[0, 255]`. Length equals `vertex_count`.
- `window_id`: opaque string identifying the inference window (echoed from `TribeOutput.window_id`); may be `null`.

> **Note (current implementation gap):** `export_json` does **not** emit per-vertex normals. Consumers that need lighting must either (a) compute normals themselves once after load, or (b) use `export_glb` which embeds them. See §5 for a normal-computation reference.

### 4.2 `export_binary(output, out_dir) → Path`

Writes a directory of raw little-endian binary buffers plus a small metadata sidecar. **This is the recommended format** for production consumers because the buffers are directly memory-mappable into GPU vertex/index/color sources without parsing.

Files written (all in `<out_dir>`):

| File | Dtype | Element layout | Total bytes |
|---|---|---|---|
| `brain_vertices.bin` | float32 | `[x0, y0, z0, x1, y1, z1, ...]`, 3 floats per vertex | `20484 × 12 = 245,808` |
| `brain_faces.bin` | int32 | `[i0, i1, i2, i3, ...]`, 3 ints per triangle | `40960 × 12 = 491,520` |
| `brain_colors.bin` | uint8 | `[r0, g0, b0, a0, r1, g1, b1, a1, ...]` | `20484 × 4 = 81,936` |
| `brain_meta.json` | JSON | metadata (below) | <1 KB |

`brain_meta.json` shape:

```json
{
  "format": "tribe_brain_mesh_v1",
  "surface": "fsaverage5",
  "surface_type": "pial",
  "vertex_count": 20484,
  "face_count": 40960,
  "bytes_per_vertex": 12,
  "bytes_per_face": 12,
  "bytes_per_color": 4,
  "files": {
    "vertices": "brain_vertices.bin",
    "faces": "brain_faces.bin",
    "colors": "brain_colors.bin"
  },
  "window_id": "<string-or-null>"
}
```

Static-vs-dynamic split: for a given `surface_type`, `brain_vertices.bin` and `brain_faces.bin` are deterministic and may be cached/bundled with the consumer app. Only `brain_colors.bin` (and `brain_meta.json`) need to be re-fetched per inference.

> **Implementation gap:** like JSON, `export_binary` does NOT emit `brain_normals.bin`. The reference plan in `agent/brain-3d-parser.md` includes one; the current code does not. Compute normals on the consumer side or extend the producer (see §7).

### 4.3 `export_glb(output, out_dir) → Path`

Writes `<out_dir>/brain_mesh.glb` — a binary glTF 2.0 file via `trimesh`. Includes:
- Vertex positions
- Triangle indices
- Per-vertex RGBA colors (as `COLOR_0`)
- Per-vertex normals (computed via `_compute_vertex_normals`, area-weighted face-normal accumulation)

Use for: tools that already speak glTF (Blender, Three.js, model-viewer, RealityKit via ModelIO). Don't use for animation — GLB does not naturally express per-frame vertex-color streams in this pipeline.

Single file, ~1.5–2 MB. No sidecar metadata.

### 4.4 `export_animation_bundle(cortical, out_dir, prefix="brain_colors_t") → Path`

Writes static geometry once + one color frame per timestep. **Note** the input here is the raw `(T, V) ndarray`, NOT a `TribeOutput`.

Files written:

| File | Notes |
|---|---|
| `brain_vertices.bin` | identical layout to §4.2 |
| `brain_faces.bin` | identical layout to §4.2 |
| `brain_colors_t0000.bin` … `brain_colors_t{T-1:04d}.bin` | one per timestep, each `20484 × 4 = 81,936 bytes` |
| `animation_meta.json` | see below |

`animation_meta.json`:

```json
{
  "format": "tribe_brain_mesh_v1",
  "kind": "animation_bundle",
  "vertex_count": 20484,
  "frame_count": 31,
  "bytes_per_color": 4,
  "frames": ["brain_colors_t0000.bin", "brain_colors_t0001.bin", ...]
}
```

Frame indexing matches `cortical[t]`. Sampling rate is implicitly **1 Hz** (TRIBE v2 emits 1 timestep/sec). For a standard 30 s window, `frame_count == 31` (inclusive of `t=0` and `t=30`).

Bundle size for a 30 s window ≈ `1 MB geometry + 31 × 80 KB ≈ 3.5 MB`.

---

## 5. Consumer Implementation Guide

Goal: a smooth, real-time 3D render of the cortical surface with activations, plus optional 1 Hz animation playback.

### 5.1 Choose a format

| Use case | Recommended format |
|---|---|
| iOS/macOS native (SceneKit, RealityKit) | `export_binary` for static + animation; `export_glb` only for one-shot model preview |
| Web (Three.js, model-viewer) | `export_glb` for one-shots; `export_binary` (fetched as ArrayBuffers) for animation |
| Desktop tooling, Blender | `export_glb` |
| Debugging / data inspection | `export_json` |

The rest of §5 assumes the **binary** path (most efficient and the only one supporting animation).

### 5.2 Bootstrapping the renderer

1. **Verify metadata.** Parse `brain_meta.json`. Reject unless `format == "tribe_brain_mesh_v1"`. Read `vertex_count`, `face_count`, `bytes_per_*` and use them to size buffers — do not hard-code `20484` even though it is stable, since the version header is the contract.
2. **Load geometry** (once per app session, ideally from app bundle):
   - Map `brain_vertices.bin` as `Float32Array` / `MTLBuffer` of length `vertex_count * 3`.
   - Map `brain_faces.bin` as `Int32Array` / index buffer of length `face_count * 3`. Renderers that require unsigned indices (WebGL2 with `UNSIGNED_INT`, Metal) accept the bit-pattern as-is — fsaverage5 indices fit comfortably in `[0, 20484)`.
3. **Compute vertex normals** (once, after geometry load) — area-weighted accumulation:
   ```text
   for each face (i0, i1, i2):
       fn = cross(v1 - v0, v2 - v0)         # length = 2*area
       normals[i0] += fn; normals[i1] += fn; normals[i2] += fn
   normalize(normals)                        # divide by L2 norm; guard zero
   ```
   Cache the result; geometry never changes for a given `surface_type`.
4. **Place the camera.** Coordinates are mm, ~±80 mm extent. Centre the mesh (`-mean(vertices)` translation) and apply a uniform scale appropriate to the engine (e.g. `0.01` for SceneKit metres, identity for Three.js scenes already working in mm). Initial orbit camera at ~2× extent on +Y looking toward origin gives a good front view.
5. **Material.** Use unlit-with-vertex-colors OR PBR with vertex colors as albedo. Recommended starting point: physically based, metallic = 0, roughness ≈ 0.6, base color = white, vertex colors enabled. Disable specular highlights if rendering for clinical/scientific clarity (they distort activation reading).

### 5.3 Streaming activations (single frame)

On each new inference:
1. Fetch `brain_colors.bin` (and updated `brain_meta.json` if `window_id` matters to your UI).
2. Validate `len(brain_colors.bin) == vertex_count * 4`. If not, abort — the producer likely changed format.
3. Upload the buffer to the existing color attribute of the mesh. Do NOT rebuild geometry. In SceneKit this means swapping the `SCNGeometrySource` of semantic `.color`; in Three.js, `geometry.attributes.color.array.set(buf); geometry.attributes.color.needsUpdate = true`.
4. Optionally fade between previous and new color buffer over ~200 ms using a shader uniform `t ∈ [0,1]` and lerping in the fragment stage. This avoids a visual pop.

### 5.4 Streaming activations (animation)

On bundle download:
1. Parse `animation_meta.json`. Capture `frame_count` and `frames` (filenames in order).
2. Pre-fetch all frame buffers in parallel (each is 80 KB; total ~2.5 MB for a 30 s clip — easy to keep in memory).
3. Driver loop at **1 frame per second** (the native fMRI rate). For smooth playback, **interpolate** between consecutive color buffers in the fragment shader — sample two buffers `c[t]` and `c[t+1]`, lerp by the sub-second phase. Without interpolation the brain visibly steps at 1 Hz which looks broken on a 60 fps display.
   - On GPUs without two-color-attributes-per-mesh support (rare), upload `c[t+1]` as a vertex attribute on a parallel mesh and blend at the framebuffer level.
4. Expose a scrubber bound to integer `t ∈ [0, frame_count)` plus a play/pause toggle.

### 5.5 Picking & ROI overlay

Vertex indices are stable — `index 12345` always refers to the same anatomical location on fsaverage5. To map a hit-tested vertex back to a Glasser ROI label, the parcellation compartment (`tribe_backend.parcellation`) emits a separate report keyed by region; the consumer can join on vertex → ROI using a precomputed lookup table (out of scope here; coordinate with parcellation output).

### 5.6 Performance budget

For a single static frame (V=20484, F=40960):
- GPU: ~80 K triangles, vertex-colored, no textures. Comfortably 120+ fps even on integrated GPUs.
- Memory: ~1 MB resident for geometry + 80 KB per active color buffer.
- Network (per inference): 80 KB colors + <1 KB metadata. Negligible.
- Animation 30 s: ~2.5 MB for the full bundle.

Bottlenecks come from rebuilding geometry on every frame (don't) or re-parsing JSON (don't, use binary).

---

## 6. Pseudocode: Minimal Loader

Engine-agnostic outline. Translate to Swift/Metal/Three.js as needed.

```text
function loadBrain(dir):
    meta = parseJSON(dir/"brain_meta.json")
    require(meta.format == "tribe_brain_mesh_v1")

    V = meta.vertex_count           # 20484
    F = meta.face_count             # 40960

    vertices = readFloat32(dir/"brain_vertices.bin", count = V*3)
    faces    = readInt32  (dir/"brain_faces.bin",    count = F*3)
    colors   = readUInt8  (dir/"brain_colors.bin",   count = V*4)
    normals  = computeAreaWeightedNormals(vertices, faces)

    geom = makeGeometry(
        positions = vertices,         # 3 floats per vertex
        normals   = normals,          # 3 floats per vertex
        colors    = colors,           # 4 unorm bytes per vertex (RGBA)
        indices   = faces             # 3 uint indices per triangle
    )
    return geom

function updateColors(geom, dir):
    colors = readUInt8(dir/"brain_colors.bin", count = geom.V * 4)
    geom.color_attribute.upload(colors)   # do NOT rebuild positions/indices

function loadAnimation(dir):
    meta = parseJSON(dir/"animation_meta.json")
    require(meta.format == "tribe_brain_mesh_v1" and meta.kind == "animation_bundle")
    frames = [ readUInt8(dir/f, count = meta.vertex_count * 4) for f in meta.frames ]
    return frames  # list of color buffers; geometry loaded separately
```

---

## 7. Known Gaps & Roadmap (informational)

The reference plan (`agent/brain-3d-parser.md`) lists capabilities not yet in the producer. Consumers should code defensively for future additions:

1. **Vertex normals in JSON / binary outputs.** Currently only `export_glb` emits normals. Likely future fields: `brain_normals.bin` (float32, V×3) and `meta.files.normals`. Forward-compatible consumers should check `meta.files` before binding.
2. **Sulcal-depth shading.** `activation_to_rgba` accepts a `sulcal_depth` argument; `BrainMeshExporter` does not pass it. Activation colors are therefore on un-shaded vertices — a flat-looking but quantitatively faithful render. If the producer starts darkening sulci, the colors will already encode it; no consumer change required.
3. **Per-hemisphere export.** The reference plan envisages `separate=True` LH/RH bundles. Not implemented today; everything is the combined mesh. If added, expect a top-level `left_hemisphere` / `right_hemisphere` block; consumers can branch on its presence.
4. **Subcortical voxel rendering.** TRIBE v2 also produces `(T, 8802)` subcortical activations; no exporter exists for them yet.

Consumers should treat unknown fields permissively (ignore) and unknown `format` values strictly (refuse).

---

## 8. Quick Reference Card

```
Vertices:          20,484  (LH 0..10241, RH 10242..20483)
Faces:             40,960  triangles, RH indices pre-offset
Coords:            float32 mm, RAS
Indices:           int32 (0..20483)
Colors:            uint8 RGBA, V*4 bytes, little-endian
Color domain:      z-scored BOLD clipped to [-3, +3], cmap RdBu_r (default)
Static geometry:   brain_vertices.bin + brain_faces.bin (deterministic per surface_type)
Dynamic per call:  brain_colors.bin + brain_meta.json
Animation rate:    1 Hz (TRIBE v2 native), interpolate on consumer for smoothness
Format header:     "tribe_brain_mesh_v1"  (always verify before use)
```
