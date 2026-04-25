# cleo / tribe-backend — Primer

Generated: 2026-04-25 (focus: Glasser parcellation unit output)

## Tech Stack
- **Language**: Python ≥3.10 (pure backend; no JS/Node)
- **Numerics**: numpy, scipy
- **Neuroimaging**: nibabel, nilearn (Glasser-360 cortical, Harvard-Oxford subcortical, fsaverage5)
- **Test**: pytest, pytest-cases, pytest-cov, hypothesis, import-linter
- **Build**: setuptools / pyproject.toml

## Architecture Pattern
**Compartmented monolith** with a single shared frozen contract (`tribe_backend/contracts.py`). Compartments: `inference/`, `mesh/`, `parcellation/`, `control/`. Cross-imports between sibling compartments are forbidden by `import-linter` — they communicate only via `TribeOutput` / `StimulusWindow`.

## Parcellation Compartment — Output Contract

Single entry point: `tribe_backend.parcellation.GlasserParcellationUnit.generate_report(output: TribeOutput) -> Report`.

### Pipeline (4 layers, bottom-up)
1. **`parcellate_cortical((T, 20484)) → dict[name, (T,) f32]`** and **`parcellate_subcortical((T, 8802))`** — average activations across vertices/voxels belonging to each named parcel. Empty masks → zeros. Atlases loaded once via `atlas.load_glasser_cortical` (~360 parcels) and `atlas.load_harvard_oxford_subcortical`. Label `0` is dropped (background).
2. **`aggregate(series, method=...)` → `dict[name, float]`** — reduces each `(T,)` series to one scalar z-score. Methods:
   - `"window_mean"` (default) — `ts.mean()` over the full window. Surfaces sustained activations.
   - `"peak"` — value at `argmax(|ts|)` (sign preserved). Surfaces transients.
   - `"peak_window"` — sliding (2·radius+1)-frame mean with largest `|mean|`. Default radius=2 → 5 frames.
   Emits `WindowSizeWarning` when `T != EXPECTED_T_PER_30S_WINDOW` (=31, 1 Hz inclusive of t=0 and t=30).
3. **`rank_and_threshold(agg, top_k=10, z_threshold=1.5)` → `list[(name, signed_z)]`** — keep `|z| ≥ threshold`, sort descending by `|z|`, truncate to `top_k`. Both directions ranked together; sign retained.
4. **`generate_report(output, method, top_k, z_threshold)` → `Report`** — wraps the ranked list with narrative text + traceability fields.

### Output dataclasses (frozen)

`RegionActivation(name, z_score, direction, description)`
- `direction` ∈ `{"activation", "deactivation"}` (split on sign of `z_score`).
- `description` filled from `parcellation/descriptions.json` (lookup by parcel name; empty string if missing).

`Report(top_regions, text, method, z_threshold, window_id)`
- `top_regions: list[RegionActivation]` — ranked, already filtered.
- `text: str` — composed via `_compose_narrative`. Per-region phrase: `"<strength> <activation|deactivation> in <name> (<description>) (z=<+|-X.XX>)"`, joined with `; `, prefixed `"Window summary: "`, terminal `.`. Empty list → `"No significant activations detected in this window."`
- `method`, `z_threshold` — echoed from the call.
- `window_id` — copied from `output.window_id` (traces back to the `StimulusWindow`).
- `to_json()` / `from_json()` — round-trip serialization (flat dict of `top_regions` + scalars).

### Strength labels (from `_strength_template(z)`)
| `|z|` range | label |
|---|---|
| ≥ 4.0 | strong |
| ≥ 2.5 | pronounced |
| ≥ 1.5 | moderate |
| < 1.5 | weak (only reachable if threshold is lowered below 1.5) |

### Invariants worth knowing
- Cortical input must be `(T, 20484)`; subcortical `(T, 8802)`. Mismatches raise `ValueError`.
- Cortical and subcortical names are merged into one dict before aggregation — name collisions across atlases would silently overwrite, so atlas loaders must keep namespaces disjoint.
- All numeric outputs are plain `float` (not `np.float32`) so JSON serialization is lossless and stable.
- `Report.to_json()` does NOT include the input arrays — it's a thin summary; full arrays stay in `TribeOutput`.

## Critical Files for Development (parcellation focus)
1. **`tribe_backend/parcellation/unit.py`** — `GlasserParcellationUnit`, `Report`, `RegionActivation`, `WindowSizeWarning`, narrative composition. Unit under test for any output-shape change.
2. **`tribe_backend/parcellation/atlas.py`** — `load_glasser_cortical`, `load_harvard_oxford_subcortical`. Touch this when changing atlas source, label remapping, or adding new parcellations.
3. **`tribe_backend/parcellation/descriptions.json`** — parcel-name → human description. Keys must match atlas names exactly; missing keys degrade silently.
4. **`tribe_backend/contracts.py`** — `TribeOutput`, `CORTICAL_VERTICES`, `SUBCORTICAL_VOXELS`, `EXPECTED_T_PER_30S_WINDOW`. Frozen contract.
5. **`tests/parcellation/`** — `test_atlas.py`, `test_parcellate.py`, `test_aggregate.py`, `test_report.py`, `test_parcellation_real_sample.py`. Mirrors the four-layer pipeline 1:1; extend here first.

## Notable
- `pyproject.toml` markers: `unit` / `integration` / `slow` / `gpu`. Default `pytest` run hits `unit`.
- import-linter contract forbids `parcellation` from importing `mesh`, `inference`, or `control` (and vice versa).
- Current branch `feat/phase5-gpu-adapter` is orthogonal to parcellation (touches `inference/gpu_runner.py`).
- No `CLAUDE.md` in repo.
