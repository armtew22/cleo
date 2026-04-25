# Parcellation Report — Ingestion Schema

Source: `tribe_backend.parcellation.GlasserParcellationUnit.generate_report(...) -> Report`
Serialization: `Report.to_json() -> str` (UTF-8 JSON; one report per `StimulusWindow`).

## Top-level object: `Report`

```json
{
  "top_regions": [RegionActivation, ...],
  "text": "Window summary: ...",
  "method": "window_mean",
  "z_threshold": 1.5,
  "window_id": "win_000123"
}
```

| Field         | Type                       | Nullable | Notes |
|---------------|----------------------------|----------|-------|
| `top_regions` | `RegionActivation[]`       | no       | Pre-sorted descending by `abs(z_score)`. Already filtered by `z_threshold`. May be empty. Length ≤ `top_k` (default 10). |
| `text`        | `string`                   | no       | Human-readable narrative. Empty-list sentinel: `"No significant activations detected in this window."` Otherwise starts with `"Window summary: "` and ends with `.`. |
| `method`      | `string` enum              | no       | One of `"window_mean"`, `"peak"`, `"peak_window"`. Tells you what the `z_score` represents. |
| `z_threshold` | `number` (float)           | no       | Minimum `abs(z_score)` retained. Anything weaker was dropped server-side. |
| `window_id`   | `string \| null`           | yes      | Echoes the source `StimulusWindow.window_id`. Use for joining to stimulus metadata. May be `null` if upstream did not set it. |

## `RegionActivation`

```json
{
  "name": "L_V1_ROI",
  "z_score": -2.84,
  "direction": "deactivation",
  "description": "Primary visual cortex (left hemisphere)."
}
```

| Field         | Type             | Nullable | Notes |
|---------------|------------------|----------|-------|
| `name`        | `string`         | no       | Parcel identifier. Cortical names come from Glasser-360; subcortical from Harvard-Oxford. Treat as opaque keys — namespaces are disjoint, so a single string identifies a unique region. |
| `z_score`     | `number` (float) | no       | **Signed** aggregated value. Sign carries direction; magnitude indicates strength. |
| `direction`   | `string` enum    | no       | `"activation"` if `z_score >= 0`, else `"deactivation"`. Redundant with sign of `z_score` — provided for convenience. |
| `description` | `string`         | no       | Human-readable region description. Empty string `""` when not in the descriptions table — do **not** treat `""` as null. |

## Enums

### `method`
| Value          | Meaning |
|----------------|---------|
| `window_mean`  | Mean across all timesteps in the window. Sustained activations. **Default.** |
| `peak`         | Signed value at the timestep with max `abs`. Transients. |
| `peak_window`  | Mean of the (2·radius+1)-frame sub-window with largest `abs(mean)`. Default radius=2 ⇒ 5 frames. |

### `direction`
- `"activation"`
- `"deactivation"`

## Ordering & filtering guarantees

- `top_regions` is **sorted descending by `abs(z_score)`** — index 0 is the strongest signal regardless of sign.
- All entries satisfy `abs(z_score) >= z_threshold`.
- Length ≤ `top_k` (server-side default 10). Consumers should not assume a fixed length.
- Order across reports is not stable for ties; do not rely on tie-break ordering.

## Numeric ranges

- `z_score`: unbounded float; typical magnitudes 0–8. Strength tiers used by the narrative:
  - `|z| >= 4.0` → "strong"
  - `|z| >= 2.5` → "pronounced"
  - `|z| >= 1.5` → "moderate"
  - `|z| <  1.5` → "weak" (only present if `z_threshold` was lowered below 1.5)
- `z_threshold`: float, ≥ 0. Defaults to `1.5`.
- All floats are JSON numbers (not `NaN`/`Infinity`); the producer emits plain Python `float`.

## Encoding

- JSON, UTF-8.
- Keys are stable and lowercase snake_case.
- No additional fields are emitted today, but consumers should **ignore unknown fields** for forward compatibility.

## Empty / edge cases

| Scenario                                | Shape |
|-----------------------------------------|-------|
| No region clears threshold              | `top_regions: []`, `text: "No significant activations detected in this window."` |
| Upstream produced no `window_id`        | `window_id: null` |
| Region present in atlas, missing description | `description: ""` |
| `T` mismatch (≠ 31 frames)              | Report still emitted; producer logs `WindowSizeWarning`. No flag in payload — detect upstream if you care. |

## Minimal consumer pseudocode

```python
report = json.loads(blob)
for r in report["top_regions"]:        # already ranked, already filtered
    name        = r["name"]
    signed_z    = r["z_score"]
    is_positive = r["direction"] == "activation"
    label       = r["description"] or name
    # render r…
window_id = report["window_id"]        # may be None
```

## Joining to other artifacts

- `window_id` is the join key against `StimulusWindow` (inference input) and the `mesh/` exports for the same window.
- Parcel `name` is **not** a join key against the mesh exports — mesh outputs are per-vertex, not per-parcel. Re-parcellate vertex data using the same atlas if you need alignment.

## Versioning

There is currently **no explicit schema version field** in `Report.to_json()`. If you need to pin a version, capture the producer's git commit alongside the payload. Treat any future addition of a version field as additive (ignore-unknown-fields rule above).
