"""Glasser HCP-MMP1 Parcellation Interpretation Unit for TRIBE v2.

Takes the raw TRIBE v2 inference output and produces named per-region
activation scores + a qualitative human-readable report.

Architecture
────────────
                TRIBE v2 output
        (T, 20484) cortical  │  (T, 8802) subcortical
                             │
              ┌──────────────┴─────────────────┐
              ▼                                 ▼
  Glasser HCP-MMP1 parcellation     Harvard-Oxford parcellation
   360 bilateral cortical ROIs        8 subcortical structures
              │                                 │
              └──────────────┬─────────────────┘
                             ▼
              368 named region activation scores
                             │
               ┌─────────────┴────────────────┐
               ▼                              ▼
       Template report               LLM narrative (Claude)
    (deterministic, fast)          (richer prose, optional)

Atlas loading is lazy — all math functions work without any atlas files.
``load_cortical_atlas()`` / ``load_subcortical_atlas()`` are only called
when you instantiate ``GlasserParcellationUnit`` and pass annotation paths,
or call the load methods explicitly.

Cortical atlas source
─────────────────────
Glasser HCP-MMP1 FreeSurfer annotation files for fsaverage5:
  lh.HCPMMP1.annot  /  rh.HCPMMP1.annot
Download: https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446
Set env vars CLEO_GLASSER_LH and CLEO_GLASSER_RH to their paths.

Subcortical atlas source
────────────────────────
Harvard-Oxford subcortical atlas (Frazier et al., 2005; Makris et al., 2006).
Fetched automatically via nilearn on first use; cached in nilearn's data dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

CORTICAL_VERTICES = 20484   # fsaverage5 (10242 LH + 10242 RH)
SUBCORTICAL_VOXELS = 8802   # Harvard-Oxford 2mm MNI space used by TRIBE

DESCRIPTIONS_PATH = Path(__file__).parent / "glasser_descriptions.json"

# nilearn Harvard-Oxford atlas label fragments → canonical names we use
_HO_LABEL_FRAGMENTS: dict[str, str] = {
    "Hippocampus":    "Hippocampus",
    "Amygdala":       "Amygdala",
    "Thalamus":       "Thalamus",
    "Caudate":        "Caudate",
    "Putamen":        "Putamen",
    "Pallidum":       "Pallidum",
    "Accumbens":      "Accumbens",
    "Lateral Ventricle": "Lateral-Ventricle",
}

_INTENSITY_THRESHOLDS = {"high": 2.0, "moderate": 1.0}

GLASSER_LLM_MODEL = "claude-sonnet-4-6"

GLASSER_SYSTEM_PROMPT = """\
You write concise neuroscience summaries for Cleo, a TRIBE v2 brain-encoding tool \
that predicts fMRI BOLD signals from multimodal stimuli.

You will be given:
- The name of the stimulus (a physical environment clip)
- The top activated brain regions (Glasser HCP-MMP1 parcels + Harvard-Oxford \
subcortical), ranked by predicted z-scored BOLD response
- Each region's common name, Glasser label, z-score, and functional role

Write 3-5 sentences describing what cognitive and emotional processes this stimulus \
is engaging, grounded in the specific regions listed. Use language appropriate for \
a clinical or research audience — not a general public. Every claim must be \
traceable to a named region. Do not speculate beyond what the activation data supports.

Output the summary only — no headings, no bullet points, no preamble.
"""

# ── atlas loading ─────────────────────────────────────────────────────────────

def load_glasser_labels(
    lh_annot: Path | str,
    rh_annot: Path | str,
) -> dict[str, np.ndarray]:
    """Parse FreeSurfer .annot files for the Glasser HCP-MMP1 parcellation.

    Returns a dict mapping each parcel label (hemisphere prefix stripped,
    e.g. ``'V1'``, ``'FFC'``) to a numpy array of vertex indices into the
    20484-long concatenated (LH first, RH second) array.
    """
    import nibabel as nib

    lh_labels, _, lh_ct = nib.freesurfer.read_annot(str(lh_annot))
    rh_labels, _, rh_ct = nib.freesurfer.read_annot(str(rh_annot))

    lh_names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in lh_ct.names]
    rh_names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in rh_ct.names]

    def _strip_prefix(name: str) -> str:
        for prefix in ("L_", "R_", "lh_", "rh_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    idx: dict[str, list[int]] = {}

    for vi, li in enumerate(lh_labels):
        if li < 0 or li >= len(lh_names):
            continue
        raw = lh_names[li]
        if not raw or raw.lower() in ("unknown", "corpuscallosum", ""):
            continue
        idx.setdefault(_strip_prefix(raw), []).append(vi)

    for vi, li in enumerate(rh_labels):
        if li < 0 or li >= len(rh_names):
            continue
        raw = rh_names[li]
        if not raw or raw.lower() in ("unknown", "corpuscallosum", ""):
            continue
        idx.setdefault(_strip_prefix(raw), []).append(vi + 10242)  # RH offset

    return {k: np.array(v, dtype=np.int32) for k, v in idx.items()}


def load_subcortical_labels() -> dict[str, np.ndarray]:
    """Fetch Harvard-Oxford subcortical atlas via nilearn and map voxels.

    Returns ``{canonical_name: voxel_indices}`` where indices point into the
    8802-long TRIBE subcortical vector (non-background voxels in ravel order).
    """
    from nilearn import datasets, image

    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    vol = image.load_img(atlas["maps"]).get_fdata().astype(int)
    flat = vol.ravel()
    label_names = [
        n.decode("utf-8") if isinstance(n, bytes) else n for n in atlas["labels"]
    ]

    # TRIBE keeps only non-background voxels (label > 0) in ravel order
    nz_labels = flat[flat > 0]  # label id for each voxel in the TRIBE 8802 vector

    idx: dict[str, np.ndarray] = {}
    for frag, canonical in _HO_LABEL_FRAGMENTS.items():
        matching_ids = [
            li for li, name in enumerate(label_names) if frag.lower() in name.lower()
        ]
        if not matching_ids:
            continue
        positions = np.flatnonzero(np.isin(nz_labels, matching_ids))
        if positions.size:
            idx[canonical] = positions.astype(np.int32)

    return idx


# ── parcellation math ─────────────────────────────────────────────────────────

def parcellate(
    output: np.ndarray,
    index_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Average vertex/voxel predictions within each named ROI.

    Parameters
    ----------
    output:
        ``(T, N)`` — TRIBE cortical or subcortical output.
    index_map:
        ``{roi_name: int_array_of_indices_into_dim1}``

    Returns
    -------
    ``{roi_name: (T,) mean timeseries}``
    """
    if output.ndim != 2:
        raise ValueError(f"output must be 2D (T, N); got shape {output.shape}")
    return {name: output[:, idx].mean(axis=1) for name, idx in index_map.items()}


def aggregate(
    parcellated: dict[str, np.ndarray],
    method: str = "mean",
    window_half: int = 2,
) -> dict[str, float]:
    """Collapse temporal dimension to a single activation score per ROI.

    Parameters
    ----------
    method:
        ``'mean'`` — sustained mean across all T frames.
        ``'peak'`` — single highest frame.
        ``'peak_window'`` — mean of ±``window_half`` frames around the peak.
    """
    if method not in ("mean", "peak", "peak_window"):
        raise ValueError(f"unknown method {method!r}; choose mean/peak/peak_window")

    scores: dict[str, float] = {}
    for name, ts in parcellated.items():
        if method == "mean":
            scores[name] = float(ts.mean())
        elif method == "peak":
            scores[name] = float(ts.max())
        else:
            pi = int(ts.argmax())
            lo = max(0, pi - window_half)
            hi = min(len(ts), pi + window_half + 1)
            scores[name] = float(ts[lo:hi].mean())
    return scores


def z_score_scores(scores: dict[str, float]) -> dict[str, float]:
    """Z-score all region activation scores against their own distribution."""
    vals = np.array(list(scores.values()), dtype=float)
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        return {k: 0.0 for k in scores}
    return {k: float((v - mu) / sigma) for k, v in scores.items()}


def rank_and_threshold(
    scores: dict[str, float],
    top_k: int = 10,
    z_threshold: float = 1.0,
) -> list[tuple[str, float]]:
    """Return top-k regions exceeding ``z_threshold``, sorted descending."""
    filtered = [(k, v) for k, v in scores.items() if v >= z_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_k]


# ── description lookup ────────────────────────────────────────────────────────

def _load_descriptions(path: Path = DESCRIPTIONS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _intensity_label(z: float) -> str:
    if z >= _INTENSITY_THRESHOLDS["high"]:
        return "high"
    if z >= _INTENSITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "mild"


def _describe_region(name: str, z: float, descs: dict[str, Any]) -> str:
    entry = descs.get(name)
    intensity = _intensity_label(z)

    if entry is None:
        return (
            f"- **{name}** (z={z:+.2f}, {intensity}): "
            f"No detailed description available for this parcel."
        )

    common = entry.get("common_name", name)
    fn = entry.get("function", "")
    impl = (
        entry.get("implications", {}).get(intensity)
        or entry.get("implications", {}).get("moderate", "")
    )
    return (
        f"- **{common}** ({name}, z={z:+.2f}, {intensity}): "
        f"{impl} Associated with {fn}."
    )


# ── report generation ─────────────────────────────────────────────────────────

def generate_template_report(
    top_regions: list[tuple[str, float]],
    stimulus_label: str = "the stimulus",
    descriptions_path: Path = DESCRIPTIONS_PATH,
) -> str:
    """Deterministic markdown report — no API calls."""
    descs = _load_descriptions(descriptions_path)

    lines = [
        "## TRIBE v2 — Cortical & Subcortical Activation Report",
        f"**Stimulus:** {stimulus_label}",
        "**Top activated regions** (z ≥ threshold, ranked by activation):\n",
    ]

    if not top_regions:
        lines.append("*No regions exceeded the activation threshold.*")
        return "\n".join(lines)

    for name, z in top_regions:
        lines.append(_describe_region(name, z, descs))

    lines += [
        "",
        "---",
        "*Predictions generated by TRIBE v2 (d'Ascoli et al., 2026 — Meta FAIR).*",
        "*Cortical surface: Glasser HCP-MMP1 parcellation (Glasser et al., 2016).*",
        "*Subcortical: Harvard-Oxford atlas.*",
        "*Values are predicted z-scored BOLD signals, not direct neural recordings.*",
    ]
    return "\n".join(lines)


def generate_llm_report(
    top_regions: list[tuple[str, float]],
    stimulus_label: str = "the stimulus",
    descriptions_path: Path = DESCRIPTIONS_PATH,
) -> str:
    """LLM-augmented narrative — richer prose, requires ANTHROPIC_API_KEY.

    Passes the ranked region list + functional descriptions to Claude and asks
    for a 3-5 sentence cohesive neuroscience summary. The score and ranking are
    computed deterministically; Claude only writes prose.
    """
    import anthropic  # lazy: keeps math importable without the SDK

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    descs = _load_descriptions(descriptions_path)

    region_lines: list[str] = []
    for name, z in top_regions:
        entry = descs.get(name, {})
        common = entry.get("common_name", name)
        fn = entry.get("function", "unknown function")
        region_lines.append(f"- {common} ({name}): z={z:+.2f} — {fn}")

    user_msg = (
        f"Stimulus: {stimulus_label}\n\n"
        "Top activated brain regions (Glasser HCP-MMP1 + Harvard-Oxford, ranked by z-score):\n"
        + "\n".join(region_lines)
        + "\n\nWrite the 3-5 sentence neuroscience summary."
    )

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=GLASSER_LLM_MODEL,
        max_tokens=400,
        system=GLASSER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    text_blocks = [b.text for b in msg.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"no text in Glasser LLM response: {msg}")
    return "\n".join(text_blocks).strip()


# ── main unit class ───────────────────────────────────────────────────────────

class GlasserParcellationUnit:
    """Full pipeline: TRIBE v2 raw output → named region scores → report.

    Instantiate with pre-computed index maps (for testing / batch use),
    or call ``load_cortical_atlas()`` / ``load_subcortical_atlas()`` to
    build them from atlas files.

    Parameters
    ----------
    cortical_index_map:
        ``{parcel_name: vertex_indices}`` for the 20484-vertex cortical surface.
        If ``None``, call ``load_cortical_atlas(lh_annot, rh_annot)`` before
        calling ``run()``.
    subcortical_index_map:
        ``{region_name: voxel_indices}`` for the 8802-voxel subcortical output.
        If ``None``, call ``load_subcortical_atlas()`` before using subcortical.
    descriptions_path:
        Path to ``glasser_descriptions.json``.
    """

    def __init__(
        self,
        cortical_index_map: dict[str, np.ndarray] | None = None,
        subcortical_index_map: dict[str, np.ndarray] | None = None,
        descriptions_path: Path = DESCRIPTIONS_PATH,
    ) -> None:
        self.cortical_idx = cortical_index_map
        self.subcortical_idx = subcortical_index_map
        self.descriptions_path = descriptions_path

    def load_cortical_atlas(self, lh_annot: Path | str, rh_annot: Path | str) -> None:
        """Parse Glasser .annot files and cache vertex index map."""
        self.cortical_idx = load_glasser_labels(lh_annot, rh_annot)

    def load_subcortical_atlas(self) -> None:
        """Fetch Harvard-Oxford atlas via nilearn and cache voxel index map."""
        self.subcortical_idx = load_subcortical_labels()

    def run(
        self,
        cortical_output: np.ndarray,
        subcortical_output: np.ndarray | None = None,
        method: str = "mean",
        top_k: int = 12,
        z_threshold: float = 1.0,
        stimulus_label: str = "the stimulus",
        llm_narrative: bool = False,
    ) -> dict[str, Any]:
        """Full pipeline: raw TRIBE output → scores + reports.

        Parameters
        ----------
        cortical_output:
            ``(T, 20484)`` predicted z-scored BOLD from TRIBE v2.
        subcortical_output:
            ``(T, 8802)`` predicted BOLD — subcortical target. Optional.
        method:
            Temporal aggregation: ``'mean'``, ``'peak'``, ``'peak_window'``.
        top_k:
            Number of top regions in the report.
        z_threshold:
            Minimum z-score (after cross-ROI z-scoring) to include.
        stimulus_label:
            Clip name shown in the report header.
        llm_narrative:
            If ``True`` and ``ANTHROPIC_API_KEY`` is set, generate a prose
            summary via Claude and add it as ``llm_report``. On failure,
            sets ``llm_report_error`` instead of raising.

        Returns
        -------
        dict with keys:
            ``cortical_scores``, ``subcortical_scores``, ``all_scores``,
            ``z_scores``, ``top_regions``, ``template_report``,
            and optionally ``llm_report`` or ``llm_report_error``.
        """
        if self.cortical_idx is None:
            raise RuntimeError(
                "Cortical atlas not loaded. "
                "Call load_cortical_atlas(lh_annot, rh_annot) first, "
                "or pass cortical_index_map at construction."
            )

        cortical_ts = parcellate(cortical_output, self.cortical_idx)
        cortical_scores = aggregate(cortical_ts, method=method)

        subcortical_scores: dict[str, float] = {}
        if subcortical_output is not None and self.subcortical_idx is not None:
            sub_ts = parcellate(subcortical_output, self.subcortical_idx)
            subcortical_scores = aggregate(sub_ts, method=method)

        all_scores = {**cortical_scores, **subcortical_scores}
        z_scores = z_score_scores(all_scores)
        top_regions = rank_and_threshold(z_scores, top_k=top_k, z_threshold=z_threshold)
        template = generate_template_report(top_regions, stimulus_label, self.descriptions_path)

        out: dict[str, Any] = {
            "cortical_scores": cortical_scores,
            "subcortical_scores": subcortical_scores,
            "all_scores": all_scores,
            "z_scores": z_scores,
            "top_regions": top_regions,
            "template_report": template,
        }

        if llm_narrative and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                out["llm_report"] = generate_llm_report(
                    top_regions, stimulus_label, self.descriptions_path
                )
            except Exception as e:  # noqa: BLE001
                out["llm_report_error"] = f"{e.__class__.__name__}: {e}"

        return out
