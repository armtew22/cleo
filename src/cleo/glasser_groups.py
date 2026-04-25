"""Map Glasser HCP-MMP1 parcel names → Destrieux ROI groups.

The remote TRIBE backend returns top-activated regions in Glasser names like
``L_FFC_ROI`` / ``R_V1_ROI``. Our caregiver-feedback layer expects activations
bucketed into three Destrieux groups: ``auditory``, ``visual``,
``limbic_adjacent``. This module is the translation table.

Naming convention upstream: ``{L|R}_{parcel}_ROI``. We strip the hemisphere
prefix and the ``_ROI`` suffix to get the bare parcel name (e.g. ``FFC``).

References:
- Glasser et al. 2016, "A multi-modal parcellation of human cerebral cortex"
- https://balsa.wustl.edu/study/show/RVVG (parcel descriptions)
"""

from __future__ import annotations

# Auditory cortex: primary (A1, MBelt), belt/parabelt, lateral STG/STS
_AUDITORY = {
    "A1", "A4", "A5",
    "LBelt", "MBelt", "PBelt", "RI", "TA2",
    "STGa", "STSda", "STSdp", "STSva", "STSvp",
}

# Visual cortex: V1-V8, MT/MST complex, fusiform/face, parahippocampal place,
# ventral/dorsal stream extensions
_VISUAL = {
    "V1", "V2", "V3", "V4",
    "V3A", "V3B", "V3CD", "V4t", "V6", "V6A", "V7", "V8",
    "LO1", "LO2", "LO3",
    "MT", "MST", "FST",
    "FFC", "PIT", "VVC",
    "PHA1", "PHA2", "PHA3",
    "VMV1", "VMV2", "VMV3",
    "DVT", "ProS",
    "PH",
}

# Limbic-adjacent cortex: insula, parahippocampal/entorhinal, temporal pole,
# orbitofrontal, anterior cingulate (ACC)
_LIMBIC = {
    "EC", "PreS", "PeEc", "Pir",
    "TGd", "TGv", "TF",
    "AAIC", "MI", "AVI", "PI", "Ig",
    "FOP1", "FOP2", "FOP3", "FOP4", "FOP5",
    "OFC", "pOFC", "13l", "11l", "47l", "47m", "47s", "a47r", "p47r",
    "25",
    "s32", "a32pr", "p32", "p32pr", "d32",
    "a24", "a24pr", "p24", "p24pr",
    "10pp", "10v", "10r", "10d",
}

# Combined lookup table — built once at import.
_GLASSER_TO_GROUP: dict[str, str] = {}
for _name in _AUDITORY:
    _GLASSER_TO_GROUP[_name] = "auditory"
for _name in _VISUAL:
    _GLASSER_TO_GROUP[_name] = "visual"
for _name in _LIMBIC:
    _GLASSER_TO_GROUP[_name] = "limbic_adjacent"


def normalize_glasser_name(name: str) -> str:
    """Strip ``L_``/``R_`` hemisphere prefix and ``_ROI`` suffix.

    >>> normalize_glasser_name("L_FFC_ROI")
    'FFC'
    >>> normalize_glasser_name("R_V1_ROI")
    'V1'
    >>> normalize_glasser_name("FFC")  # already bare
    'FFC'
    """
    s = name
    for prefix in ("L_", "R_", "lh_", "rh_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("_ROI"):
        s = s[:-4]
    return s


def group_for_region(name: str) -> str | None:
    """Return ``'auditory' | 'visual' | 'limbic_adjacent'`` or ``None`` if unmapped.

    Unmapped regions (e.g. somatomotor, frontal control parcels) are deliberately
    dropped from the caregiver-load score — they don't correspond to the three
    sensory channels the brief is structured around.
    """
    return _GLASSER_TO_GROUP.get(normalize_glasser_name(name))


def aggregate_top_regions(
    top_regions: list[dict],
) -> dict[str, dict[str, float]]:
    """Translate upstream ``top_regions[]`` into our 3-group ``roi_stats`` shape.

    Each entry in ``top_regions`` should have ``name`` and ``z_score`` (matching
    the TRIBE backend response). We bucket regions by Destrieux group and
    compute per-group peak/sustained — which feed directly into
    ``feedback.compute_load_score`` and ``feedback.generate_feedback``.

    Empty groups get all-zero stats (so a group with no upstream activation
    above threshold scores 0, not negative).
    """
    buckets: dict[str, list[float]] = {
        "auditory": [],
        "visual": [],
        "limbic_adjacent": [],
    }
    for r in top_regions:
        name = r.get("name") or ""
        z = r.get("z_score")
        if z is None:
            continue
        g = group_for_region(name)
        if g is not None:
            buckets[g].append(float(z))

    out: dict[str, dict[str, float]] = {}
    for g, zs in buckets.items():
        if zs:
            peak = max(zs)
            sustained = sum(zs) / len(zs)
        else:
            peak = 0.0
            sustained = 0.0
        out[g] = {
            "n_vertices": len(zs),
            "peak": peak,
            "sustained": sustained,
            "z_peak": peak,
            "z_sustained": sustained,
        }
    return out
