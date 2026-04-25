"""ROI definitions on the fsaverage5 cortical mesh (Destrieux 2009 a2009s atlas).

TRIBE v2 outputs `preds: (T, 20484)` on fsaverage5 (10242 LH + 10242 RH vertices,
LH first then RH concatenated — the field convention). This module groups a
small set of Destrieux parcels into ROI families relevant to sensory load:

  - Auditory: primary (Heschl's) + auditory belt (planum temporale/polare,
    lateral STG)
  - Visual:   V1 (calcarine), early visual (cuneus), occipital pole, lateral
    occipital, fusiform
  - Limbic-adjacent (cortical proxies for hippocampus/amygdala):
    parahippocampal, lingual/entorhinal-adjacent, anterior insula, temporal
    pole

`PARCELS_BY_GROUP` maps each group name to a list of Destrieux parcel names
exactly as they appear in the atlas's `labels` array.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

LH_VERTEX_COUNT = 10242
RH_VERTEX_COUNT = 10242
TOTAL_VERTICES = LH_VERTEX_COUNT + RH_VERTEX_COUNT  # 20484

PARCELS_BY_GROUP: dict[str, list[str]] = {
    "auditory": [
        "G_temp_sup-G_T_transv",   # Heschl's gyrus (primary auditory)
        "G_temp_sup-Plan_tempo",   # planum temporale (auditory belt)
        "G_temp_sup-Plan_polar",   # planum polare (auditory belt)
        "G_temp_sup-Lateral",      # lateral STG (parabelt)
    ],
    "visual": [
        "S_calcarine",             # V1
        "G_cuneus",                # early dorsal visual
        "Pole_occipital",          # occipital pole
        "G_occipital_middle",      # lateral occipital
        "G_oc-temp_lat-fusifor",   # fusiform (incl. FFA region)
    ],
    "limbic_adjacent": [
        "G_oc-temp_med-Parahip",        # parahippocampal gyrus (hippocampal proxy)
        "S_oc-temp_med_and_Lingual",    # lingual/entorhinal-adjacent
        "G_insular_short",              # short insular gyri (anterior insula)
        "G_Ins_lg_and_S_cent_ins",      # long insular gyrus + central insular sulcus
        "Pole_temporal",                # temporal pole
    ],
}


@dataclass(frozen=True)
class Atlas:
    """Destrieux atlas labels per fsaverage5 vertex.

    `labels` is the parcel name per vertex in the concatenated (LH+RH) order.
    """

    labels: np.ndarray         # shape (20484,), dtype object/str
    parcel_names: list[str]    # the unique parcel name set


@lru_cache(maxsize=1)
def load_atlas() -> Atlas:
    """Fetch Destrieux 2009 surface atlas in fsaverage5 space (cached)."""
    from nilearn import datasets  # local import — heavy

    destrieux = datasets.fetch_atlas_surf_destrieux()
    raw_labels = [
        n.decode("utf-8") if isinstance(n, bytes) else n
        for n in destrieux["labels"]
    ]
    map_left = np.asarray(destrieux["map_left"])
    map_right = np.asarray(destrieux["map_right"])

    if map_left.shape[0] != LH_VERTEX_COUNT or map_right.shape[0] != RH_VERTEX_COUNT:
        raise RuntimeError(
            f"Destrieux maps not on fsaverage5: got "
            f"LH={map_left.shape[0]}, RH={map_right.shape[0]}; "
            f"expected {LH_VERTEX_COUNT} each"
        )

    lh_names = np.array(raw_labels, dtype=object)[map_left]
    rh_names = np.array(raw_labels, dtype=object)[map_right]
    labels = np.concatenate([lh_names, rh_names])

    return Atlas(labels=labels, parcel_names=raw_labels)


def vertex_indices_for_group(group: str) -> np.ndarray:
    """Return vertex indices (into the 20484-long concatenated array) for a group."""
    if group not in PARCELS_BY_GROUP:
        raise KeyError(f"unknown group {group!r}; known: {list(PARCELS_BY_GROUP)}")

    atlas = load_atlas()
    parcel_set = PARCELS_BY_GROUP[group]
    missing = [p for p in parcel_set if p not in atlas.parcel_names]
    if missing:
        raise RuntimeError(
            f"parcel names not found in Destrieux atlas: {missing}. "
            f"Compare against atlas.parcel_names."
        )
    mask = np.isin(atlas.labels, parcel_set)
    return np.flatnonzero(mask)


def all_group_vertex_indices() -> dict[str, np.ndarray]:
    """Convenience: { group_name -> vertex indices } for every defined group."""
    return {g: vertex_indices_for_group(g) for g in PARCELS_BY_GROUP}
