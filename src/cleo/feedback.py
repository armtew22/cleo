"""Caregiver feedback generation from TRIBE ROI activations.

Takes the per-ROI sustained z-scores produced by `pipeline.aggregate_rois` and
the scene caption, then asks Claude to write a single caregiver-facing markdown
block (traffic-light icon, score 1-10, dominant-channel explanation, before/
during/distress action lists).

When TRIBE v2's subcortical head is available, callers can also pass an
optional ``subcortical_z`` dict with z-scores for ``Amygdala`` / ``Hippocampus``
/ ``Thalamus``. These compete with the cortical Destrieux groups for the
dominant-driver position and feed directly into the score, so the brief reflects
true limbic/thalamic activation rather than cortical proxies alone.

The numeric load score is derived deterministically from the top z;
the LLM only writes prose. This keeps the score auditable and lets us regenerate
phrasing without changing severity.
"""

from __future__ import annotations

import os
from typing import Any

FEEDBACK_MODEL = "claude-sonnet-4-6"

ROI_LABEL = {
    "auditory": "auditory cortex (Heschl's, planum temporale, lateral STG)",
    "visual": "visual cortex (V1, cuneus, occipital pole, fusiform)",
    "limbic_adjacent": "limbic-adjacent cortex (parahippocampal, anterior insula, temporal pole)",
}

ROI_FRIENDLY = {
    "auditory": "auditory cortex",
    "visual": "visual cortex",
    "limbic_adjacent": "limbic-adjacent cortex",
}

SUBCORTICAL_FRIENDLY = {
    "Amygdala": "amygdala (threat response)",
    "Hippocampus": "hippocampus (context & novelty)",
    "Thalamus": "thalamus (sensory gating)",
}


def compute_load_score(
    roi_stats: dict[str, dict[str, float]],
    subcortical_z: dict[str, float] | None = None,
) -> int:
    """Map cortical (+ optional subcortical) z-scores to a 1-10 load score.

    Uses the highest z across all available channels as the dominant driver:
        z=-2 → 1, z=0 → 5, z=+2 → 9, z=+2.5 → 10
    """
    candidates: list[float] = [s["z_sustained"] for s in roi_stats.values()]
    if subcortical_z:
        candidates.extend(float(v) for v in subcortical_z.values())
    top_z = max(candidates)
    raw = 5 + 2 * top_z
    return int(max(1, min(10, round(raw))))


def score_icon(score: int) -> str:
    if score <= 3:
        return "🟢"
    if score <= 6:
        return "🟡"
    return "🔴"


def _resolve_dominant(
    roi_stats: dict[str, dict[str, float]],
    ranking: list[str],
    subcortical_z: dict[str, float] | None,
) -> tuple[str, str]:
    """Return (dominant_key, friendly_label).

    Subcortical structures win ties only if their z exceeds the top cortical z.
    """
    top_cortical_key = ranking[0]
    top_cortical_z = roi_stats[top_cortical_key]["z_sustained"]

    if subcortical_z:
        sub_name, sub_z = max(subcortical_z.items(), key=lambda kv: kv[1])
        if sub_z > top_cortical_z:
            return sub_name, SUBCORTICAL_FRIENDLY.get(sub_name, sub_name)

    return top_cortical_key, ROI_FRIENDLY[top_cortical_key]


SYSTEM_PROMPT = """You write caregiver-facing sensory-load briefings for Cleo, a TRIBE v2 tool that predicts cortical and subcortical activation from a 30-second clip of a physical environment.

You will be given:
- A factual caption of the scene
- TRIBE-derived cortical z-scores (auditory cortex, visual cortex, limbic-adjacent cortex)
- (Optional) TRIBE-derived subcortical z-scores (Amygdala, Hippocampus, Thalamus)
- A pre-computed load score (1-10) — DO NOT change it
- A pre-computed traffic-light icon — use it verbatim
- The dominant channel (the structure most activated, cortical or subcortical)

Write a single markdown block in this exact structure:

## {ICON} Score: {N} / 10
**{One-line bold headline summarising the load.}**

{2-4 plain-language sentences describing what is happening in this specific environment. Reference concrete observables from the caption (e.g. "concrete floors", "fluorescent lighting", "open-plan layout"). Name the dominant channel and why it is the driver. If subcortical activation is high, name the structure (amygdala / hippocampus / thalamus) plainly — these matter more than the cortical proxies when present.}

**Before you go in:**
- {3-4 bullets matched to the dominant channel and the scene}

**While you're inside:**
- {2-3 bullets — positioning, what to avoid, what to monitor}

**Watch for:** {one sentence on early-warning signs of overload}.

{One closing line: visit-duration suggestion, or an alternative time/place hint if score ≥ 6.}

*Cleo v0.1 | Powered by TRIBE v2 — Meta FAIR*
*Not a clinical tool — always trust your own knowledge of the person you support*

Style rules:
- Address "the person you support" — never assume a name, age, or relationship
- Calm, factual, non-judgmental. Avoid loaded words ("overwhelming", "stressful", "scary", "chaotic")
- Reference scene specifics from the caption — generic advice is useless
- Match recommendations to the dominant channel:
    auditory cortex → noise-reducing headphones, acoustic positioning (carpet zones, away from kitchens/PAs), predictable arrival
    visual cortex → lighting choices (avoid flicker/mixed colour temperature), clear sightlines, reduce motion exposure
    limbic-adjacent cortex → predictability, named exit route, time-boxed visits, advance planning
    amygdala (threat response) → keep your own voice low and even, advance preparation, slow pace, frequent reassurance, rehearse the exit route
    hippocampus (context & novelty) → use a familiar route, show photos of the space in advance, avoid first-time visits at peak hours
    thalamus (sensory gating) → headphones AND tinted lenses if needed, reduce parallel inputs (don't talk while walking), single-focus tasks only
- Never invent details not in the caption (no "PA system" unless the caption mentions one)
- Use the icon and score exactly as given — do not recompute
- Output the markdown block only. No preamble, no follow-up commentary.
"""


def _format_roi_summary(
    roi_stats: dict[str, dict[str, float]],
    ranking: list[str],
    subcortical_z: dict[str, float] | None,
) -> str:
    lines = ["Cortical activation (sustained z-score vs. whole-cortex baseline):"]
    for g in ranking:
        s = roi_stats[g]
        lines.append(
            f"- {ROI_LABEL[g]}: z_sustained={s['z_sustained']:+.2f}, "
            f"z_peak={s['z_peak']:+.2f}"
        )
    if subcortical_z:
        lines.append("")
        lines.append("Subcortical activation (z-score vs. whole-brain regions):")
        for name, z in sorted(subcortical_z.items(), key=lambda kv: kv[1], reverse=True):
            label = SUBCORTICAL_FRIENDLY.get(name, name)
            lines.append(f"- {label}: z={float(z):+.2f}")
    return "\n".join(lines)


def generate_feedback(
    caption: str,
    roi_stats: dict[str, dict[str, float]],
    ranking: list[str],
    subcortical_z: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate caregiver markdown from caption + TRIBE ROI stats.

    Parameters
    ----------
    caption:
        Objective scene description from ``captioning.caption_clip``.
    roi_stats:
        Destrieux 3-group cortical stats from ``pipeline.aggregate_rois``.
    ranking:
        Cortical groups ranked by ``z_sustained`` desc.
    subcortical_z:
        Optional ``{Amygdala|Hippocampus|Thalamus: z_score}``. When present
        these compete with cortical groups for dominant-channel position and
        feed into the score.
    """
    import anthropic  # lazy: keeps math importable without the SDK

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    score = compute_load_score(roi_stats, subcortical_z)
    icon = score_icon(score)
    dominant_key, dominant_label = _resolve_dominant(roi_stats, ranking, subcortical_z)
    roi_summary = _format_roi_summary(roi_stats, ranking, subcortical_z)

    user_message = (
        f"SCENE CAPTION:\n{caption}\n\n"
        f"TRIBE ACTIVATION:\n{roi_summary}\n\n"
        f"PRE-COMPUTED SCORE: {score} / 10\n"
        f"ICON: {icon}\n"
        f"DOMINANT CHANNEL: {dominant_label}\n\n"
        "Write the caregiver markdown block per the template."
    )

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=FEEDBACK_MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text_blocks = [b.text for b in msg.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"no text in feedback response: {msg}")
    markdown = "\n".join(text_blocks).strip()

    return {
        "score": score,
        "icon": icon,
        "dominant_roi": dominant_key,
        "dominant_label": dominant_label,
        "markdown": markdown,
    }
