# Cleo — TRIBE v2 sensory-load pipeline

Offline ML pipeline for an environment-scout app for autistic adults / SPD users.
Takes a 30-second mp4 clip of a physical space, runs Meta's
[TRIBE v2](https://github.com/facebookresearch/tribev2) multimodal brain-encoding
model on it, and emits two layers of output:

1. **Caregiver layer** — a 1-10 sensory-load score + actionable markdown brief
   for the person supporting a visit (written by Claude, grounded in TRIBE predictions).
2. **Neuroscience layer** — 368 named region activation scores (Glasser 360-parcel
   cortical + 8 Harvard-Oxford subcortical) with a structured report and optional
   LLM narrative.

## What TRIBE v2 predicts

TRIBE v2 outputs predicted fMRI BOLD signals on the **fsaverage5 cortical surface**
(20,484 vertices) and in **8,802 subcortical voxels** (Harvard-Oxford atlas, 2mm MNI):

| Output | Shape | Space |
|--------|-------|-------|
| Cortical | `(T, 20484)` | fsaverage5 surface (LH + RH) |
| Subcortical | `(T, 8802)` | Harvard-Oxford 8 bilateral structures |

Values are predicted z-scored BOLD — relative activation above baseline,
not raw electrical recordings.

**Cortical ROI groups (Destrieux, for caregiver score):**
- **Auditory** — primary auditory cortex (Heschl's) + auditory belt
- **Visual** — V1, early visual, lateral occipital, fusiform
- **Limbic-adjacent** — parahippocampal, entorhinal-adjacent, anterior insula,
  temporal pole

**Glasser parcels (for neuroscience report):**
- 360 bilateral cortical parcels (Glasser HCP-MMP1 parcellation)
- 8 subcortical structures: Hippocampus, Amygdala, Thalamus, Caudate, Putamen,
  Pallidum, Accumbens, Lateral Ventricle

> ⚠️ The Destrieux limbic-adjacent group is a **cortical proxy** for limbic
> involvement — not a direct subcortical readout. The Glasser unit provides
> actual subcortical predictions (amygdala, hippocampus) via TRIBE's subcortical
> output head.

## Install (on the GPU box)

Floor: ~24 GB VRAM. TRIBE bundles LLaMA-3.2-3B, V-JEPA2, and Wav2Vec-BERT.

```bash
sudo apt-get install -y ffmpeg

git clone https://github.com/facebookresearch/tribev2.git
cd tribev2 && pip install -e . && cd ..

git clone <this repo> cleo
cd cleo && pip install -e .

export ANTHROPIC_API_KEY=sk-ant-...
```

First inference downloads model weights from HuggingFace (~5 min on fast link).

### Optional: Glasser atlas (for 360-parcel neuroscience report)

```bash
# Download lh.HCPMMP1.annot and rh.HCPMMP1.annot from:
# https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446

export CLEO_GLASSER_LH=/path/to/lh.HCPMMP1.annot
export CLEO_GLASSER_RH=/path/to/rh.HCPMMP1.annot
```

## Run

```bash
cleo-forecast --video clip.mp4 --out forecast.json
```

Outputs:
- `forecast.json` — full pipeline JSON
- `forecast.md` — caregiver markdown (🟢🟡🔴 score, before/during/distress tips)
- `forecast_glasser.md` — neuroscience report (360 Glasser + 8 subcortical, if atlas set)

### Flags

| Flag | Effect |
|------|--------|
| `--no-feedback` | Skip caregiver brief (TRIBE + caption only) |
| `--no-glasser` | Skip Glasser parcellation |
| `--glasser-llm` | Add LLM prose narrative to the Glasser report |
| `--md-out PATH` | Override caregiver markdown output path |

## Output format

```jsonc
{
  "video": "/abs/path/clip.mp4",
  "caption": "A room approximately 12m by 8m with concrete floors...",
  "preds_shape": {
    "cortical": [30, 20484],
    "subcortical": [30, 8802]
  },
  "roi_stats": {
    "auditory":        {"z_sustained": 1.74, "z_peak": 2.12, ...},
    "visual":          {"z_sustained": 0.41, ...},
    "limbic_adjacent": {"z_sustained": 0.18, ...}
  },
  "ranking_by_sustained_z": ["auditory", "visual", "limbic_adjacent"],
  "feedback": {
    "score": 8,
    "icon": "🔴",
    "dominant_roi": "auditory",
    "markdown": "## 🔴 Score: 8 / 10 ..."
  },
  "glasser": {
    "top_regions": [["A1", 2.8], ["Amygdala", 2.3], ...],
    "template_report": "## TRIBE v2 — Cortical & Subcortical Activation Report ...",
    "llm_report": "The stimulus engages primary auditory cortex ..."  // if --glasser-llm
  }
}
```

## Verification (once GPU is available)

Drop two contrasting 30 s clips into `data/test_clips/` (e.g. quiet library + busy café),
run both, compare. Auditory `z_sustained` should be substantially higher in the busy
clip. If it isn't, check parcel names against the Destrieux atlas:

```python
import cleo.rois
print(cleo.rois.load_atlas().parcel_names)
```

For Glasser: confirm that faces → FFC (fusiform face complex) is top-ranked,
scenes → PHA1/2/3 (parahippocampal), speech → A5/STSva, emotionally charged clips
→ Amygdala. This replicates the IBC localizer validation from the TRIBE v2 paper
(Sections 2.5 and 2.6).

## What's deliberately not built

- **Reference-clip calibration.** Scores are within-clip z-scored, not anchored
  against a library of canonical spaces. Add: process e.g. `library.mp4`, `cafe.mp4`
  once, cache their stats, report new clips as percentiles. ~30–60 min of GPU.
- **HTTP API.** Wrapping `run_forecast` in FastAPI is ~30 lines.
- **CI.** `python -m unittest test_pipeline test_feedback test_glasser` runs locally;
  wiring to GitHub Actions is straightforward.
- **App / auth / storage.** Out of pipeline scope.

## Layout

```
src/cleo/
  captioning.py            # ffmpeg frame sampling → Claude vision → objective caption
  rois.py                  # Destrieux parcel groups + atlas loader (caregiver layer)
  pipeline.py              # demux → caption → TRIBE → ROI aggregation + Glasser
  feedback.py              # Destrieux z-scores → 1-10 load score + caregiver markdown
  glasser.py               # Glasser 360-parcel + Harvard-Oxford subcortical unit
  glasser_descriptions.json  # functional descriptions for 70+ named regions
  forecast.py              # CLI entrypoint

tests (no GPU required):
  test_pipeline.py         # aggregate_rois, rank_groups
  test_feedback.py         # compute_load_score, score_icon, generate_feedback (mocked)
  test_glasser.py          # parcellate, aggregate, z_score, rank, template report, unit.run
  test_captions.py         # manual dev script (needs sample_video.mp4 + API key)
```
