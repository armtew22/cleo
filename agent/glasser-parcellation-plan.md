# Glasser Parcellation Interpretation Unit for TRIBE v2

## Implementation Plan

**Objective:** Build a module that takes the raw inference output of TRIBE v2 and produces human-readable qualitative reports such as: *"This environment triggers emotional reactivity in the amygdala, which may increase threat vigilance and affective memory encoding."*

---

## 1. Understanding the Model Output (Detailed)

### 1.1 What TRIBE v2 Produces

TRIBE v2 is a transformer-based encoding model that predicts fMRI BOLD signals from multimodal stimuli (video, audio, text). At inference time, given a stimulus window, the model outputs a **dense prediction of brain activity across the entire brain at 1 Hz temporal resolution**.

The output is split into two spatial targets:

#### Cortical Output

- **Shape:** `(T, 20484)` — a time series of `T` seconds, each with 20,484 scalar values.
- **Space:** The 20,484 values correspond to vertices on the **fsaverage5 cortical surface** mesh (Jenkinson et al., 2012). This is a standardized brain surface representation used in FreeSurfer, where each vertex is a point on the folded cortical sheet.
- **Value:** Each scalar is a predicted z-scored BOLD signal — the blood-oxygen-level-dependent response measured by fMRI. Positive values indicate predicted activation above baseline; negative values indicate suppression.
- **Coordinate system:** The vertices are in fsaverage5 surface space, split into left hemisphere (10,242 vertices) and right hemisphere (10,242 vertices).

#### Subcortical Output

- **Shape:** `(T, 8802)` — same temporal resolution, 8,802 voxels.
- **Space:** The 8,802 voxels span 8 subcortical structures defined by the **Harvard-Oxford atlas** (Frazier et al., 2005; Makris et al., 2006; Desikan et al., 2006) at 2mm MNI resolution.
- **Structures:** Hippocampus, Lateral Ventricles, Amygdala, Thalamus, Caudate, Putamen, Pallidum, Accumbens.

#### How the Output Is Generated (Code Path)

From the codebase (`https://github.com/facebookresearch/tribev2`):

1. **Feature extraction** (frozen, cached): Stimuli pass through three pretrained models:
   - **Text → Llama-3.2-3B** (Grattafiori et al., 2024): Contextualized word embeddings, `D_text = 2048`, at 2 Hz.
   - **Audio → Wav2Vec-Bert-2.0** (Chung et al., 2021): Audio embeddings, `D_audio = 1024`, resampled from 50 Hz to 2 Hz.
   - **Video → V-JEPA-2-Giant** (Assran et al., 2025): Spatially averaged patch embeddings, `D_video = 1280`, at 2 Hz.

2. **Modality fusion**: Each modality's multi-layer embeddings are grouped into `L` groups, averaged, projected to `D=384` via a linear layer + layer norm, then concatenated → `D_model = 3 × 384 = 1152`.

3. **Transformer encoder**: 8-layer, 8-head transformer with learnable positional + subject embeddings. Input window: `T=100s` at `f_stim=2 Hz` (200 timesteps). Output is adaptively pooled from 2 Hz → 1 Hz (`f_fMRI`).

4. **Subject block**: A subject-conditional linear layer projects the transformer output to the target space. For zero-shot / in-silico use, the "unseen subject" linear layer is used instead (trained with subject dropout, `p=0.1`).

5. **Final output**: `(T, N_targets)` where `N_targets = 20484` (cortical) or `8802` (subcortical). Values are predicted z-scored BOLD signals.

#### Key Implementation Detail: Hemodynamic Lag

The fMRI timeseries is offset by **5 seconds** relative to stimuli. To predict brain activity in the window `[0, T]`, the model receives stimuli from `[-5, T-5]`. This accounts for the hemodynamic delay between neural firing and the BOLD response peak.

### 1.2 What the Output Means Physiologically

Each output value represents predicted **blood oxygenation** at a spatial location. This is an indirect measure of neural activity — when neurons fire, local blood flow increases to deliver oxygen, producing the BOLD signal. The z-scoring normalizes within each session, so values represent relative activation compared to baseline.

---

## 2. The Glasser Parcellation Mapping Layer

### 2.1 What It Does

The Glasser Multi-Modal Parcellation (Glasser et al., 2016) divides each hemisphere into **180 regions** (360 total). Each region has a neuroscience-grounded label (e.g., `FFC` = fusiform face complex, `V1` = primary visual cortex, `44` = Broca's area).

This module will:

1. Load the Glasser parcellation labels for fsaverage5.
2. For each of the 360 parcels, average all vertex-level predictions that fall within that parcel.
3. Combine with the 8 subcortical ROIs (already defined by Harvard-Oxford atlas regions).
4. Produce a vector of **368 named region activations** per time point.

### 2.2 Parcellation Lookup Table

The mapping from Glasser labels to functional descriptions is the interpretive core. Example entries:

| Glasser Label | Common Name | Functional Description |
|---|---|---|
| `FFC` | Fusiform Face Complex (FFA) | Face perception, identity recognition, social evaluation |
| `PHA1/2/3` | Parahippocampal Area (PPA) | Scene/place processing, spatial context, environmental encoding |
| `V4t` | Extrastriate Body Area (EBA) | Body part perception, biological motion |
| `TPOJ1/2/3` | Temporo-Parietal-Occipital Junction | Multisensory integration, spatial attention, self-other distinction |
| `PGi` | Temporo-Parietal Junction (TPJ) | Theory of mind, empathy, social cognition |
| `44` / `45` | Broca's Area | Speech production, syntactic processing, verbal working memory |
| `A1` | Primary Auditory Cortex | Basic sound processing, tonotopic representation |
| `A5` | Associative Auditory Cortex | Higher-order auditory processing, speech perception |
| `STSvp/STSva` | Superior Temporal Sulcus | Voice processing, audiovisual integration, social perception |
| `V1/V2/V3` | Early Visual Cortex | Edge detection, orientation, spatial frequency |
| `TE1a` | Visual Word Form Area (VWFA) | Reading, orthographic processing |
| `PEF` | Parietal Eye Fields | Saccadic planning, spatial attention |
| --- | **Subcortical (Harvard-Oxford)** | --- |
| `Amygdala` | Amygdala | Emotional reactivity, threat detection, fear conditioning, valence tagging |
| `Hippocampus` | Hippocampus | Episodic memory encoding, spatial navigation, contextual binding |
| `Thalamus` | Thalamus | Sensory relay, arousal regulation, attentional gating |
| `Caudate` | Caudate Nucleus | Reward processing, goal-directed behavior, habit formation |
| `Putamen` | Putamen | Motor execution, procedural learning, reward prediction |
| `Pallidum` | Globus Pallidus | Motor output modulation, action selection |
| `Accumbens` | Nucleus Accumbens | Reward, motivation, pleasure, reinforcement learning |

A comprehensive version of this table (all 360 + 8 regions) should be maintained as a JSON/YAML config file in the module.

---

## 3. Implementation Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TRIBE v2 Inference                │
│  Input: video + audio + text stimulus               │
│  Output: (T, 20484) cortical + (T, 8802) subcort.   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           Glasser Parcellation Unit                 │
│                                                     │
│  1. Vertex → Parcel averaging (360 cortical ROIs)   │
│  2. Voxel → Region averaging (8 subcortical ROIs)   │
│  3. Temporal aggregation (mean / peak / window)     │
│  4. Thresholding & ranking                          │
│  5. Lookup table → functional descriptions          │
│  6. Narrative generation (template or LLM)          │
│                                                     │
│  Output: Qualitative report string                  │
└─────────────────────────────────────────────────────┘
```

### 3.1 Module: `glasser_parcellation_unit.py`

```python
# Pseudocode / structural outline

class GlasserParcellationUnit:
    def __init__(self, parcellation_path, roi_descriptions_path):
        """
        Args:
            parcellation_path: Path to Glasser parcellation labels
                for fsaverage5 (e.g., lh.HCPMMP1.annot, rh.HCPMMP1.annot)
            roi_descriptions_path: Path to JSON mapping
                parcel labels → functional descriptions
        """
        self.parcel_labels = load_glasser_labels(parcellation_path)
        # shape: (20484,) with integer labels 0-360
        self.subcortical_mask = load_harvard_oxford_mask()
        # shape: (8802,) with integer labels 0-8
        self.roi_descriptions = load_json(roi_descriptions_path)

    def parcellate_cortical(self, cortical_output):
        """
        Args:
            cortical_output: np.ndarray of shape (T, 20484)
        Returns:
            dict[str, np.ndarray]: parcel_name → (T,) time series
        """
        result = {}
        for parcel_id in range(1, 361):
            mask = self.parcel_labels == parcel_id
            parcel_name = self.id_to_name[parcel_id]
            result[parcel_name] = cortical_output[:, mask].mean(axis=1)
        return result

    def parcellate_subcortical(self, subcortical_output):
        """
        Args:
            subcortical_output: np.ndarray of shape (T, 8802)
        Returns:
            dict[str, np.ndarray]: region_name → (T,) time series
        """
        result = {}
        for region_id, region_name in enumerate(SUBCORTICAL_NAMES):
            mask = self.subcortical_mask == region_id
            result[region_name] = subcortical_output[:, mask].mean(axis=1)
        return result

    def aggregate(self, parcellated, method="mean"):
        """
        Collapse temporal dimension → single activation score per ROI.
        Methods: 'mean', 'peak', 'peak_window' (5s around max)
        """
        scores = {}
        for name, ts in parcellated.items():
            if method == "mean":
                scores[name] = ts.mean()
            elif method == "peak":
                scores[name] = ts.max()
            elif method == "peak_window":
                peak_idx = ts.argmax()
                window = ts[max(0, peak_idx-2):peak_idx+3]
                scores[name] = window.mean()
        return scores

    def rank_and_threshold(self, scores, top_k=10, z_threshold=1.5):
        """
        Return the top-k regions exceeding the z-score threshold.
        """
        filtered = {k: v for k, v in scores.items() if v > z_threshold}
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def generate_report(self, cortical_output, subcortical_output,
                        method="mean", top_k=10, z_threshold=1.5):
        """
        Full pipeline: raw output → qualitative report.
        """
        cortical_parcels = self.parcellate_cortical(cortical_output)
        subcortical_parcels = self.parcellate_subcortical(subcortical_output)

        all_parcels = {**cortical_parcels, **subcortical_parcels}
        scores = self.aggregate(all_parcels, method=method)
        top_regions = self.rank_and_threshold(scores, top_k, z_threshold)

        report_lines = []
        for region_name, score in top_regions:
            desc = self.roi_descriptions[region_name]
            intensity = "strong" if score > 3.0 else "moderate" if score > 2.0 else "mild"
            report_lines.append(
                f"- **{desc['common_name']}** ({region_name}): "
                f"{intensity} activation (z={score:.2f}). "
                f"Associated with {desc['function']}."
            )

        return "\n".join(report_lines)
```

### 3.2 Module: `roi_descriptions.json` (excerpt)

```json
{
  "FFC": {
    "common_name": "Fusiform Face Complex",
    "function": "face perception, identity recognition, and social evaluation",
    "network": "ventral visual",
    "implications": {
      "high": "The stimulus contains salient face information, likely triggering social cognitive processing and identity evaluation circuits.",
      "moderate": "Some face-like or socially relevant features are present in the stimulus."
    }
  },
  "Amygdala": {
    "common_name": "Amygdala",
    "function": "emotional reactivity, threat detection, fear conditioning, and affective valence tagging",
    "network": "limbic",
    "implications": {
      "high": "This environment triggers strong emotional reactivity in the amygdala, which may heighten threat vigilance, strengthen affective memory encoding, and modulate autonomic arousal responses.",
      "moderate": "Mild emotional salience detected, suggesting the stimulus carries affective weight without triggering a full threat response."
    }
  },
  "Hippocampus": {
    "common_name": "Hippocampus",
    "function": "episodic memory encoding, spatial navigation, and contextual binding",
    "network": "medial temporal",
    "implications": {
      "high": "The stimulus engages episodic memory systems, suggesting the environment is being encoded into long-term memory with rich contextual associations.",
      "moderate": "Some contextual or spatial features are being processed for memory integration."
    }
  }
}
```

---

## 4. Dependencies & Data Files Required

### 4.1 From the TRIBE v2 Codebase

- **Repository:** `https://github.com/facebookresearch/tribev2`
- **Model weights:** `https://huggingface.co/facebook/tribev2`
- **Key files to inspect:**
  - Model forward pass / inference script (defines `(T, 20484)` and `(T, 8802)` output tensors)
  - Subject block logic (unseen subject mode for group-level predictions)
  - The adaptive average pooling layer that decimates from 2 Hz → 1 Hz
  - Modality dropout logic (for single-modality inference)

### 4.2 Parcellation Files

- **Glasser HCP-MMP1 for fsaverage5:**
  - `lh.HCPMMP1.annot` and `rh.HCPMMP1.annot` — FreeSurfer annotation files mapping each of the 10,242 vertices per hemisphere to one of 180 parcels.
  - Available via: `https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446`
  - Also obtainable via `nilearn.datasets.fetch_atlas_surf_destrieux()` or `mne.datasets`.

- **Harvard-Oxford Subcortical Atlas:**
  - Included in FSL (`https://fsl.fmrib.ox.ac.uk/fsl/fslwiki`)
  - 2mm MNI resolution, 8 bilateral structures.
  - Also available via `nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')`.

### 4.3 Python Dependencies

```
numpy
nilearn          # Surface operations, atlas fetching, GLM
nibabel          # NIfTI / FreeSurfer annotation I/O
scipy            # Stats, spatial correlation
scikit-learn     # ICA (optional, for network decomposition)
torch            # TRIBE v2 model inference
```

---

## 5. Report Generation Strategies

### 5.1 Template-Based (Fast, Deterministic)

Use the `implications` field from `roi_descriptions.json` keyed by activation intensity. This produces consistent, citation-ready language with no API calls.

**Example output:**
> **Top activated regions for stimulus "urban_traffic_scene_01":**
>
> - **Amygdala** (z=3.41, strong): This environment triggers strong emotional reactivity in the amygdala, which may heighten threat vigilance, strengthen affective memory encoding, and modulate autonomic arousal responses.
> - **V5/MT Complex** (z=2.87, moderate): Motion processing circuits are engaged, indicating salient dynamic visual content in the stimulus.
> - **Superior Temporal Sulcus** (z=2.63, moderate): Audiovisual integration and social perception systems are active, suggesting the presence of voices or biological motion.

### 5.2 LLM-Augmented (Richer, Flexible)

Pass the ranked activation list + ROI descriptions to an LLM to generate a cohesive narrative paragraph. Useful for non-technical audiences.

**Prompt template:**
```
Given the following brain region activations predicted by TRIBE v2
for the stimulus "{stimulus_name}":

{ranked_regions_with_scores}

Write a 3-5 sentence summary describing what cognitive and emotional
processes this stimulus is likely engaging, grounded in the
neuroscience of each region. Use accessible language.
```

---

## 6. Validation Plan

1. **Replicate IBC visual localizers** (Section 2.5 of the paper): Run faces, places, bodies, characters, tools through the pipeline. Confirm that the top-ranked parcels match the expected ROIs (FFA for faces, PPA for places, etc.).

2. **Replicate IBC language contrasts** (Section 2.6): Confirm that speech vs. silence ranks A5/STS/45 highest; emotional vs. physical pain ranks TPJ/MTG highest.

3. **Subcortical sanity checks**: Confirm amygdala activation is higher for emotionally charged stimuli (e.g., threat scenes) than neutral ones.

4. **Cross-reference with NeuroSynth** (Section 2.7): Correlate the parcellated activation maps with NeuroSynth meta-analytic maps for keywords like "emotion", "face", "language" to validate the functional labeling.

---

## 7. References

- **Paper:** d'Ascoli, S., et al. (2026). *A foundation model of vision, audition, and language for in-silico neuroscience.* FAIR at Meta. Published March 25, 2026.
- **Code:** `https://github.com/facebookresearch/tribev2`
- **Weights:** `https://huggingface.co/facebook/tribev2`
- **Demo:** `https://aidemos.atmeta.com/tribev2`
- **Glasser Parcellation:** Glasser, M.F., et al. (2016). *A multi-modal parcellation of human cerebral cortex.* Nature, 536(7615), 171-178.
- **Harvard-Oxford Atlas:** Frazier et al. (2005); Makris et al. (2006); Desikan et al. (2006).
- **NeuroSynth:** Kent, J.D., et al. (2026). *Neurosynth Compose.* Imaging Neuroscience.
