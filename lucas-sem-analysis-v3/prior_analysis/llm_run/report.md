# SEM Cross-Section Dataset: Analysis Report

*31 samples in 3 batches, analysed 2026-10-03.*

**How this was produced**
- **Per-sample analyses:** Claude Opus 5.5 at max effort, one API call per sample. Each call sent 3 detector overviews plus 2 native-resolution crops, with the threshold measurements as context. Results are in `samples/*.json`.
- **Threshold measurements:** a local Python script (`analyze_sem.py`) running a 3-class multi-Otsu segmentation of each BSE image.
- **This report:** written by Claude Code from those 31 analyses and the measurements, without a further API call.

All percentages are **2D area fractions of a single ~175 × 50 µm field per sample**.

---

## 1. Key findings

1. **One material system throughout.** All 31 samples are cross-sections of a calendered **graphite + Si-based (most likely SiOx) blended Li-ion anode**. Graphite was identified with high confidence in every sample. The Si phase was medium confidence: EDS is needed to confirm SiOx vs Si vs Si/C.
2. **The Si-phase content is measured reliably.**
   - Typical value: **~5–6 % of the section area**.
   - Range: 3–14 %.
   - The threshold measurement and Claude's visual estimate agree closely (Spearman ρ = 0.85, mean absolute difference 0.6 percentage points).
3. **No significant batch difference in Si content.** Batch_1's higher mean comes from two samples, `4ih2ggld` and `5n1q8atc`.
4. **Porosity is the least certain quantity.** The sections are not resin-filled, so open pores show grey sub-surface material.
   - Threshold measurement: 6–15 %. This is a lower bound.
   - Claude's visual estimate: 6–25 %.
   - The two correlate only weakly (ρ = 0.38).
5. **Batch_3 looks more porous, but this is not proven.** It is the highest batch on both measures. The difference is only statistically significant in Claude's visual estimate (p = 0.003), not in the threshold measurement (p = 0.14). See §4.
6. **No major defects in any sample.** Recurring features are:
   - graphite basal-plane cracking;
   - slit gaps between flakes;
   - partial debonding around Si particles;
   - a sub-population of mottled/porous Si particles, some with low-Z rims.

   Whether any of this reflects cycling cannot be told from the images alone.

---

## 2. Dataset and method

| Item | Value |
|---|---|
| Samples | 31 (Batch_1: 7, Batch_2: 7, Batch_3: 17) |
| Detectors per sample | BSE + Inlens + ETD (27 samples) or BSE + Inlens + SE (4: `rxax5ozo`, `utfgcjfa`, `vc2whyaq`, `x77cy643`) |
| Pixel size | 25 nm (from TIFF resolution tags; original SEM metadata was stripped) |
| Field of view | ~175 µm wide × 40–58 µm tall |
| Images | 8-bit greyscale stored as RGB, LZW-compressed TIFF, ~20 MB each |
| Orientation | Graphite flakes are strongly aligned horizontally in every sample. The vertical axis is almost certainly the through-thickness (calendering) direction. |
| Position in the electrode | Most fields lie inside the coating. `epqdaau9` shows the Cu current collector at the bottom edge; `cfe5vt7s` appears to include the free surface at the top. |

**Threshold measurements (BSE only):**
- median filter, then a 3-class multi-Otsu threshold;
- the darkest class is counted as **pore**;
- the brightest class, after a morphological opening (removes features < 0.3 µm wide) and removal of objects < 2 µm², is counted as **SiOx particles**.

Graphite and the carbon-binder domain (CBD) share the middle class and cannot be separated by this method.

---

## 3. Material system and phases

| Phase | Appearance | Typical amount | Confidence |
|---|---|---|---|
| **Graphite** (flake/platelet, some spheroidised) | Darkest solid in BSE. Lamellar texture, basal-plane cracks and frayed ends in Inlens/ETD. 10–35 µm long, 4–12 µm thick. | ~68–77 % of section | High (31/31) |
| **Si-based additive** (most likely SiOx, possibly Si or Si/C) | Brighter in BSE (higher mean Z). Angular, fracture-faceted, 1–10 µm. Sits in the interstices between flakes. | ~5–6 % (3–14 %) | Medium |
| **Carbon-binder domain** | Lacy, nanoporous web at particle necks and thin films on surfaces. Same BSE grey as graphite; resolved only in Inlens. | ~5–10 % (visual, uncertain) | Medium (22) / low (8) |
| **Pores** | Black in all detectors where deep. Mostly slit-shaped and parallel to the flakes, plus voids at triple junctions. | 6–25 % (see §6) | High for presence, low for amount |

**Si sub-populations.** Most samples have dense, uniform bright particles. About 12 samples also show a **mottled/porous/granular** variant:
- `ffwubibz`, `iv6g2oq0`, `i9jiqjwl`, `r17byphk`, `rxax5ozo`, `9luzk4jm`, `ptg8lmto`, `ufdvpb81`, `utfgcjfa`, `xgj4xftb`, `mgxahqnk`, `x7u69zsw`.

About 6 samples show **0.3–1 µm low-Z rims/shells** around Si particles:
- `fzrt2k6r`, `iv6g2oq0`, `71vgq3fw`, `utfgcjfa`, `vc2whyaq`, `x77cy643`.

These could be a carbon coating, a second additive (Si/C composite), or degradation/SEI from cycling. The images alone cannot decide this.

---

## 4. Batch comparison

Values are mean ± SD (min–max). "thr" means the threshold measurement and "Claude" means the visual estimate. Area fractions are in %.

| Batch | n | Pore (thr) | Pore (Claude) | Si phase (thr) | Si phase (Claude) | Si size, µm (Claude) |
|---|---|---|---|---|---|---|
| Batch_1 | 7 | 8.2 ± 1.6 (5.7–9.8) | 12.6 ± 3.1 (6–15) | 7.4 ± 3.4 (4.1–13.6) | 6.7 ± 2.4 (4.5–11) | 3.9 (3–5) |
| Batch_2 | 7 | 9.3 ± 1.4 (7.3–11.4) | 14.0 ± 2.0 (11–17) | 5.4 ± 1.3 (3.9–7.6) | 5.5 ± 1.7 (4–9) | 3.5 (3–4) |
| Batch_3 | 17 | 10.3 ± 2.1 (8.3–14.9) | 18.6 ± 4.4 (13–25) | 5.3 ± 0.8 (3.9–7.4) | 5.3 ± 1.1 (3–7) | 3.4 (2.5–4.5) |

**Statistical tests** (Kruskal–Wallis across batches; pairwise Mann–Whitney):

| Measure | All batches | 1 vs 2 | 1 vs 3 | 2 vs 3 |
|---|---|---|---|---|
| Pore (thr) | p = 0.14 | 0.25 | 0.065 | 0.45 |
| Pore (Claude) | **p = 0.003** | 0.47 | **0.004** | **0.016** |
| Si phase (thr) | p = 0.29 | 0.28 | 0.14 | 0.92 |
| Si phase (Claude) | p = 0.38 | 0.25 | 0.22 | 0.97 |

**Interpretation**
- **Si content:** there is no meaningful batch difference. Batch_1 is more variable because `5n1q8atc` (13.6 %) and `4ih2ggld` (10.5 %) are well above the rest. Those two images also have identical dimensions, so they may come from the same electrode region.
- **Porosity:** Batch_1 < Batch_2 < Batch_3 on both measures, so the ordering is consistent. The size of the effect, however, depends on the method:
  - the gap between Claude's estimate and the threshold value averages 4.4 points in Batch_1 and 4.7 in Batch_2, but **8.4 points in Batch_3**;
  - so either Batch_3 has more shallow, grey-floored open pores (which the threshold misses), or the visual estimates drifted upwards for those samples.
  - **Conclusion:** treat "Batch_3 is more porous" as a hypothesis to test with a better segmentation (§8), not as a result.
- **Heterogeneity:** Claude scored every sample 2 or 3 out of 5. That score doesn't separate the samples and shouldn't be used to compare them.

**Possible confound: imaging sessions cut across batches.** Several groups of images share an identical height and detector set, which suggests the same imaging session or crop:

| Field height | Samples |
|---|---|
| 52.0 µm | `ffwubibz` (B1), `r17byphk` (B2), `cfe5vt7s` (B3) |
| 51.7 µm, SE detector | `rxax5ozo` (B2), `utfgcjfa`, `vc2whyaq`, `x77cy643` (B3) |
| 53.7 µm | `f1vzngrs` (B1), `epqdaau9` (B2) |
| 53.9 µm | `fzrt2k6r` (B1), `b3esycq1` (B2) |
| 56.8 µm | `i9jiqjwl` (B2), `pl8uabbv` (B3) |

If "batch" means a manufacturing batch, this is fine. Confirm with the organisers what a batch represents before drawing conclusions.

---

## 5. Sample ranking and outliers

**Most porous** (both measures high):
- `hzumfsms` (thr 14.7 %, Claude 25 %): a horizontal void band up to 35 × 15 µm.
- `0grcilhi` (14.9 / 24)
- `9luzk4jm` (12.5 / 22)
- `ufdvpb81` (12.4 / 17)

**Large disagreement** (threshold normal, Claude high):
- `kbdh4tri` (8.6 / 25), `xgj4xftb` (9.4 / 25), `tuy3zymq` (9.0 / 22), `cfe5vt7s` (9.7 / 20).
- In `kbdh4tri` the threshold mask catches the slit pores, but many grey, textured pore floors are not counted. The real value probably lies between the two numbers.

**Densest:**
- `f1vzngrs` (5.7 / 6): both methods agree; it is the least porous sample in the dataset.
- `5n1q8atc` (thr 6.2 %, Claude 14 %).

**Highest Si phase:**
- `5n1q8atc` (13.6 / 11), `4ih2ggld` (10.5 / 9), `b3esycq1` (7.6 / 9), `x7u69zsw` (7.4 / 7).

**Lowest Si phase:**
- `avn74qx1` (3.9 / 4), `hawkfj64` (3.9 / 4.5), `r17byphk` (4.0 / 4), `ufdvpb81` (4.8 / 3).

**Uneven Si distribution** (possible incomplete blending at the 50–100 µm scale):
- `mgxahqnk`: concentrated in the right third; left 60 % sparse.
- `ufdvpb81`: ~7 % in the right half vs < 1 % in the lower-left quadrant.
- `avn74qx1` and `iv6g2oq0`: right-hand enrichment.
- `xgj4xftb`: upper half.
- `4ih2ggld`: upper-centre cluster.

**Individually notable:**
- `epqdaau9`: Cu current collector visible along the bottom edge. It is the only sample with an absolute through-thickness reference. The bright foil also inflates its threshold "bright phase" value, so mask it before reusing that number.
- `r17byphk`: Inlens shows some graphite regions black and others grey with no BSE difference. This could be surface potential, electronic connectivity or lithiation state, and is worth investigating if the samples were cycled.
- `uhdslk0o`: an oversized ~35 µm spheroidised graphite particle with internal voids up to ~5 µm.
- `pl8uabbv`: an anomalous top-left corner with nodular agglomerates, large cavities and looser packing.
- `ufdvpb81`: the most moderate-severity findings (4), including a possible delamination gap along the bottom edge.

---

## 6. Defects and imaging artefacts

**Defects** (no sample has a *major* one):

| Feature | Prevalence | Notes |
|---|---|---|
| Graphite basal-plane cracking / delamination, frayed flake ends | Nearly all samples; moderate in ~10 | Typical of natural flake graphite and calendering; cycling cannot be ruled out |
| Inter-flake slit gaps / debonding, sometimes linking into horizontal bands | Most samples; banded in `vc2whyaq`, `i9jiqjwl`, `r17byphk`, `x77cy643`, `utfgcjfa` | May come from calendering spring-back, drying, or preparation; matters for in-plane vs through-plane transport |
| Gaps / debonding around Si particles | ~10 samples (e.g. `iv6g2oq0`, `uhdslk0o`, `ptg8lmto`, `tuy3zymq`) | Consistent with rigid particles in a compliant matrix, or with volume change during cycling |
| Cracked / fragmented Si particles | `fzrt2k6r`, `b3esycq1`, `rxax5ozo`, `ufdvpb81` and a few others | Moderate in 4 samples |
| Large voids (5–35 µm) | `hzumfsms`, `0grcilhi`, `9luzk4jm`, `4ih2ggld`, `3806gxp0`, `uhdslk0o`, `pl8uabbv` | Packing defects between misaligned flakes |
| CBD agglomerates | A few samples, minor | — |

**Artefacts:**

| Artefact | Prevalence | Effect |
|---|---|---|
| **Open, non-resin-filled pores** showing sub-surface material | Essentially every sample; rated significant in 30/31 | **The main limitation:** pore floors read as solid in BSE |
| Curtaining / ion-milling striations | Common, mostly in ETD/SE | Minor; BSE is largely unaffected |
| BSE shot noise | Common | Minor; inflates the raw bright class, handled by filtering |
| Inlens edge brightening, charging, banding | Common; strong in `avn74qx1`, `cfe5vt7s`, `r17byphk` | **Don't threshold Inlens intensity directly** |
| Tile stitching seams, an unpolished strip at the top (`5n1q8atc`, `i9jiqjwl`) | Occasional | Minor |

---

## 7. Reliability of the measurements

| Quantity | Verdict |
|---|---|
| **Si-phase area fraction** (threshold, filtered) | **Trustworthy** as a 2D area fraction, within ~1–2 points. Check `epqdaau9`, where the Cu foil is included. |
| Raw bright fraction | **Not usable.** Inflated 1.5–2× by rims, edges and noise. |
| Si particle count and median size | Count roughly right after the 2 µm² filter. Median size biased low by fragments and sectioning: 2D section diameters underestimate 3D particle size. |
| **Pore fraction** (threshold) | **Lower bound only.** It counts deep black voids; open pores with grey floors and CBD nanoporosity are missed. |
| Pore fraction (Claude visual) | Plausible but **not a measurement**. Weakly correlated with the threshold value. Calendered graphite anodes are commonly ~25–40 % porous by volume, so even 15–25 % may be low; 2D area fraction ≠ volume porosity, and sub-resolution pores are invisible here. |
| Graphite / CBD split | **Not available** from BSE thresholding; graphite and CBD have the same BSE grey level. |
| Claude's qualitative observations (defects, rims, clustering, locations) | Specific and generally consistent across samples. Locations are given in µm and can be spot-checked. They should be treated as *expert-style annotations*, not ground truth. |

Each sample is a single field of ~175 × 50 µm, so field-to-field variation within an electrode is unknown. It could be as large as the batch differences.

---

## 8. Recommended next steps

1. **Better segmentation without manual labels.** Train a random-forest pixel classifier using all 3 detectors, with automatically generated training labels (`segment_rf.py`). This gives 4 phases (pore, graphite, SiOx, CBD), recovers some grey-floored pores, and lets the batch porosity difference be re-tested properly.
2. **Ask the organisers:**
   - what a "batch" represents (formulation, calendering pressure, cycling history);
   - whether the electrodes were cycled;
   - whether bulk porosity is known (from coating mass, thickness and density).

   The answers decide how to read the cracks, rims and mottled Si particles. A known bulk porosity would also calibrate the image-based numbers.
3. **Confirm the Si phase:** EDS on a few dense and mottled bright particles, to distinguish SiOx, Si and Si/C.
4. **Particle size distributions:** use watershed splitting of the SiOx mask, and report them as 2D section sizes or apply a stereological correction.
5. **Transport metrics:** once a reliable segmentation exists, compute in-plane vs through-plane tortuosity (e.g. TauFactor). The strong horizontal flake alignment and the slit-pore bands suggest these differ substantially.
6. **Representativeness:** if more fields per sample exist, analyse them, so field-to-field variation can be estimated before comparing batches.
7. **Housekeeping:** mask the Cu foil in `epqdaau9`, and avoid Inlens intensity for any thresholding.

---

## Appendix: per-sample key numbers

Pore and Si values are area %; sizes are in µm. Het. = Claude's heterogeneity score (1–5). Moderate defects = count rated moderate (none were major). Full details for each sample are in `samples/<batch>_<id>.json`.

| Batch | Sample | Pore thr | Pore Claude | Si thr | Si Claude | Si size | Het. | Moderate defects |
|---|---|---|---|---|---|---|---|---|
| 1 | 4ih2ggld | 9.7 | 15 | 10.5 | 9 | 5 | 3 | 1 |
| 1 | 5n1q8atc | 6.2 | 14 | 13.6 | 11 | 5 | 3 | 0 |
| 1 | f1vzngrs | 5.7 | 6 | 6.7 | 6 | 3 | 3 | 2 |
| 1 | ffwubibz | 8.2 | 12 | 4.1 | 4.5 | 3 | 2 | 0 |
| 1 | fzrt2k6r | 9.0 | 15 | 5.7 | 6.5 | 3.5 | 2 | 1 |
| 1 | iv6g2oq0 | 9.8 | 13 | 5.2 | 4.5 | 3.5 | 3 | 1 |
| 1 | uhdslk0o | 8.8 | 13 | 5.7 | 5.7 | 4 | 3 | 3 |
| 2 | 3806gxp0 | 10.5 | 15 | 5.2 | 5.5 | 3.7 | 3 | 1 |
| 2 | avn74qx1 | 9.8 | 11 | 3.9 | 4 | 3.5 | 3 | 0 |
| 2 | b3esycq1 | 9.3 | 14 | 7.6 | 9 | 4 | 2 | 2 |
| 2 | epqdaau9 | 7.3 | 12 | 5.9* | 6 | 3.5 | 3 | 1 |
| 2 | i9jiqjwl | 8.2 | 17 | 6.0 | 5 | 3.5 | 2 | 1 |
| 2 | r17byphk | 11.4 | 15 | 4.0 | 4 | 3 | 3 | 1 |
| 2 | rxax5ozo | 8.9 | 14 | 4.9 | 5 | 3.5 | 3 | 2 |
| 3 | 0grcilhi | 14.9 | 24 | 4.7 | 5 | 3 | 3 | 2 |
| 3 | 71vgq3fw | 8.5 | 18 | 5.8 | 6 | 3.5 | 2 | 1 |
| 3 | 9luzk4jm | 12.5 | 22 | 5.4 | 4.5 | 3.5 | 3 | 2 |
| 3 | cfe5vt7s | 9.7 | 20 | 5.9 | 6.5 | 4 | 3 | 1 |
| 3 | hawkfj64 | 9.7 | 18 | 3.9 | 4.5 | 3 | 3 | 1 |
| 3 | hzumfsms | 14.7 | 25 | 4.4 | 5 | 3 | 3 | 3 |
| 3 | kbdh4tri | 8.6 | 25 | 5.6 | 6 | 4.5 | 2 | 1 |
| 3 | mgxahqnk | 9.8 | 16 | 6.3 | 6.5 | 3 | 3 | 2 |
| 3 | pl8uabbv | 9.7 | 15 | 4.9 | 5.5 | 2.8 | 3 | 3 |
| 3 | ptg8lmto | 8.4 | 13 | 4.6 | 3.7 | 3.5 | 2 | 2 |
| 3 | tuy3zymq | 9.0 | 22 | 5.2 | 5 | 2.5 | 2 | 2 |
| 3 | ufdvpb81 | 12.4 | 17 | 4.8 | 3 | 3.5 | 3 | 4 |
| 3 | utfgcjfa | 9.8 | 13 | 5.1 | 5 | 4.5 | 3 | 2 |
| 3 | vc2whyaq | 9.6 | 15 | 5.3 | 5 | 3.5 | 3 | 1 |
| 3 | x77cy643 | 9.5 | 14 | 5.9 | 7 | 3 | 3 | 1 |
| 3 | x7u69zsw | 8.3 | 15 | 7.4 | 7 | 3 | 2 | 1 |
| 3 | xgj4xftb | 9.4 | 25 | 5.1 | 4.4 | 4 | 3 | 1 |

\* Includes part of the Cu current collector.

**API cost:** $33.45 for the 31 per-sample analyses (Claude Opus 5.5, max effort). There were also unmeasured charges from interrupted requests: 8 dropped per-sample streams and 3 dataset-report attempts.
