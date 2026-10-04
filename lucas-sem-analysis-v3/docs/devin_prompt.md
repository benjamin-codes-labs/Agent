# Task: label-free segmentation of SEM battery-electrode cross-sections

Repository: <REPO URL>
Data: <ZIP FILE NAME> in the repository root

## Ground rules
- **Do not start any processing, training or segmentation until I approve a plan.** Your first deliverable is a written plan (see "Step 0").
- **I will not hand-label anything.** Any method that needs training labels must generate them automatically, or use an AI annotator, and must say so.
- **Ask before spending money**, e.g. on paid API calls for an AI annotator. Include a cost estimate.
- **Never commit secrets** such as API keys or `.env` files.
- **Don't commit large binary outputs to git.** That covers raw TIFFs, label maps and overlays. Use `.gitignore` plus Git LFS, or a GitHub release, and tell me which you chose.
- **Treat the raw data as read-only.**

## Step 0: extract, inspect, then propose a plan (and wait)
1. Extract the zip. Verify the contents:
   - 3 folders (`Batch_1`, `Batch_2`, `Batch_3`);
   - 93 TIFFs = 31 samples × 3 detectors;
   - file names `img_<sampleid>_<detector>.tif`.
2. Propose a clean data structure and implement it, if I approve. For example:

       data/raw/<batch>/<sample_id>/{BSE,Inlens,ETD|SE}.tif    (or keep the originals and use a manifest)
       data/manifest.csv     batch, sample_id, detector, path, width, height, nm_per_px, sha256
       data/processed/       derived/intermediate files (git-ignored)
       src/                  code
       outputs/segmentation/<batch>/<sample_id>/  labels.png, overlay.jpg, siox_instances.png
       outputs/metrics/      phase_fractions.csv, particle_sizes.csv, validation.csv
       reports/              method + results write-up

3. Write the plan:
   - which classes;
   - which detectors;
   - how training labels are produced;
   - which model;
   - how accuracy is validated without hand labels;
   - outputs, run time and compute;
   - any API cost;
   - main risks.

   **If you think there is a better approach than the one I sketch below, propose it and explain why.** Then stop and wait for my approval.

## Dataset facts (already established)
- **Content:** polished SEM cross-sections of a **calendered graphite + Si-based (most likely SiOx) blended Li-ion anode**. This holds for all 31 samples.
- **Detectors, co-located per sample:**
  - BSE (atomic-number contrast; SiOx is brighter than carbon);
  - Inlens (fine topography/edges; the only detector where the carbon-binder domain is clearly visible);
  - ETD in 27 samples, or SE in 4 (`rxax5ozo`, `utfgcjfa`, `vc2whyaq`, `x77cy643`).
- **Image format:**
  - 8-bit greyscale stored as 3-channel RGB (all channels equal), LZW-compressed TIFF, ~20 MB each;
  - ~7000 × 1600–2300 px;
  - **25 nm/px**, from the TIFF resolution tags. Other SEM metadata was stripped.
  - Field of view ~175 × 40–58 µm.
  - `tifffile` needs `imagecodecs` for LZW; Pillow reads them directly. Set `Image.MAX_IMAGE_PIXELS = None`.
- **Co-registration:** the three detector images per sample have identical dimensions, but **pixel alignment has not been verified**. Check it, e.g. by cross-correlation, before using multi-detector features.
- **Orientation:** graphite flakes are strongly horizontal, so the vertical axis is the through-thickness direction.
- **Known pitfalls:**
  - **Pores are NOT resin-filled.** Open pores show sub-surface material with graphite-like grey in BSE, so intensity thresholds undercount porosity. This is the hardest problem in the dataset.
  - **Graphite and carbon-binder (CBD) have the same BSE grey.** They are separable only by texture, mainly in Inlens.
  - **Inlens has charging, edge brightening and banding.** Use its texture, not its raw intensity.
  - **`epqdaau9` shows the Cu current collector** (very bright) along its bottom edge. Exclude it.
  - **`5n1q8atc` and `i9jiqjwl` have an unpolished strip at the top.**
  - **Imaging sessions may cut across batches.** Several groups of images share identical heights (e.g. `ffwubibz`/B1, `r17byphk`/B2 and `cfe5vt7s`/B3 are all 52.0 µm tall). Keep this in mind for batch comparisons.

## What I want
- **A 4-class semantic segmentation:** 0 = pore, 1 = graphite, 2 = SiOx, 3 = carbon-binder domain, 255 = excluded (Cu foil, unpolished strips, borders).
  - Format: one uint8 label map per sample, at full or half resolution (state which).
  - Plus a coloured overlay JPEG per sample.
- **Instance segmentation of SiOx particles**, e.g. a distance-transform watershed, to give particle size distributions. Report them as 2D section sizes.
- **Metrics CSVs:**
  - per-sample phase area fractions (excluded pixels left out of the denominator);
  - per-batch mean ± SD, with Kruskal–Wallis and pairwise Mann–Whitney tests;
  - SiOx particle count, median and p90 equivalent diameter.
- **A validation report** comparing your results against the baseline numbers below, with an accuracy estimate.
- **Reproducible code** (`requirements.txt`, one command per stage) and a short README.

## Suggested approach (a starting point, not a requirement)
1. Preprocessing: per-image percentile normalisation, light denoising, alignment check, exclusion masks.
2. **Automatic training labels** (confident pixels only, roughly 5–15 % of each image):
   - pore = very dark in BSE and Inlens, eroded;
   - SiOx = compact bright BSE objects > 2 µm², eroded;
   - graphite = interiors of large, smooth, mid-grey regions (low Inlens texture);
   - CBD = mid-grey with dense lacy high-frequency Inlens texture, away from graphite interiors.
3. A random-forest (or XGBoost GPU) pixel classifier on multi-scale features (intensity, gradients, Hessian/texture, sigma ≈ 1–8 px) from **BSE + Inlens**. Ablate BSE-only vs BSE+Inlens vs BSE+Inlens+ETD/SE, and only keep the third detector if it clearly helps.
4. 1–2 self-training rounds (add high-confidence predictions as labels).
5. Optional: distil into a U-Net on GPU for smoother boundaries. Use SAM/micro-SAM for SiOx outlines if useful.
6. **Validation without hand labels:**
   - stability across training subsets and seeds;
   - agreement with the baseline below;
   - visual overlays;
   - optionally an AI annotator (e.g. Claude) classifying ~300 random points from zoomed multi-detector crops, as an independent accuracy estimate. Ask me before running it.
7. Pixel-level classification should come from the trained model. Vision LLMs cannot produce reliable pixel masks; use them only for checking or point annotation.

## Baseline results to check your method against
Two earlier, independent measurements exist:
- **thr:** a 3-class multi-Otsu threshold on median-filtered BSE. Pore = darkest class. SiOx = brightest class after a 6-px-radius opening and removal of objects < 2 µm².
- **vis:** a per-sample expert-style visual estimate by Claude Opus 5.5 from all 3 detectors.

Both are 2D area %.

**What we know about their reliability:**
- **SiOx area: trustworthy.** thr and vis agree (Spearman 0.85, mean absolute difference 0.6 pp). **Your SiOx fractions should land within ~1.5 pp of thr**, except `epqdaau9`, whose thr value includes the Cu foil.
- **Pore: thr is a LOWER BOUND** (it counts only black, deep voids). vis is plausible but not a measurement; the two correlate weakly (Spearman 0.38). **Your pore fraction should be ≥ thr in every sample.** Most samples are likely somewhere between thr and vis.
- **Graphite and CBD (visual only):** graphite ~68–77 %, CBD ~5–10 %. The CBD figure is uncertain.

**Expected rankings:**
- `f1vzngrs` = densest (thr 5.7 %, vis 6 %).
- `hzumfsms` and `0grcilhi` = most porous.
- `5n1q8atc` and `4ih2ggld` = highest SiOx.
- `avn74qx1`, `hawkfj64`, `r17byphk` = lowest SiOx.
- Uneven SiOx distribution in `mgxahqnk`, `ufdvpb81`, `avn74qx1`, `iv6g2oq0`.

**Batch level** (mean ± SD):

| batch | n | pore thr | pore vis | SiOx thr | SiOx vis |
|---|---|---|---|---|---|
| Batch_1 | 7 | 8.2 ± 1.6 | 12.6 ± 3.1 | 7.4 ± 3.4 | 6.7 ± 2.4 |
| Batch_2 | 7 | 9.3 ± 1.4 | 14.0 ± 2.0 | 5.4 ± 1.3 | 5.5 ± 1.7 |
| Batch_3 | 17 | 10.3 ± 2.1 | 18.6 ± 4.4 | 5.3 ± 0.8 | 5.3 ± 1.1 |

- Kruskal–Wallis p-values: pore thr 0.14, pore vis 0.003, SiOx thr 0.29, SiOx vis 0.38.
- **Open question for you to settle:** is Batch_3 really more porous? Only the visual estimate says so significantly.

**Per sample** (area %):

| batch | sample | pore thr | pore vis | SiOx thr | SiOx vis |
|---|---|---|---|---|---|
| 1 | 4ih2ggld | 9.7 | 15 | 10.5 | 9 |
| 1 | 5n1q8atc | 6.2 | 14 | 13.6 | 11 |
| 1 | f1vzngrs | 5.7 | 6 | 6.7 | 6 |
| 1 | ffwubibz | 8.2 | 12 | 4.1 | 4.5 |
| 1 | fzrt2k6r | 9.0 | 15 | 5.7 | 6.5 |
| 1 | iv6g2oq0 | 9.8 | 13 | 5.2 | 4.5 |
| 1 | uhdslk0o | 8.8 | 13 | 5.7 | 5.7 |
| 2 | 3806gxp0 | 10.5 | 15 | 5.2 | 5.5 |
| 2 | avn74qx1 | 9.8 | 11 | 3.9 | 4 |
| 2 | b3esycq1 | 9.3 | 14 | 7.6 | 9 |
| 2 | epqdaau9 | 7.3 | 12 | 5.9 (incl. Cu) | 6 |
| 2 | i9jiqjwl | 8.2 | 17 | 6.0 | 5 |
| 2 | r17byphk | 11.4 | 15 | 4.0 | 4 |
| 2 | rxax5ozo | 8.9 | 14 | 4.9 | 5 |
| 3 | 0grcilhi | 14.9 | 24 | 4.7 | 5 |
| 3 | 71vgq3fw | 8.5 | 18 | 5.8 | 6 |
| 3 | 9luzk4jm | 12.5 | 22 | 5.4 | 4.5 |
| 3 | cfe5vt7s | 9.7 | 20 | 5.9 | 6.5 |
| 3 | hawkfj64 | 9.7 | 18 | 3.9 | 4.5 |
| 3 | hzumfsms | 14.7 | 25 | 4.4 | 5 |
| 3 | kbdh4tri | 8.6 | 25 | 5.6 | 6 |
| 3 | mgxahqnk | 9.8 | 16 | 6.3 | 6.5 |
| 3 | pl8uabbv | 9.7 | 15 | 4.9 | 5.5 |
| 3 | ptg8lmto | 8.4 | 13 | 4.6 | 3.7 |
| 3 | tuy3zymq | 9.0 | 22 | 5.2 | 5 |
| 3 | ufdvpb81 | 12.4 | 17 | 4.8 | 3 |
| 3 | utfgcjfa | 9.8 | 13 | 5.1 | 5 |
| 3 | vc2whyaq | 9.6 | 15 | 5.3 | 5 |
| 3 | x77cy643 | 9.5 | 14 | 5.9 | 7 |
| 3 | x7u69zsw | 8.3 | 15 | 7.4 | 7 |
| 3 | xgj4xftb | 9.4 | 25 | 5.1 | 4.4 |

**Signs your method is working:**
1. SiOx is within ~1.5 pp of thr for each sample, and the expected rankings above are reproduced.
2. Pore ≥ thr in every sample, and it visibly captures grey-floored open pores in the overlays.
3. Phase fractions change by < ~1 pp when you retrain on a different subset of samples or with a different seed.
4. Overlays look right on the hard cases: `kbdh4tri` (thr 8.6 vs vis 25, many grey pore floors), `epqdaau9` (Cu foil excluded), `r17byphk` (Inlens contrast varies between graphite regions with no BSE difference; don't let that flip graphite to another class).
5. CBD appears as lacy material at particle necks and thin films, not as whole graphite particles.

**If SiOx deviates strongly from thr, or pore falls below thr, treat it as a bug until explained.**

## Hardware (if code should also run locally)
Windows laptop: 16 CPU cores, 34 GB RAM, NVIDIA RTX 5070 Ti Laptop (12 GB). The RTX 50-series needs recent CUDA builds of any GPU library.

## When you finish
Send me a short summary covering:
- what you did;
- per-batch phase fractions;
- how they compare with the baseline;
- your accuracy estimate and its basis;
- known failure cases;
- where the outputs are.
