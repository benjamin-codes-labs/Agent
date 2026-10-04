# Task: label-free 4-class segmentation of SEM battery-anode cross-sections, with a fast deployable model

Repository: <REPO PATH or URL>
Data: `<PATH TO>/Hackathon-Polaron/Batch_{1,2,3}/img_<sampleid>_<detector>.tif` (93 TIFFs, 31 samples × 3 detectors)
Hardware: Windows laptop, 16 CPU cores, 34 GB RAM, NVIDIA RTX 5070 Ti Laptop (12 GB). RTX 50-series needs the CUDA 12.8 PyTorch build (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`). Use the GPU for the student model; everything else is CPU.

## Ground rules
- **Do not start any processing, training or segmentation until I approve a plan.** Your first deliverable is a written plan (Step 0). Read everything below first; most design decisions are already made, so the plan should be short and mostly confirm, flag disagreements, and give concrete parameters.
- **I will not hand-label anything.** All training labels must be generated automatically; say so explicitly in the plan.
- **Ask before spending money** (paid API calls). Include a cost estimate.
- **Never commit secrets** (API keys, `.env`).
- **Don't commit large binary outputs to git** (raw TIFFs, label maps, overlays, model weights). Use `.gitignore` and a GitHub release (preferred) or Git LFS; tell me which.
- **Treat the raw data as read-only.**
- Work stage by stage; after each stage print the key numbers (seed fractions, sanity-gate results, phase fractions) and stop if a sanity gate fails.

## Step 0: inspect, confirm, propose, wait
1. Verify: 3 folders, 93 TIFFs, names `img_<sampleid>_<detector>.tif`, 31 samples, detectors per sample.
2. Build `data/manifest.csv` (batch, sample_id, detector, path, width, height, nm_per_px, sha256). Flag groups of samples with identical image heights (same imaging session).
3. Write a short plan: classes, detectors, label generation, teacher and student models, validation, outputs, run time, API cost, risks. If you think something below is wrong, say so and propose the alternative. Then **stop and wait for approval**.

## Dataset facts (already established — do not re-derive, just verify quickly)
- Content: polished SEM cross-sections of a calendered graphite + SiOx blended Li-ion anode, all 31 samples.
- Detectors, co-located per sample: BSE (Z-contrast; SiOx brighter than carbon), Inlens (fine topography/edges; only detector where the carbon-binder domain is clearly visible), ETD in 27 samples or SE in 4 (`rxax5ozo`, `utfgcjfa`, `vc2whyaq`, `x77cy643`). Treat SE as ETD-equivalent after per-image normalisation.
- Format: 8-bit greyscale stored as 3-channel RGB, LZW TIFF, 14–23 MB each; 6996–7000 × 1904–2316 px; **25.000 nm/px** (XResolution ≈ 1 015 998 px/inch). Field of view ~175 × 48–58 µm. `Pillow` reads them directly (`Image.MAX_IMAGE_PIXELS = None`); `tifffile` needs `imagecodecs`.
- **The three RGB channels are identical except the last two columns (x = 6998–6999) in some files, where R ≠ G,B.** Read channel G (index 1) and exclude an 8-px border.
- **Co-registration is verified**: phase cross-correlation BSE↔Inlens and BSE↔ETD/SE on all samples gives 0.0 ± 0.1 px shift. No registration needed; multi-detector per-pixel features are safe.
- Orientation: graphite flakes are horizontal; the vertical axis is through-thickness.
- Known pitfalls:
  - **Pores are NOT resin-filled.** Open pores show sub-surface material with graphite-like grey in BSE, so BSE intensity thresholds undercount porosity. **Key observation from inspection (e.g. `kbdh4tri`): those grey-floored pore interiors are dark in Inlens and in ETD/SE** (poor SE collection from inside a cavity), and are ringed by a bright Inlens edge. So the pore cue is Inlens/ETD darkness, not BSE darkness.
  - Graphite and carbon-binder domain (CBD) have the same BSE grey; separable only by texture (CBD = lacy, bubbly, high-frequency texture at particle necks and thin films), mainly in Inlens.
  - Inlens has charging, edge brightening, banding, and flake-to-flake channelling contrast (`r17byphk`: graphite flakes differ in Inlens brightness with no BSE difference). Use Inlens texture (gradients, Hessian, local std), never raw Inlens intensity, for the graphite/CBD decision.
  - `epqdaau9`: Cu current collector = saturated-white curved band along the bottom edge in all detectors. Exclude it (BSE > 0.97 after normalisation, connected to the bottom edge, dilate ~20 px).
  - `5n1q8atc` and `i9jiqjwl`: unpolished strip at the top. Exclude it (detect via anomalous Inlens high-frequency energy in the top rows; confirm visually).
  - Imaging sessions cut across batches: `ffwubibz`/B1, `r17byphk`/B2, `cfe5vt7s`/B3 share height 2080 px; `hzumfsms`/`9luzk4jm` 2088; `kbdh4tri`/`71vgq3fw` 2060; `0grcilhi`/`hawkfj64`/`mgxahqnk` 1904; `4ih2ggld`/`5n1q8atc` 2316. Report batch comparisons with this in mind.

## What I want
1. **4-class semantic segmentation:** 0 = pore, 1 = graphite, 2 = SiOx, 3 = CBD, 255 = excluded (Cu foil, unpolished strips, borders). One uint8 label map per sample at **half resolution (50 nm/px, ~3500 × 1000)** unless you argue for full, plus a coloured overlay JPEG per sample.
2. **A fast deployable model (the "student")**: a small CNN that segments a new BSE+Inlens pair in **≤ 3 s on the RTX 5070 Ti** (and ≤ 20 s on CPU via ONNX Runtime). This is a hard requirement — a 30 s/image random-forest pipeline is not acceptable as the final product, only as the label generator.
3. **Instance segmentation of SiOx particles** (distance-transform watershed) → 2D section size distributions.
4. **Metrics CSVs:** per-sample phase area fractions (excluded pixels out of the denominator); per-batch mean ± SD with Kruskal–Wallis and pairwise Mann–Whitney (Holm-corrected); SiOx particle count, median and p90 equivalent diameter per sample.
5. **A validation report** against the baseline numbers below, with an accuracy estimate and its basis.
6. **Reproducible code**: `requirements.txt` (+ `requirements-gpu.txt`), one command per stage, short README.

## Required approach: teacher → student

### Layout
```
data/raw/Batch_{1,2,3}/img_<sid>_<det>.tif    originals, read-only, git-ignored
data/manifest.csv                              committed
data/processed/                                half-res normalised arrays, features, exclusion masks (git-ignored)
src/                                           one module per stage: manifest, preprocess, seeds, features, train_teacher,
                                               predict_teacher, train_student, predict, instances, metrics, validate, report
models/                                        teacher (LightGBM .txt), student (.pt, .onnx) — git-ignored, release asset
outputs/segmentation/<batch>/<sid>/            labels.png (uint8), overlay.jpg, siox_instances.png (uint16)
outputs/metrics/                               phase_fractions.csv, batch_stats.csv, particle_sizes.csv, siox_summary.csv, validation.csv
reports/                                       report.md, overlay contact sheets, hard-case panels
```

### Stage 1: preprocessing (CPU, ~10 min)
Read channel G → 2× block-mean downsample → per-image percentile normalisation (p0.5–p99.5 → [0,1], computed on non-excluded pixels) → light denoise (Gaussian σ=1 or 3×3 median). Build exclusion masks (Cu, unpolished strips, 8-px border; the R≠G columns fall inside the border). Save as `.npy` per sample/detector.

### Stage 2: automatic seed labels (confident pixels only, 5–15 % of each image per class where possible; no hand labels)
- SiOx: BSE above the upper multi-Otsu threshold, opening r=3 px, area ≥ 2 µm² (= 800 px at half res), eroded 2 px.
- Pore: Inlens < its p5 AND (ETD/SE < its p10 where available) AND BSE not SiOx-bright; eroded 2 px. **Do not require BSE darkness** — that is what makes the baseline `thr` a lower bound.
- Graphite: BSE mid-grey AND low Inlens texture (local std at σ=4 below its p40) AND distance-to-nearest-edge > 6 px (interiors of large smooth regions), eroded.
- CBD: BSE mid-grey AND high Inlens high-frequency energy (DoG σ1–σ3 above p80) AND outside graphite interiors AND ≥ 4 px from any SiOx or pore seed.
Print the seed fraction per class per image and save a seed overlay for 3 samples for me to look at.

### Stage 3: teacher = LightGBM pixel classifier (CPU)
- Features from BSE and Inlens (ETD/SE in the ablation): Gaussian, gradient magnitude, Laplacian, Hessian eigenvalues, local std at σ = 1, 2, 4, 8 px; plus large-scale context: BSE minus its 32-px local mean (depth/shadow cue), Inlens minus its 16-px local mean (charging/channelling compensation). ~60 features, float32, computed per tile to bound memory.
- Train on ~2 M class-balanced seed pixels across all 31 samples. LightGBM multiclass, ~500 trees, early stopping on a held-out seed subset.
- Self-training: 2 rounds — predict all pixels, add pixels with max-prob ≥ 0.9 that survive a 5-px majority filter, retrain.
- Post-process: remove objects < 0.1 µm², apply exclusion mask. Save per-pixel max-prob too (needed for the student).
- Ablation: BSE-only vs BSE+Inlens vs BSE+Inlens+ETD/SE. Judge by: SiOx error vs `thr`, pore ≥ `thr`, stability across seeds, and overlays. Keep the third detector only if it clearly helps.
- **Sanity gates (automatic; if any fails, stop and report instead of continuing):** SiOx within 1.5 pp of `thr` for every sample except `epqdaau9`; pore ≥ `thr` for every sample; the expected rankings below hold.

### Stage 4: student = small U-Net on the GPU (the deliverable model)
- Inputs: 2 channels (BSE, Inlens) at half res, per-image normalised (3 channels if the ablation keeps ETD/SE — but prefer 2, since 4 samples lack ETD).
- Targets: teacher labels with **ignore_index = 255 on pixels where teacher max-prob < 0.7** and on excluded pixels, so the student learns only from confident labels and interpolates the rest.
- Architecture: `segmentation_models_pytorch.Unet` with a light encoder (`resnet18` or `timm-efficientnet-b0`, ImageNet weights, first conv adapted to 2 channels), 4 output classes, ~3–12 M params. Loss: cross-entropy + Dice. AMP. AdamW, cosine schedule, ~40 epochs over random 512×512 crops (≈ 3000 crops/epoch).
- Augmentation: horizontal flips only (keep the through-thickness axis), per-channel brightness/contrast/gamma jitter, Gaussian noise, synthetic horizontal banding on Inlens, small elastic deformation. No vertical flips, no 90° rotations.
- Split: 25 training samples / 6 held out (2 per batch) for student-vs-teacher agreement on unseen images.
- Inference: 512-px tiles, 64-px overlap, soft-blended; export ONNX; time it on GPU and on CPU and report both. Target ≤ 3 s per image on the 5070 Ti.
- Acceptance: passes the same sanity gates as the teacher; ≥ 90 % pixel agreement with the teacher on confident pixels of the 6 held-out samples; phase fractions within 1 pp of the teacher's. If the student is better than the teacher on overlays (it usually smooths speckle), say so and use the student for the final label maps; state which model produced the delivered labels.

### Stage 5: SiOx instances
Binary SiOx → fill holes → Euclidean distance transform → h-maxima markers (h ≈ 3 px) → watershed. Per particle: area (µm²), equivalent diameter (µm), aspect ratio, centroid, sample_id. Report 2D section sizes only.

### Stage 6: metrics, validation, report
- `phase_fractions.csv`: per sample, 4 classes, excluded fraction, n_valid_px.
- `batch_stats.csv`: mean ± SD per batch per class; Kruskal–Wallis p; pairwise Mann–Whitney (Holm). Also repeat with the same-session height groups noted, so I can see whether "Batch_3 is more porous" survives.
- `validation.csv`: per sample, model vs `thr` and `vis` for pore and SiOx; Spearman correlations; rankings; stability SD from retraining the teacher on 3 random 20-sample subsets × 2 seeds (target < 1 pp); student-vs-teacher agreement/confusion on held-out samples; inference timings.
- `reports/report.md`: method, results tables, accuracy estimate and its basis, failure cases with crops, what was excluded. Overlay contact sheet of all 31 samples and zoomed panels for `kbdh4tri`, `epqdaau9`, `r17byphk`, `5n1q8atc`, `i9jiqjwl`.
- Optional, ask first: AI-annotator check — ~300 random points, each a 3-detector 256-px crop with a marker, classified by Claude via the API as an independent accuracy estimate (≈ $2–3 with Sonnet; ≈ $10 with Opus). Vision LLMs must not produce pixel masks; point annotation and visual QA only.

## Signs the method is working
1. SiOx within ~1.5 pp of `thr` per sample; expected rankings reproduced.
2. Pore ≥ `thr` in every sample and the overlays visibly capture grey-floored open pores (check `kbdh4tri`).
3. Phase fractions change < ~1 pp across retraining subsets/seeds.
4. Overlays right on the hard cases: `kbdh4tri` (thr 8.6 vs vis 25, many grey pore floors), `epqdaau9` (Cu excluded), `r17byphk` (Inlens channelling contrast must not flip graphite to another class).
5. CBD appears as lacy material at particle necks and thin films, not as whole graphite particles.
6. Student inference ≤ 3 s/image on the GPU and agrees with the teacher.

**If SiOx deviates strongly from `thr`, or pore falls below `thr`, treat it as a bug until explained.**

## Baseline results to validate against
Two earlier, independent measurements (2D area %):
- **thr:** 3-class multi-Otsu on median-filtered BSE. Pore = darkest class. SiOx = brightest class after a 6-px opening and removal of objects < 2 µm².
- **vis:** per-sample expert-style visual estimate by a vision LLM from all 3 detectors.

Reliability: **SiOx — trustworthy** (thr vs vis Spearman 0.85, MAD 0.6 pp); your SiOx should be within ~1.5 pp of thr except `epqdaau9` (thr includes Cu). **Pore — thr is a LOWER BOUND** (black deep voids only); vis is plausible but not a measurement (Spearman 0.38 between them); your pore must be ≥ thr in every sample, most likely between thr and vis. **Graphite ~68–77 %, CBD ~5–10 %** (visual only; CBD uncertain).

Expected rankings: `f1vzngrs` densest (thr 5.7, vis 6); `hzumfsms`, `0grcilhi` most porous; `5n1q8atc`, `4ih2ggld` highest SiOx; `avn74qx1`, `hawkfj64`, `r17byphk` lowest SiOx; uneven SiOx distribution in `mgxahqnk`, `ufdvpb81`, `avn74qx1`, `iv6g2oq0`.

Batch level (mean ± SD):

| batch | n | pore thr | pore vis | SiOx thr | SiOx vis |
|---|---|---|---|---|---|
| Batch_1 | 7 | 8.2 ± 1.6 | 12.6 ± 3.1 | 7.4 ± 3.4 | 6.7 ± 2.4 |
| Batch_2 | 7 | 9.3 ± 1.4 | 14.0 ± 2.0 | 5.4 ± 1.3 | 5.5 ± 1.7 |
| Batch_3 | 17 | 10.3 ± 2.1 | 18.6 ± 4.4 | 5.3 ± 0.8 | 5.3 ± 1.1 |

Kruskal–Wallis p: pore thr 0.14, pore vis 0.003, SiOx thr 0.29, SiOx vis 0.38. **Open question to settle: is Batch_3 really more porous?**

Per sample (area %):

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

## Expected run times (so you can tell me if yours differ)
Preprocess + seeds ~10 min; teacher features + training + 2 self-training rounds ~1 h on 16 cores; ablations + stability ~1.5 h; student training 30–60 min on the 5070 Ti; instances/metrics/report ~30 min. Teacher inference ~30–60 s/image; student ≤ 3 s/image on GPU.

## When you finish
Send me a short summary: what you did; per-batch phase fractions; comparison with the baseline; accuracy estimate and its basis; which model (teacher or student) produced the delivered labels and its measured inference time; known failure cases; where the outputs are.
