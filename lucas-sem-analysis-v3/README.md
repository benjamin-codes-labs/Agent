# SEM anode segmentation (Polaron hackathon)

> **Version 3** = v2 + the teammate's method (training-image guard, known-location matcher, GET4 range rule, calibration, forced mode); see `reports/report.md` §0. v1 is in [`../lucas-sem-analysis/`](../lucas-sem-analysis/), v2 in [`../lucas-sem-analysis-v2/`](../lucas-sem-analysis-v2/). Demo: `wechat-send-v3.zip` in the release.

> **Continuing this work (human or LLM)? Start with [HANDOFF.md](HANDOFF.md).**

Label-free 4-phase segmentation of 31 SEM cross-sections of a calendered **graphite + SiOx** Li-ion anode
(3 batches; BSE + Inlens + ETD/SE detectors, 25 nm/px), plus a fast U-Net that segments a new image pair in
under a second on a laptop GPU. On top of that sits a batch-identification stage (v2): an imaging-fingerprint model
and a material model, combined by an agreement rule that can answer "Batch 1 or Batch 2" or "unsure". It writes a
text-only facts file for an LLM narrator.

| value | class | colour in overlays |
|---|---|---|
| 0 | pore (including open, grey-floored pores) | blue |
| 1 | graphite | purple |
| 2 | SiOx | orange |
| 3 | carbon-binder domain (CBD, "binder") | green |
| 255 | excluded (8-px border, Cu current collector, unpolished strips) | red |

Results, validation and caveats are in **[reports/report.md](reports/report.md)**. Headline numbers (v2):

| Measure | Result |
|---|---|
| Dataset composition (mean of 31 samples) | pore 17.3 % (deep 9.5 + open 7.7), graphite 61.0 %, SiOx 6.3 %, CBD 15.4 % |
| Independent accuracy (240 random points, AI-annotated, blind) | 81.7 % for 4 classes (95 % CI 76.3-86.1 %), **87.1 % for 3 classes** (binder merged into carbon); 100 % on high-confidence points |
| Per-class precision | SiOx 0.96, graphite 0.83, pore 0.69, CBD 0.52 (graphite-vs-binder split is experimental) |
| Retraining stability | < 0.8 pp |
| Inference per image (segmentation) | 0.31 s GPU (RTX 5070 Ti), 1.5 s CPU (ONNX) |
| Batch models (leave-one-location-out, balanced accuracy) | fingerprint 0.70 (p = 0.002; 0.66 with sessions held out); material (segmentation features + DINOv2) 0.61 (p = 0.005) |
| Batch decision (agreement rule, may abstain) | answers 84 % of locations, 85 % right when it answers (sessions held out: 74 %, 91 %) |
| v3 additions (teammate's method) | training-image guard + known-location matcher (0 false matches); calibrated High tier (10/12 right); **forced mode 0.80 balanced accuracy, p = 0.001** (0.70 with sessions held out) |
| Caveats | Batch 1 and Batch 2 are statistically indistinguishable in this data; acquisition properties alone predict batch at 0.68 |

## Repository layout

```
data/
  raw/Batch_{1,2,3}/img_<sid>_<det>.tif   original TIFFs (git-ignored, read-only)
  manifest.csv                            one row per TIFF: size, pixel size, sha256, channel check
  qc.csv                                  per sample: detector registration shifts, image-height (session) group
  exclusions.json                         visually confirmed regions to exclude
  baseline/baseline.csv                   earlier measurements used for validation (thr, vis)
  processed/                              normalised half-res arrays, features, labels (git-ignored)
  nnUNet_raw/Dataset501_SEMAnode/         the segmented dataset in nnU-Net v2 format (git-ignored, release asset)
src/                                      one module per stage (see below)
models/                                   teacher.txt, student.pt / .onnx, bid_model.pkl, fingerprint_model.pkl (git-ignored, release asset)
outputs/
  segmentation/<batch>/<sid>/             labels.png, overlay.jpg, siox_instances.png, bse.png, inlens.png (release asset)
  batchid/<sid>.json, <sid>_evidence.jpg  per-location text-only facts for the narrator layer + evidence maps
  metrics/                                phase fractions, batch stats, particles, validation, KPIs, look ledger, ...
reports/                                  report.md, report_method.md and figures
prior_analysis/                           the earlier vision-LLM analysis that produced the baseline numbers
docs/                                     plan.md (task specification), devin_prompt.md
```

## Method in one paragraph

No pixel was labelled by hand.
1. **Rule-based seeds** label the unambiguous pixels: bright compact SiOx, deep black voids, flat graphite interiors.
2. **Claude Opus 5.5** labels the hard regions by classifying ~1,200 superpixels from BSE | Inlens | ETD crops. These
   are open pores whose floor shows grey sub-surface material (the sections are not resin-filled), and binder vs
   particle rims.
3. **A LightGBM teacher** trained on these labels labels every pixel. It uses multi-scale BSE + Inlens features and
   two self-training rounds.
4. **A U-Net student** (ResNet-18 encoder, 2-channel input) is distilled from the teacher's confident pixels and
   produces the delivered label maps.

A physical constraint keeps SiOx to BSE-bright pixels. Accuracy is checked against the earlier threshold/visual
baselines, retraining stability, held-out samples, and an independent Claude classification of uniformly random
points.

**Batch identification** (report §6) runs two models:
- **Fingerprint model** (`src/fingerprint.py`, `src/fingerprint_model.py`): one shrinkage-LDA per detector view on
  noise, banding, stripe, grey-level and compressibility measurements.
- **Material model** (`src/dino.py`, `src/bid_*.py`): 5 segmentation features + a DINOv2 texton head, with exact
  evidence maps.

`src/decision.py` combines them by agreement and gives a confidence tier and track record. `src/map_text.py`
describes the maps in words for a text-only LLM.

## Setup

```bash
pip install -r requirements.txt
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps -r requirements-gpu.txt
```

Put the raw data in `data/raw/Batch_{1,2,3}/`. Inside the hackathon repo, the shared `../data/raw/batch_{1,2,3}/` is used automatically; set `SEM_RAW_DIR` to point elsewhere. Without a GPU, `python run_cpu_pipeline.py` rebuilds the label maps and overlays from the LightGBM teacher and leaves the committed U-Net metrics untouched. The AI-annotation stages need an Anthropic API key in `.env`
(`ANTHROPIC_API_KEY=...`, git-ignored).

## Running

```bash
python run_pipeline.py                      # all stages except the paid AI annotation
python run_pipeline.py --from student       # resume from a stage
```

Stages, in order:
1. **Segmentation:** `manifest` → `preprocess` → `seeds` → `features` → `superpixels` → `ai_apply` → `teacher` →
   `ablation` → `stability` → `student` → `student_predict` → `export_onnx` → `finalize` → `instances` →
   `metrics` → `validate` → `figures` → `nnunet`
2. **Batch identification:** `dino` → `bid_features` → `bid_cache` → `bid_score` → `bid_null` → `bid_fit` →
   `fingerprint` → `fingerprint_model` → `decision` → `texture_features` → `known_spot_index` → `known_spot_eval` →
   `v3_prepare` → `v3_evaluate` → `v3_fit` → `bid_explain`
3. **Exploratory analyses:** `bid_compare`, `kpi`, `kpi_topo`, `b12`, `roughness_check`, `same_batch`,
   `fingerprint_b12`, `head_ablation`, `cascade_test`, `final_model`, `compare_versions`

AI annotation (paid, run once; results are committed in `outputs/metrics/`):

```bash
python -m src.ai_labels prepare && python -m src.ai_labels submit && python -m src.ai_labels collect --wait
python -m src.ai_validate prepare-uniform && python -m src.ai_validate submit uniform && python -m src.ai_validate collect uniform --wait
python -m src.ai_labels prepare-active && python -m src.ai_labels submit --round 2 && python -m src.ai_labels collect --round 2 --wait   # v2 active learning
```

## Segment a new image pair

```bash
python -m src.predict --bse img_X_BSE.tif --inlens img_X_Inlens.tif --out outputs/predict/X          # GPU
python -m src.predict --bse img_X_BSE.tif --inlens img_X_Inlens.tif --out outputs/predict/X --cpu    # ONNX, CPU
```

Whole chain for a new image (segmentation → features → DINO → fingerprint → decision → evidence map + facts JSON),
about 21 s on a laptop GPU:

```bash
python -m src.bid_explain predict --bse img_X_BSE.tif --inlens img_X_Inlens.tif --etd img_X_ETD.tif --out outputs/batchid/new/X
```

- `--etd` is optional: an ETD or SE image of the same field, used by the fingerprint model.
- `--holdout <sid>` refits all batch models, calibration and thresholds without a known location (honest demo on a
  training image).
- `--forced` always answers (the teammate's forced mode).
- **v3:** an image identical to a training image is refused, and a known spot (any detector, or a crop) is
  recognised. See `wechat-send-v3/README.md`.
- `outputs/batchid/new/X/facts.json` starts with `summary_sentences` and `decision`. See `wechat-send-v2/README.md`
  for a worked example.

## Train an nnU-Net on the segmented dataset

`python -m src.export_nnunet` writes `data/nnUNet_raw/Dataset501_SEMAnode`:
- BSE and Inlens are the two input channels;
- label 0 = pore, which is nnU-Net's "background";
- excluded pixels get the ignore label 4.

Then:

```bash
pip install nnunetv2
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 2d 0
```

## Large files

Raw data, label maps, the nnU-Net dataset and model weights are git-ignored. They go to the GitHub release
instead (`python -m src.package_release --tag v3.0`, which writes `release/*.zip`).
