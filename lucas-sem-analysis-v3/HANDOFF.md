# HANDOFF: read this first (for the next agent / LLM)

This file is the context you need to continue the work without redoing it. Detailed results are in
`reports/report.md` (v3; §0 is the v3 update); how to run things is in `README.md`.

## 1. Project in three lines
- **Data:** 31 SEM cross-sections of a graphite + SiOx Li-ion anode, 3 batches (7 / 7 / 17). Each location has
  BSE + Inlens + ETD-or-SE images, 25 nm/px.
- **Done (v2):**
  - a label-free 4-phase segmentation (pore, graphite, SiOx, binder; U-Net, 0.31 s per image on GPU);
  - two batch models (imaging fingerprint + material) combined by an agreement rule that may answer
    "Batch 1 or Batch 2" or "unsure";
  - a text-only `facts.json` per prediction for an LLM narrator.
- **Next (planned by the team):** the narrator / multi-agent layer that reads `facts.json`.

## 2. State of each stage

| Stage | Status | Key output | Key number |
|---|---|---|---|
| Inventory / QC | done | `data/manifest.csv`, `data/qc.csv` | 93 TIFFs; detectors co-registered to < 0.4 px |
| Preprocessing + exclusions | done | `data/processed/<sid>/*.npy`, `data/exclusions.json` | 1.1 % excluded |
| Training labels | done (**paid, cached**) | `outputs/metrics/ai_superpixel_labels.csv` (round 1), `ai_superpixel_labels_r2.csv` (round 2) | 1,178 + 992 superpixels labelled by Claude Opus 5.5 |
| Teacher (LightGBM) | done | `models/teacher.txt` | all per-sample gates pass |
| Student (U-Net) | done | `models/student.pt`, `student.onnx` | 96 % agreement with teacher on held-out samples (confident px) |
| Delivered segmentation | done | `outputs/segmentation/<batch>/<sid>/labels.png` | 81.7 % (4 classes) / 87.1 % (3 classes) vs independent AI points |
| Material batch model | done | `models/bid_model.pkl`, `outputs/metrics/bid_scores.json` | 0.61 balanced LOO, p = 0.005 |
| Fingerprint batch model | done | `models/fingerprint_model.pkl`, `outputs/metrics/fingerprint_model_cv.json` | 0.70 LOO, 0.66 session-out, p = 0.002 |
| Decision layer | done | `src/decision.py`, `outputs/metrics/decision_eval.json` | answers 84 %, 85 % right (LOO); 74 % / 91 % session-out |
| B1 vs B2 investigation | done | `same_batch.json`, `triangle_test.json`, `fingerprint_b12.json` | indistinguishable |
| Facts per location / new image | done | `outputs/batchid/<sid>.json`, `bid_explain predict` | text-only, decision first |
| **v3: teammate's method on top of v2** | done | `src/v3.py`, `src/known_spot.py`, `src/texture_model.py`, `src/get4.py`, `outputs/metrics/v3_eval.json` | decision unchanged from v2 (84 % / 85 %); forced mode 0.80 balanced (p = 0.001), 0.70 session-out; matcher 0 false matches |
| Demo | done | `wechat-send-v3/` (+ zip); v2 demo in `wechat-send-v2/` | hold-out `71vgq3fw` → Batch 3, High, correct; training image refused; crop matched |

## 3. Decisions already made (don't redo them unless you have a reason)

**Segmentation:**
1. **No hand labels.**
   - Rule seeds cover the easy pixels and AI-annotated superpixels the hard ones.
   - The rule-based binder seeds were **dropped** (the annotator contradicted them 22/41 times).
   - Round 2 (v2) labelled the regions where the v1 U-Net was uncertain.
2. **SiOx physical constraint** (`src/physics.py`): SiOx only where smoothed BSE is above the upper multi-Otsu
   threshold.
3. **Segmentation detectors: BSE + Inlens only.** ETD/SE changed fractions by < 0.5 pp, and 4 locations have SE
   instead of ETD.
4. **Report composition as 3 classes** (pore / carbon / SiOx); the graphite-vs-binder split is experimental (binder
   precision ~0.5).

**Batch identification:**

5. **Decision = agreement rule, no tuned thresholds** (`src/decision.py`):
   - both models agree → that batch;
   - both in {B1, B2} but split → "Batch_1 or Batch_2";
   - otherwise → "unsure".
6. **Facts for the 31 known locations use leave-one-out probabilities** (honest). New images use models trained on
   all 31; `--holdout <sid>` retrains without a known location for demos.
7. **Batch-ID discipline:** every scored configuration is appended to `outputs/metrics/look_ledger.csv`.
   **Every extra scored configuration inflates the reported accuracy.** Log anything new as exploratory, and test
   the final system once on the blind set.

**LLM input:**

8. **v3 additions (teammate's design, rebuilt from his description):**
   - training-image guard, then known-location matcher (15 SD **and** 2x the next location; the margin was needed
     for crops);
   - calibrated probabilities and a nested High threshold (>= 90 % held-out accuracy);
   - GET4 range rule;
   - forced mode (`--forced`).
   The tile texture model does **not** vote, because it lowered held-out accuracy
   (`outputs/metrics/v3_eval_with_texture.json`).
9. **The LLM gets text only.** Maps are described in words by `src/map_text.py`. `summary_sentences` are filled in
   from numbers in the same file.

## 4. The most important open issues
1. **Session confound.**
   - Acquisition properties alone predict batch at 0.68.
   - The fingerprint model reads noise, banding and grey levels, i.e. how the image was taken.
   - It survives session hold-out (0.66), but a blind set imaged in new sessions or with other settings may score
     lower.
   - Ask the organisers (report §10).
2. **Batch 1 vs Batch 2 look identical** (report §7):
   - 3/77 features differ (3.9 expected by chance);
   - the blinded odd-one-out test scores 27 % (chance 33 %).
   - Leads: B2 ~3.5 pp more porous [0.9, 6.3]; B1 has stronger ETD vertical stripes (likely curtaining).
3. **Batch 2 is the weak batch:** 3/7 answered "unsure" in leave-one-out.

## 5. Known weaknesses of the segmentation
- **pore ↔ binder and graphite ↔ binder confusion:** binder precision 0.52, pore 0.69.
- **Porosity:** 17.3 % vs 19.6 % [15.1, 25.1] from independent points (v1 was 14.9 %).
- **Binder amount is partly a session readout.**
- **Student vs teacher pore fraction differs by up to 5.5 pp** on one held-out sample.

## 6. Suggested next steps
1. **Narrator layer.**
   - Feed `facts.json` as text and quote only numbers that are in it.
   - Start from `summary_sentences` and `decision`; never present a model's own pick as the answer.
2. **Organiser answers** (report §10):
   - If imaging settings were identical and B1/B2 differ by design, re-test the B1-vs-B2 leads (porosity, BSE
     contrast ratio, ETD stripes) as a declared new configuration.
   - If the blind set is in new sessions, expect the fingerprint model to weaken and lean on the material model.
3. **Get the teammate's original v3 classifier code into the repo.**
   - Our fingerprint model is an exact replication (23/31).
   - The texture model and forced mode are rebuilt from his description, and his texture model may score better.
   - Compare on the same nested folds (`src/v3.py evaluate`).
4. **Segmentation:** train the nnU-Net on `data/nnUNet_raw/Dataset501_SEMAnode`, or label more pore/binder superpixels
   (costs API money).

## 7. Practicalities
- **Python 3.14, Windows.** GPU needs torch 2.11.0+cu128. Install torch **before** `segmentation-models-pytorch` with
  `--no-deps`, or pip replaces it with a CPU build.
- **Windows multiprocessing:** inline `python -` scripts cannot use `ProcessPoolExecutor`; put the code in a module.
- **Write text files with `encoding="utf-8"`.** The default cp1252 corrupts files containing →, µ, ±.
- **Paid stages** call the Anthropic API: `src/ai_labels.py`, `src/ai_validate.py`, `src/triangle_test.py`. Their
  results are committed. Don't re-run them unless you mean to (≈ $80 spent in v1 + v2).
- **The API key lives in `.env` (git-ignored). Never commit it.**
- **Large files** (raw data, label maps, models, nnU-Net dataset) are in the GitHub release zips. Rebuild them with
  `python -m src.package_release --tag v2.0`.
- **To restore the raw data:** put `Batch_1/2/3` under `data/raw/`, then run `python run_pipeline.py`. All stages are
  deterministic except GPU training. `compare_versions` needs the v1 snapshot in `outputs/metrics/v1/` and
  `models/v1/`.
