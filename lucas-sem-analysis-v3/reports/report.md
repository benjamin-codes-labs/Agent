# SEM anode segmentation and batch identification: report (v3)

*31 samples (Batch_1: 7, Batch_2: 7, Batch_3: 17), 2026-10-03. No pixel was labelled by hand. v1 of this report is
in the git history; its numbers are kept in `outputs/metrics/v1/` and `models/v1/`.*

## 0. v3 update: v2 + the teammate's method

v3 keeps everything in v2 and adds the teammate's (Guanyi's) v3 design. His original classifier code was not
available, so `src/texture_model.py`, `src/known_spot.py` and `src/v3.py` rebuild it from his description, reusing
his `field_matching.py` edge maps and his `GET4.py` error bars (copied unchanged to `src/get4.py`).

**What a new image goes through:**
1. **Training-image guard:** an image pixel-identical to one of the 93 training images is refused.
2. **Known-location matcher:** a strong edge-map match (>= 15 SD and 2x the next location) means a known spot, and
   its batch is returned as certain.
3. **v2 agreement rule on calibrated probabilities:**
   - imaging side (fingerprint) vs material side (segmentation + DINOv2);
   - answers: batch / "Batch 1 or Batch 2" / "unsure (leaning X)";
   - **High tier** = average calibrated probability above the level at which held-out answers were >= 90 % right.
4. **GET4 material-range rule:** can turn a "Batch 1 or Batch 2" into one batch.
5. **Forced mode** (`--forced`): fingerprint decides "Batch 3 or not"; then the range rule, else a Batch 1-vs-2
   fingerprint specialist.
6. **Tile texture model:** reported in `facts.json`; it does not vote (see below).

**How it was tested.** Everything chosen from data is chosen without the held-out location (nested): the models,
calibration temperatures, the High threshold and the batch ranges. The DINOv2 texton vocabulary (label-free) is
fitted per outer fold. Results: `outputs/metrics/v3_eval.json`.

| | v2 | **v3** | v3, texture voting (teammate's spec) |
|---|---|---|---|
| LOO: answers given | 26/31 (84 %) | **26/31 (84 %)** | 25/31 (81 %) |
| LOO: right when answering | 22/26 (85 %) | **22/26 (85 %)** [66, 94] | 20/25 (80 %) |
| LOO: High tier right | 7/7 | **10/12** | 11/12 |
| Sessions held out: answers given | 23/31 (74 %) | **23/31 (74 %)** | 22/31 (71 %) |
| Sessions held out: right when answering | 21/23 (91 %) | **21/23 (91 %)** [73, 98] | 20/22 (91 %) |
| Sessions held out: High tier right | 5/5 | **6/7** | 7/8 |
| **Forced mode, balanced accuracy (always answers)** | — | **0.80** (B1 6/7, B2 5/7, B3 14/17), permutation **p = 0.001** (1000 shuffles; null 99th pct 0.60) | same |
| Forced mode, sessions held out | — | **0.70** (B1 4/7, B2 5/7, B3 14/17) | same |

**Single models** (LOO balanced accuracy): fingerprint 0.70, material side 0.61, tile texture 0.58 (BSE 0.68,
Inlens 0.43, ETD/SE 0.58).

**Known-location matcher** (`outputs/metrics/known_spot_eval.json`):
- full images matched through another detector's view: 93/93;
- random 50 % crops: 71/93 via another detector, 86/93 of the same image;
- **false matches to a different location: 0/93 for full images and 0/93 for crops.**

A 2x margin over the next location was added to the teammate's 15 SD threshold: without it, crops gave 12/93 false
matches.

**What this means:**
- **The abstaining decision is unchanged from v2.** Calibration does not change which batch a model picks, so v2's
  rule gives the same answers. What v3 adds to it is a data-driven High tier: more High answers (12 vs 7), but 2
  of them wrong.
- **The tile texture model, as rebuilt, does not help.** Letting it vote lowered accuracy when answering from 85 %
  to 80 %, so it is reported, not used. The teammate's original may do better.
- **The GET4 range rule almost never fires** (1/31 locations, right). With 7 locations per batch, the composition
  ranges of Batch 1 and Batch 2 overlap almost completely.
- **Forced mode is the best always-answer system so far:** 0.80, against 0.70 for the fingerprint model alone. It
  gets 11 of the 14 Batch 1/2 locations right in leave-one-out, but only 9/14 with sessions held out, so most of
  its Batch 1-vs-2 skill may be session-related.
- **Forced mode was designed by the teammate after seeing these data.** The p-value tests luck, not that design
  choice.
- **The matcher only matters if blind images show the 31 known spots.** Whether that counts is for the organisers to
  say (§10).

**Demo** (`wechat-send-v3/`): `71vgq3fw` run as a new image, with everything refitted without it.
- **New image:** Batch 3, High, correct. The matcher found no known location: best 11 SD, below 15.
- **The same image without `--holdout`:** refused as a training image.
- **A 50 % crop of the same spot:** matched at 152 SD, so Batch 3 (certain).
- **Runtime:** about 30 s with saved models (BSE + Inlens + ETD).

**API cost of v3: $0** (no API calls).

## 1. Summary

**Segmentation (deliverable).**
- 4-class label maps for all 31 samples at 50 nm/px: 0 pore, 1 graphite, 2 SiOx, 3 binder (CBD), 255 excluded.
- They come from a 12.5 M-parameter U-Net that segments a new BSE + Inlens pair in **0.31 s on the RTX 5070 Ti** and
  **1.5 s on CPU (ONNX)**.
- Dataset composition (mean of samples): **pore 17.3 %** (deep 9.5 % + open/grey-floored 7.7 %), **graphite 61.0 %**,
  **SiOx 6.3 %**, **binder 15.4 %**.

**Accuracy (independent AI annotator, blind, uniformly random points).**
- **4 classes: 81.7 %** (95 % CI 76.3-86.1 %); 100 % on the 113 points the annotator marked high-confidence.
- **3 classes (pore / carbon / SiOx, binder merged into carbon): 87.1 %** (82.2-90.7 %).
- **Per-class precision:** SiOx 0.96, graphite 0.83, pore 0.69, **binder 0.52**. The graphite-vs-binder split is the
  unreliable part, so v2 reports composition primarily as 3 classes.
- **v2 vs v1:** the same on random points (81.5 vs 81.9 % on the same 238 points), better in the regions where v1 was
  uncertain (**70 vs 64 %**), and less biased on porosity (17.3 % vs v1 14.9 %; independent estimate 19.6 %
  [15.1, 25.1]).

**Batch identification (v2: two models + an agreement rule that may abstain).**
- **Imaging-fingerprint model** (a teammate's design, replicated): **0.70** balanced accuracy leave-one-location-out,
  permutation p = 0.002; **0.66** with whole imaging sessions held out; "Batch 3 vs Batch 1-or-2" 0.88.
- **Material model** (segmentation features + DINOv2 texture): **0.61** (p = 0.005); 0.56 session-out.
- **Agreement rule** (answer only when both models agree; "Batch 1 or 2" when they split between those two; otherwise
  "unsure"):
  - leave-one-location-out: answers **26/31 (84 %)**, right **22/26 (85 %)** when it answers;
  - whole sessions held out: answers **23/31 (74 %)**, right **21/23 (91 %)**;
  - **High-confidence tier: 7/7** right (session-out 5/5).
- **Batch 1 and Batch 2 are statistically indistinguishable in this data** (§7). An honest system has to be able to
  say "Batch 1 or Batch 2", and v2 does.
- **The batches were partly imaged under different conditions** (§8). Acquisition properties alone predict batch at
  0.68, and the fingerprint model is largely an acquisition model. Accuracy on blind images taken in new sessions may
  be lower; the session-out numbers are the better guide.

**Explanation for a text-only LLM.** Every prediction writes a `facts.json`:
- the final decision first, then the two models;
- the segmentation and evidence maps described in words (phases, 3x3 grid, largest objects, evidence by phase and
  region);
- fingerprint measurements with plain-language meanings and an empirical calibration line;
- summary sentences built only from those numbers.

The LLM never sees an image.

## 2. What is new in v2

| | v1 | v2 |
|---|---|---|
| Training labels | 1,178 AI-labelled superpixels | **+ 992 superpixels in the regions where v1 was uncertain** (active learning, round 2) |
| Segmentation, random points | 81.9 % | 81.5 % (same 238 points) |
| Segmentation, v1-uncertain regions (97 held-out superpixels) | 63.9 % | **70.1 %** |
| Porosity (dataset mean) vs independent estimate 19.6 % | 14.9 % | **17.3 %** |
| Composition reporting | 4 classes | **3 reliable classes** + the binder split marked experimental |
| Batch models | material model only | **fingerprint model + material model** |
| Batch answer | always one batch | one batch, **"Batch 1 or Batch 2"**, or **"unsure"**, with a confidence tier and track record |
| Batch accuracy, leave-one-out | 0.59 balanced, always answers | **85 % when answering, 84 % coverage** |
| Batch accuracy, sessions held out | 0.47 | **91 % when answering, 74 % coverage** |
| New-image inputs | BSE + Inlens | BSE + Inlens, **ETD/SE optional** (fingerprint model) |
| LLM input | numbers | numbers **+ maps described in text** |
| Demo | training model on a training image | **honest hold-out**: both batch models retrained without the demo location (`--holdout`) |

## 3. What was delivered

| Path | Content |
|---|---|
| `outputs/segmentation/<batch>/<sid>/` | `labels.png` (uint8), `overlay.jpg`, `siox_instances.png` (uint16), `bse.png`, `inlens.png` |
| `outputs/metrics/phase_fractions.csv` | per sample: 4 classes, deep / open pore, SiOx after the baseline's size filter, excluded %, valid px |
| `outputs/metrics/batch_stats.csv` | per batch x class: mean, SD, range; Kruskal-Wallis; Holm-corrected Mann-Whitney; session-stratified permutation p |
| `outputs/metrics/validation.csv`, `validation_summary.json`, `v1_vs_v2.json` | all segmentation validation numbers, v1 vs v2 |
| `outputs/metrics/fingerprint_values.csv`, `fingerprint_model_cv.json` | 51 fingerprint measurements per location; honest per-view probabilities and calibration |
| `outputs/metrics/decision_eval.json` | the agreement rule's track record: coverage, accuracy, per type, per tier, per sample |
| `outputs/metrics/same_batch.json`, `triangle_test.json`, `fingerprint_b12.json` | the Batch 1 vs Batch 2 investigation (§7) |
| `outputs/metrics/look_ledger.csv` | every scored batch-ID configuration, in order |
| `outputs/batchid/<sid>.json`, `<sid>_evidence.jpg` | per-location facts for the LLM (v2 layout, honest leave-one-out decision) and evidence map |
| `models/student.pt`, `student.onnx`, `teacher.txt`, `bid_model.pkl`, `fingerprint_model.pkl` | deployable models (release asset) |
| `data/nnUNet_raw/Dataset501_SEMAnode/` | the segmented dataset in nnU-Net v2 format (release asset) |
| `wechat-send-v2.zip` | one-image demo: input TIFFs → outputs, `SUMMARY.jpg`, README |

**New image, whole chain:**

```
python -m src.bid_explain predict --bse X_BSE.tif --inlens X_Inlens.tif [--etd X_ETD.tif] --out out/X
```

It takes about 21 s on this laptop: 8 s of fingerprint measurements, the rest segmentation, features, DINOv2 and
explanation. `--holdout <sid>` retrains both batch models without a known location first, for demos.

## 4. Method

See [report_method.md](report_method.md) for the full description. In short:
- **Labels (no hand annotation):**
  - Rule seeds label the unambiguous pixels.
  - Claude Opus 5.5 labelled 1,178 superpixels in the ambiguous regions (round 1).
  - **v2, round 2 (active learning):** Claude labelled 992 more superpixels, chosen where the v1 U-Net was least
    certain (pore vs binder, and other low-confidence regions) plus a random pore/binder control set. 803 were used
    for training and 189 held out.
  - Rule-based binder seeds were dropped: the annotator contradicted them 22/41 times.
- **Teacher:** LightGBM on 54 multi-scale BSE + Inlens features, with self-training and a physical SiOx constraint
  (SiOx must be BSE-bright).
- **Student:** a U-Net distilled from the teacher's confident pixels; it delivers the label maps.
- **Batch models and decision layer:** §6.
- **Validation:**
  - against the earlier threshold/visual baselines;
  - retraining stability;
  - held-out samples;
  - independent AI point checks (uniform and class-stratified), with the annotator never shown a model output.

## 5. Segmentation results

### 5.1 Phase fractions by batch (% of valid area, mean ± SD, v2)

| Phase | Batch_1 (7) | Batch_2 (7) | Batch_3 (17) | Kruskal-Wallis p | Session-stratified p | Holm MW 1v2 / 1v3 / 2v3 |
|---|---|---|---|---|---|---|
| pore (all) | 14.4 ± 1.9 | 17.8 ± 3.4 | 18.2 ± 2.1 | **0.004** | **0.031** | 0.076 / **0.0005** / 0.62 |
| — deep (BSE-dark) | 8.1 ± 1.6 | 9.4 ± 1.5 | 10.2 ± 1.9 | 0.071 | 0.065 | 0.42 / 0.079 / 0.45 |
| — open (grey-floored) | 6.3 ± 1.3 | 8.5 ± 2.3 | 8.1 ± 1.1 | 0.030 | 0.40 | 0.15 / 0.020 / 0.90 |
| graphite | 63.1 ± 5.6 | 61.9 ± 3.0 | 59.8 ± 4.5 | 0.33 | 0.081 | n.s. |
| SiOx | 7.9 ± 3.4 | 5.7 ± 1.3 | 5.9 ± 0.9 | 0.20 | 0.58 | n.s. |
| binder (CBD) | 14.7 ± 3.9 | 14.6 ± 3.6 | 16.1 ± 4.6 | 0.67 | 0.56 | n.s. |

**How to read this table:**
- **Batch 1 is less porous.** In v2 this now also holds when only samples from the same imaging session are compared
  (p = 0.031; v1 0.073).
- **SiOx:** Batch 1's higher mean comes from two samples imaged in one session (`4ih2ggld`, `5n1q8atc`, the only
  2316-px images). The rest of Batch 1 is 4.7-7.4 %.
- **Binder** amount clusters by imaging session, so treat it as partly a session readout.
- **SiOx particles:** median equivalent diameter 1.2-1.3 um; p90 3.4-4.0 um; 14-19 particles per 1000 um^2.

### 5.2 Validation (v2; v1 in brackets where it changed)

| Check | Result | Target |
|---|---|---|
| SiOx vs `thr` (same size filter), mean abs. diff | **0.13 pp**, Spearman 0.97 | within 1.5 pp per sample: **met in all 31** |
| Pore >= `thr` (BSE lower bound) | **31 / 31** | every sample |
| Pore between `thr` and `vis` | 13 / 31 (v1 20) — v2 pore now more often above the visual estimate | "most" |
| Expected rankings | **4 / 4** (v1 3/4) | all |
| Stability (teacher refit, 3 subsets x 2 seeds), SD of fractions | mean 0.01-0.25 pp, max 0.72 pp | < 1 pp: **met** |
| Student vs teacher, 6 held-out samples | 96.1 % of confident px (91.0 % all px) | >= 90 %: **met** |
| Student vs teacher fractions, held-out samples | max diff 5.5 pp (pore; v1 3.9) | within 1 pp: **not met** |
| AI point check, uniform (n = 240) | **81.7 %** [76.3, 86.1]; 100 % on the 113 high-confidence points | — |
| Same, 3 classes (binder merged into carbon) | **87.1 %** [82.2, 90.7] (v1 87.5 %) | — |
| AI point check, per predicted class (n ≈ 30 each), precision | SiOx 0.96, graphite 0.83 (v1 0.77), pore 0.69 (v1 0.75), binder 0.52 (v1 0.50) | — |
| Held-out round-1 AI superpixels (n = 141) | 84.4 % (v1 83.7 %) | — |
| **Held-out round-2 superpixels in v1-uncertain regions (n = 97)** | **70.1 %** [60.4, 78.3] (v1 63.9 %) | — |
| Inference time per image | **0.31 s** GPU, **1.5 s** CPU (ONNX); GPU/CPU label agreement 99.99 % | <= 3 s GPU, <= 20 s CPU: **met** |

**Independent dataset composition.** From the uniform points (Claude's labels, 95 % CI):

| | pore | graphite | SiOx | binder |
|---|---|---|---|---|
| AI annotator | 19.6 [15.1, 25.1] | 58.8 [52.4, 64.8] | 7.9 [5.1, 12.0] | 13.8 [10.0, 18.7] |
| Model at the same points | 17.9 | 64.6 | 5.0 | 12.5 |

The model's porosity is now inside the annotator's interval (v1 was below it).

**Confusion on the uniform points** (rows = Claude, columns = model; pore, graphite, SiOx, binder):

```
pore      33   7   0   7
graphite   4 135   0   2
SiOx       0   2  12   5
binder     6  11   0  16
```

**Known weaknesses:**
1. **Pore ↔ binder.** Lacy binder is itself nanoporous, and pore walls look textured. It is the hardest call for the
   annotator too.
2. **Graphite ↔ binder.** Binder precision is ~0.5. Use the 3-class composition (pore / carbon / SiOx) for
   decisions, and the binder number only as an indication.
3. **Some open-pore interiors still read as graphite.** Sub-surface flakes seen through a pore look like graphite.
4. **Student and teacher disagree by up to 5.5 pp** in pore fraction on one held-out sample (`epqdaau9`, pore ↔
   binder swaps).

## 6. Batch identification

### 6.1 The models

All scores are leave-one-location-out unless stated: each location is predicted by a model trained on the other 30.
"Session-out" holds out all locations that share an imaging-session proxy (image height) together.

| Model | Balanced accuracy (LOO) | B1 / B2 / B3 | Session-out | Permutation p | Role in v2 |
|---|---|---|---|---|---|
| **Fingerprint (FP)**: shrinkage LDA per detector view on 51 acquisition measurements, views averaged | **0.70** (23/31) | 5/7, 4/7, 14/17 | **0.66** (3/7, 5/7, 14/17) | **0.002** | input 1 of the decision |
| **Material (F)**: 5 segmentation features (S) + DINOv2 texton head (D), averaged | 0.61 | 5/7, 2/7, 14/17 | 0.56 | 0.005 | input 2 of the decision; evidence maps |
| S alone | 0.48 | 5/7, 1/7, 10/17 | 0.48 | 0.09 | — |
| D alone (diagnostic) | 0.62 | 3/7, 5/7, 12/17 | 0.54 | — | — |
| FP + S averaged (exploratory) | 0.66 | 5/7, 3/7, 14/17 | 0.66 | 0.007 | not used: no better than FP |
| Acquisition probes only (nuisance check) | 0.68 | 4/7, 4/7, 15/17 | — | — | warning |
| FP, "Batch 3 vs Batch 1-or-2" | 0.88 | — | 0.85 | — | — |

**The fingerprint model** measures each detector image's noise, not the material:
- noise level and spikiness;
- correlation along and across scan lines;
- horizontal banding and its period;
- vertical stripe strength;
- grey-level range, clipping and number of grey levels used;
- file compressibility.

It is a teammate's method, rebuilt (`src/fingerprint.py`, `src/fingerprint_model.py`) and replicated exactly
(23/31). It runs on whichever views are present (BSE, Inlens, ETD/SE).

**Calibration of the fingerprint probabilities** (how often the top pick was right, by the probability it claimed):

| Claimed | LOO | Session-out |
|---|---|---|
| 0.99-1.00 | 8/8 | 7/7 |
| 0.90-0.99 | 4/6 | 0/3 |
| 0.60-0.90 | 8/12 | 11/14 |
| < 0.60 | 3/5 | 4/7 |

Raw LDA probabilities are over-confident below 0.99, so `facts.json` reports the empirical rate next to the claim.

### 6.2 Decision and uncertainty layer (`src/decision.py`)

**Rule.** No threshold was tuned:
- both models pick the same batch → that batch (**specific**);
- they disagree but both pick Batch 1 or Batch 2 → **"Batch 1 or Batch 2"** (**pair**);
- otherwise → **"unsure"**.

**Confidence tier:**
- **High:** specific answer, both models ≥ 0.6, the fingerprint's detector views agree, and no novelty or shortcut
  flags.
- **Medium:** all other specific answers, and pair answers.
- **Low:** unsure.

A prediction set (every batch whose average probability is ≥ 0.2) is reported alongside.

**Results** (`outputs/metrics/decision_eval.json`; the inputs are each model's honest held-out probabilities):

| | Leave-one-location-out | Sessions held out |
|---|---|---|
| Coverage (answers given) | **26/31 (84 %)** | **23/31 (74 %)** |
| Right when answering | **22/26 (85 %)** [66, 94] | **21/23 (91 %)** [73, 98] |
| Specific answers right | 19/23 (83 %) | 15/17 (88 %) |
| "Batch 1 or Batch 2" answers right | 3/3 | 6/6 |
| High tier | **7/7** | **5/5** |
| Medium tier | 15/19 (79 %) | 16/18 (89 %) |
| Unsure (true batch) | 5 (B2 x3, B3 x2) | 8 (B2 x4, B3 x4) |

**Outcome per true batch, leave-one-location-out:**
- **Batch 1:** 4 right, 2 "B1 or B2", 1 wrong (called B2).
- **Batch 2:** 2 right, 1 "B1 or B2", 1 wrong (called B1), 3 unsure.
- **Batch 3:** 13 right, 2 wrong (called B2), 2 unsure.

**How to read these numbers honestly:**
- **"Right when answering" is not balanced accuracy.** Batch 3 is 17 of 31 locations, and a "Batch 1 or Batch 2"
  answer is easier to get right than a specific batch. The specific-answer rate (83 % / 88 %) and the per-batch
  outcomes above are the stricter view.
- **Batch 2 is the weak batch.** It is the one most often answered "unsure".
- **The rule was designed after both models' leave-one-out results were known.** It has no tuned threshold, but it
  is still a post-hoc design on 31 locations; it is logged as exploratory in the look ledger. The blind set is the
  real test.

### 6.3 Explanation for a text-only LLM

`python -m src.bid_explain predict` (new image) and `samples` (the 31 known locations) write `facts.json`. Its
fields, in order:

| Field | Content |
|---|---|
| `summary_sentences` | plain-English sentences filled in from the numbers below; nothing else is generated text |
| `field_guide` | tells the reader that `decision` is the final answer and the models' own picks are not |
| `decision` | answer, type, confidence tier, reasons, prediction set, average probabilities, track record of this answer type |
| `phases` | 4-class and 3-class composition, with the reliability note |
| `fingerprint_model` | probabilities overall and per detector, the measurements that pushed hardest (with plain meanings and robust z), calibration line |
| `material_model` | probabilities (fused, S, D), the 5 features with their push toward the winner, evidence statistics, `map_text` |
| `material_model.map_text` | the maps in words: evidence by phase and by region of the 3x3 grid, composition per grid cell, notable regions, largest pores and SiOx particles, Clark-Evans clustering |
| `qc`, `acquisition`, `novelty` | excluded area, Cu foil, nearest training acquisition, "unlike anything seen" flags |

**Evidence maps are exact.** Per-token contributions of the DINOv2 head sum to its logits (reconstruction error
~1e-13), so "58 % of the evidence lies on graphite" is a computed fact, not a saliency heuristic.

**Training locations get honest facts.** For the 31 known locations, the `decision` and fingerprint probabilities
come from the leave-one-out models, not from models that saw them.

**Demo** (`wechat-send-v2/`, location `71vgq3fw`, Batch 3, both batch models retrained without it):
- **Answer: Batch 3, High confidence.** Fingerprint 1.00 (all three views agree), material model 0.63.
- **Composition:** pore 16.7 %, carbon 76.8 %, SiOx 6.5 %.
- **Strongest cue:** BSE graininess lower than in any training image (z −3.0).
- **Runtime:** 32 s including the retraining.

### 6.4 Exploratory variants (all logged; none replaces the above)

| Variant | LOO | Session-out | Verdict |
|---|---|---|---|
| F + raw BSE brightness head | 0.59 | 0.52 | worse than F; brightness alone drops from 0.65 to 0.52 when sessions are held out (session shortcut) |
| Two-step: F for "Batch 3?", KPI classifier for "B1 or B2?" (v2 labels) | 0.70 | 0.70 | B1-vs-B2 step p = 0.08 with 3 of 77 KPIs picked per fold; not adopted |
| Cascades (DINO decides B1 vs B2; head chosen in nested CV) | 0.56-0.61 | — | no gain |
| Earlier fingerprint version (all 53 features, one LDA) | 0.63 | 0.56 | superseded by the per-view design |
| Fingerprint, Batch 1 vs Batch 2 only (n = 14) | 9/14 | 8/14 | p = 0.21: not better than chance |

## 7. Are Batch 1 and Batch 2 the same?

Tested because no model separates them reliably. **In this data they are statistically indistinguishable.**

| Test | B1 vs B2 | B1 vs B3 | B2 vs B3 |
|---|---|---|---|
| Features with p < 0.05 (77 features; 3.9 expected by chance), equal n = 7 | **3** (perm p 0.46) | 13.5 (p 0.013) | 6 (p 0.13) |
| Multivariate energy distance, permutation p | **0.19** | 0.009 | 0.12 |
| Blinded odd-one-out test by Claude Opus 5.5 (30 triads each; chance 33 %) | **27 %** | 33 % | 50 % (p 0.04) |
| Nearest-neighbour mixing within B1 + B2 (same-batch share vs chance) | **0.38 vs 0.46** (p 0.89) | — | — |

**The odd-one-out test** shows the vision model three crops, two from one batch, and asks which is the odd one out.
It can tell Batch 2 from Batch 3 a little (50 %), but is at chance for everything involving Batch 1, and below
chance for B1 vs B2.

**Possible small differences** (B2 − B1, 95 % CI; n = 7 + 7, so these are leads, not findings):
- **Porosity +3.5 pp** [0.9, 6.3] (p = 0.04). The smallest difference these samples could reliably detect is 4.5 pp.
- **Open porosity +2.2 pp** [0.6, 4.1].
- **SiOx/graphite BSE contrast ratio +0.33** [0.08, 0.57]. It also tracks Inlens grey level, i.e. acquisition.
- **ETD graphite roughness lower in B2** (d = −1.6).
- **No difference** in binder (−0.1 pp), graphite (−1.2 pp) or SiOx particle size.

**The fingerprint B1-vs-B2 lead** is ETD vertical-stripe strength: B1 higher, Cliff's δ = 0.76, same direction in all 3
same-session B1/B2 pairs. Vertical stripes in ETD images are typically ion-milling curtaining, and this measure
correlates with porosity (ρ = −0.53). It may reflect a preparation difference that tracks porosity, but the
B1-vs-B2 classifier built on it is not better than chance (p = 0.21).

**Conclusion.** Either Batch 1 and Batch 2 are the same material, or they differ by less than these 14 locations can
resolve. The decision layer therefore answers "Batch 1 or Batch 2" when the evidence splits. We would ask the
organisers whether the two batches differ by design (§10).

## 8. The session confound and what the fingerprint measures

- **Image height marks an imaging session, and sessions mix batches.** For example, `ffwubibz` B1, `r17byphk` B2 and
  `cfe5vt7s` B3 were all imaged at 2080 px; `rxax5ozo` B2 and three B3 locations at 2068 px with the SE detector.
- **Acquisition properties alone predict batch at 0.68** (image size, grey levels, noise, banding). The batches were
  at least partly imaged under different conditions or in different sessions.
- **The fingerprint model is mostly an acquisition model.** Its strongest cues are noise level, banding and grey
  levels. It holds up when sessions are held out (0.66), so it is not just recognising image sizes, but it would
  likely degrade on a blind set imaged in new sessions or with different settings.
- **The material model is weaker but measures the material.** Its evidence lies on phases, and its S features are
  physical quantities.
- **The agreement rule uses both,** so an answer needs the imaging fingerprint and the material to point the same
  way.
- **Several batch effects weaken within sessions.** v2's porosity difference survives (session-stratified p = 0.031).
  Binder amount clusters by session.

## 9. Non-ML image KPIs (exploratory, v1 labels)

35 KPIs were computed with plain image processing:
- **thresholds:** multi-Otsu dark/bright fractions;
- **shapes:** pore and SiOx shapes, clustering, fractal dimension;
- **contrast:** contrast ratios, grey-level moments;
- **texture:** flake alignment, power-spectrum slope;
- **heterogeneity:** window CVs, lacunarity.

**Results:**
- **No KPI survives the false-discovery correction.**
- An exploratory classifier on them scores 0.49 (p = 0.14), i.e. not better than chance.
- The one lead is the SiOx/graphite BSE contrast ratio (B1 1.20, B2 1.53, B3 1.67; Cliff's δ B1 vs B2 −0.76). BSE
  contrast depends on mean atomic number, so a real difference would mean a different SiOx stoichiometry. But it
  also tracks acquisition settings.

Details: `outputs/metrics/kpi_stats.csv`, `kpi_topo_values.csv`, `b12_screen.csv`.

## 10. Questions for the organisers

1. **Imaging conditions:** were kV, beam current, working distance, dwell time, detector brightness/contrast and
   scan settings the same for all batches? Which images were taken in the same session?
2. **The blind set:**
   - Will it be imaged in the same sessions and with the same settings as the training images, or in new ones?
   - Will it include all three detectors (BSE, Inlens, ETD/SE)?
   - Are blind images new fields from the same samples, or new samples?
3. **Batch 1 vs Batch 2:**
   - Do they differ by design (formulation, SiOx grade or stoichiometry, calendering, binder content), or are they
     nominally the same?
   - What difference are we expected to detect?
4. **Scoring:**
   - Is an abstention ("unsure") or a set answer ("Batch 1 or Batch 2") acceptable?
   - Is the score accuracy, balanced accuracy, or something that rewards calibrated confidence?
5. **Sample preparation:** were all batches ion-milled the same way? ETD vertical-stripe strength (curtaining)
   differs between Batch 1 and Batch 2.
6. **Ground truth:** is any manually segmented image or measured porosity available to check the segmentation
   against?

## 11. Limitations

- **There is no ground truth.** Segmentation accuracy is estimated against an AI annotator. It is good on clear cases
  (100 % on high-confidence points) and unreliable on the pore/binder boundary.
- **2D area fractions are not volume fractions.** Open-pore detection depends on seeing sub-surface material.
- **One field (~175 x 50 um) per location;** field-to-field variation within an electrode is unknown.
- **31 locations.** One Batch_1 location is worth 4.8 balanced-accuracy points, and every confidence interval above
  is wide.
- **Model choices were made on the same 31 locations** that score them: the decision rule, the fingerprint design and
  the DINO head. Everything beyond the pre-registered v1 configurations is logged in `look_ledger.csv`.
- **Batch identification depends partly on imaging conditions** (§8).

## 12. API cost

Claude Opus 5.5 at max effort, via the Batch API where possible:

| Item | Cost |
|---|---|
| v1: superpixel labels (round 1) | $29.30 |
| v1: uniform + stratified point checks, pilot | $8.03 |
| **v2: active-learning superpixel labels (round 2)** | **$29.88** |
| **v2: blinded odd-one-out (triangle) test** | **$11.25** |
| **v2: advisory "referee" call (ran out of tokens, no output) + one cancelled call** | **≈ $1** |
| **v1 + v2 total** | **≈ $80** |
| Earlier per-sample analysis (separate, before v1) | $33.45 |
