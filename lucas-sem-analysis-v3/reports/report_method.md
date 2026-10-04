## Method

### Data and preprocessing
31 samples (Batch_1: 7, Batch_2: 7, Batch_3: 17), each imaged with BSE, Inlens and ETD (27) or SE (4), 25.000 nm/px,
~175 x 40-58 um fields. All 93 files were verified (`data/manifest.csv`: size, pixel size, sha256). The three RGB
channels are identical except at the first or last image column in 39 files, which fall inside the excluded
8-px border. Detectors are co-registered to within 0.4 px (phase correlation, `data/qc.csv`), so per-pixel
multi-detector features are safe.

Each detector is read from the green channel, 2x block-averaged to 50 nm/px, normalised per image (p0.5-p99.5 of
non-excluded pixels -> [0, 1]) and median-filtered (3x3). Excluded (label 255): an 8-px (full-res) border; the Cu
current collector in `epqdaau9` (saturated BSE connected to the image edge, dilated 1 um); and two visually
confirmed unpolished regions (`i9jiqjwl` top strip, `5n1q8atc` top-left patch; `data/exclusions.json`).

### Labels without hand annotation
1. **Rule seeds** (`src/seeds.py`) for unambiguous pixels: SiOx = compact BSE-bright objects >= 2 um^2 (eroded);
   pore = black in BSE and ETD/SE, or dark in both Inlens and ETD/SE (eroded); graphite = interiors of large,
   smooth mid-grey regions.
2. **AI-annotated superpixels** (`src/ai_labels.py`): the plan's assumption that grey-floored open pores are dark
   in Inlens does not hold in every sample (in e.g. `kbdh4tri` polished graphite is the darkest Inlens phase),
   so no intensity rule finds them. 1,178 SLIC superpixels (~1 um) were shown to Claude Opus 5.5 (max effort,
   Batch API) as BSE | Inlens | ETD crops with the superpixel outlined, mixed across samples within each
   request. 186 already-seeded superpixels were included blind as a calibration check: the annotator agreed with
   the pore, graphite and SiOx rules 100 % of the time (44/44, 53/53, 48/48) but with the CBD rule only 19/41
   (it called the rest pore walls or graphite slivers). The CBD rule was therefore dropped and CBD is learned from
   the AI labels only. Labels with medium/high confidence were used (692 of 1,178); 20 % of superpixels were held
   out from training for validation.

### Teacher (LightGBM, `src/teacher.py`)
54 per-pixel features from BSE and Inlens (Gaussian, gradient magnitude, LoG, Hessian eigenvalues, local SD at
sigma 1, 2, 4, 8 px; fine-texture energy; value minus local mean as depth/charging context). Class-balanced
training pixels from seeds + AI labels (AI pixels weighted x4), 300 trees, two self-training rounds (pixels with
max-prob >= 0.9 that agree with a 5x5 majority filter are added as labels; intermediate rounds predicted on a
stride-2 grid). Post-processing: objects < 0.1 um^2 are absorbed by their surroundings, and a physical
constraint (`src/physics.py`): a pixel can only be SiOx if its smoothed BSE value is above the image's upper
multi-Otsu threshold (SiOx is defined by Z-contrast); small holes inside SiOx particles are filled. Without the
constraint, rims and bright sub-surface walls inflated SiOx by 1.5-4 pp.

### Student (U-Net, `src/student.py`)
`segmentation_models_pytorch` U-Net, ResNet-18 encoder (ImageNet weights, 2-channel input BSE + Inlens), slim
decoder, 12.5 M parameters. Targets: teacher labels, ignoring pixels with teacher max-prob < 0.7 and excluded
pixels. Cross-entropy + Dice, AdamW + one-cycle, AMP, 512 x 512 crops; augmentation = horizontal flips only (the
vertical axis is the through-thickness direction), per-channel contrast/brightness/gamma, noise, synthetic
Inlens banding, small elastic deformation. 25 samples train, 6 held out (2 per batch). Inference: 512-px tiles,
64-px overlap, soft-blended; the same SiOx constraint; exported to ONNX.

### Validation without hand labels
- Agreement with the two earlier baselines: `thr` (BSE multi-Otsu threshold; a lower bound for pores) and `vis`
  (vision-LLM per-sample estimate). SiOx is compared with `thr` after the same size filter `thr` used
  (opening 0.15 um, objects >= 2 um^2).
- Expected rankings; teacher retraining stability on random 20-sample subsets x 2 seeds.
- Student vs teacher on the 6 held-out samples.
- **Independent AI point check** (`src/ai_validate.py`): 240 points drawn uniformly over all valid pixels
  (excluding superpixels used for training), each classified by Claude Opus 5.5 from a 6.4 um three-detector crop
  with the point ringed, without seeing any model output -> unbiased accuracy and an independent estimate of the
  dataset's phase composition; plus 30 points per predicted class -> per-class precision.
