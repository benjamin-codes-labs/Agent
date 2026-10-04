"""Export the segmented dataset in nnU-Net v2 raw format, so an nnU-Net can be trained on it directly.

  data/nnUNet_raw/Dataset501_SEMAnode/
    imagesTr/<case>_0000.png  BSE     (uint8, normalised, 50 nm/px)
    imagesTr/<case>_0001.png  Inlens
    labelsTr/<case>.png       0 pore ("background" in nnU-Net terms), 1 graphite, 2 SiOx, 3 CBD, 4 ignore
    dataset.json

Excluded pixels (border, Cu foil, unpolished strips) and, optionally, low-confidence pixels become the
nnU-Net "ignore" label, so they do not contribute to the loss.

Then:  set nnUNet_raw / nnUNet_preprocessed / nnUNet_results, and run
       nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
       nnUNetv2_train 501 2d 0

Usage: python -m src.export_nnunet [--min-confidence 0.7]
"""
import argparse

import numpy as np
from PIL import Image

from . import config as C

DS_DIR = C.ROOT / "data" / "nnUNet_raw" / "Dataset501_SEMAnode"
IGNORE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="mark pixels whose delivering model max-prob is below this as ignore")
    ap.add_argument("--source", default="student")
    args = ap.parse_args()
    (DS_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DS_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)
    n = 0
    for batch, sid in C.sample_ids():
        case = f"{batch.lower()}_{sid}"
        ch = C.load_channels(sid, ["BSE", "Inlens"])
        for i, d in enumerate(("BSE", "Inlens")):
            Image.fromarray((ch[d] * 255).astype(np.uint8)).save(DS_DIR / "imagesTr" / f"{case}_{i:04d}.png")
        lab = np.load(C.proc_dir(sid) / "final_labels.npy").copy()
        if args.min_confidence > 0:
            maxp = np.load(C.proc_dir(sid) / f"{args.source}_maxp.npy")
            lab[maxp < int(args.min_confidence * 255)] = C.EXCLUDED
        lab[lab == C.EXCLUDED] = IGNORE
        Image.fromarray(lab.astype(np.uint8)).save(DS_DIR / "labelsTr" / f"{case}.png")
        n += 1
    C.write_json(DS_DIR / "dataset.json", {
        "name": "SEMAnode",
        "description": "Graphite + SiOx anode SEM cross-sections, 4-phase segmentation from label-free teacher/student pipeline",
        "channel_names": {"0": "BSE", "1": "Inlens"},
        "labels": {"background": 0, "graphite": 1, "SiOx": 2, "CBD": 3, "ignore": IGNORE},
        "numTraining": n,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "note": "label 0 ('background' as nnU-Net requires) is the PORE class",
    })
    print(f"nnU-Net dataset with {n} cases -> {DS_DIR}")


if __name__ == "__main__":
    main()
