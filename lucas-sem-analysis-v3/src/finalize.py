"""Stage 5a: choose the delivered label maps and write the segmented dataset.

Per sample, writes outputs/segmentation/<batch>/<sid>/:
  labels.png    uint8 label map at 50 nm/px: 0 pore, 1 graphite, 2 SiOx, 3 CBD, 255 excluded
  overlay.jpg   colour overlay on BSE (blue pore, purple graphite, orange SiOx, green CBD, red excluded)
  bse.png, inlens.png   the normalised half-resolution inputs the model saw (uint8)
and data/processed/<sid>/final_labels.npy for the later stages.

Usage: python -m src.finalize --source student|teacher
"""
import argparse

import numpy as np
from PIL import Image

from . import config as C


def main():
    ap = argparse.ArgumentParser()
    # student, teacher, or a tagged teacher run such as teacher_cpu (src.teacher train --tag _cpu)
    ap.add_argument("--source", default="student")
    args = ap.parse_args()
    for batch, sid in C.sample_ids():
        lab = np.load(C.proc_dir(sid) / f"{args.source}_labels.npy")
        np.save(C.proc_dir(sid) / "final_labels.npy", lab)
        out = C.OUT_SEG / batch / sid
        out.mkdir(parents=True, exist_ok=True)
        ch = C.load_channels(sid, ["BSE", "Inlens"])
        Image.fromarray(lab).save(out / "labels.png")
        Image.fromarray(C.overlay_rgb(ch["BSE"], lab, alpha=0.45)).save(out / "overlay.jpg", quality=88)
        for det, name in (("BSE", "bse"), ("Inlens", "inlens")):
            Image.fromarray((ch[det] * 255).astype(np.uint8)).save(out / f"{name}.png")
    C.write_json(C.OUT_SEG / "README.json", {
        "source_model": args.source, "nm_per_px": C.NM_PER_PX,
        "label_values": {**{str(k): c for k, c in enumerate(C.CLASSES)}, "255": "excluded"},
        "files": {"labels.png": "uint8 label map", "overlay.jpg": "colour overlay on BSE",
                  "siox_instances.png": "uint16 SiOx particle instance ids (0 = none)",
                  "bse.png / inlens.png": "normalised half-resolution inputs (uint8)"}})
    print(f"wrote {len(C.sample_ids())} samples from the {args.source} model -> {C.OUT_SEG}")


if __name__ == "__main__":
    main()
