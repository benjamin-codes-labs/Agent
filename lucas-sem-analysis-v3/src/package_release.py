"""Zip the large, git-ignored deliverables for a GitHub release.

  release/sem_anode_segmentation_<tag>.zip   outputs/segmentation/ (label maps, overlays, instances, inputs)
                                             + outputs/metrics/ + reports/
  release/nnunet_Dataset501_SEMAnode_<tag>.zip
  release/models_<tag>.zip                   student.pt, student.onnx, teacher.txt, student_split.json,
                                             bid_model.pkl, fingerprint_model.pkl

Upload with:  gh release create <tag> release/*.zip --title "<tag>" --notes-file reports/report.md

Usage: python -m src.package_release --tag v3.0
"""
import argparse
import zipfile
from pathlib import Path

from . import config as C

REL = C.ROOT / "release"


def add_tree(z, root, arc_root):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            z.write(p, Path(arc_root) / p.relative_to(root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1.0")
    args = ap.parse_args()
    REL.mkdir(exist_ok=True)
    with zipfile.ZipFile(REL / f"sem_anode_segmentation_{args.tag}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, C.OUT_SEG, "segmentation")
        add_tree(z, C.OUT_MET, "metrics")
        add_tree(z, C.REPORTS, "reports")
        z.write(C.ROOT / "README.md", "README.md")
    nn = C.ROOT / "data" / "nnUNet_raw" / "Dataset501_SEMAnode"
    if nn.exists():
        with zipfile.ZipFile(REL / f"nnunet_Dataset501_SEMAnode_{args.tag}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            add_tree(z, nn, "Dataset501_SEMAnode")
    with zipfile.ZipFile(REL / f"models_{args.tag}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("student.pt", "student.onnx", "teacher.txt", "student_split.json", "bid_model.pkl", "fingerprint_model.pkl",
                     "v3_model.pkl"):
            if (C.MODELS / name).exists():
                z.write(C.MODELS / name, name)
        if (C.PROC / "known_spot_index.npz").exists():        # reference edge maps + hashes for the v3 matcher
            z.write(C.PROC / "known_spot_index.npz", "known_spot_index.npz")
    for p in sorted(REL.glob(f"*_{args.tag}.zip")):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
