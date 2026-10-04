"""Segment a new BSE + Inlens image pair with the student model (GPU via PyTorch, or CPU via ONNX Runtime).

Input: the raw TIFFs as exported by the microscope (25 nm/px). Output: labels.png (uint8, 50 nm/px:
0 pore, 1 graphite, 2 SiOx, 3 CBD, 255 excluded), overlay.jpg and fractions.json in the output folder.

Usage:
  python -m src.predict --bse path/img_X_BSE.tif --inlens path/img_X_Inlens.tif --out outputs/predict/X
  python -m src.predict ... --cpu          # ONNX Runtime on CPU, no PyTorch/CUDA needed
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import config as C
from .preprocess import cu_foil, read_half
from .student import segment


def prepare(bse_path, inlens_path):
    raw = {"BSE": read_half(bse_path), "Inlens": read_half(inlens_path)}
    h, w = raw["BSE"].shape
    exclude = np.zeros((h, w), bool)
    b = C.BORDER_RAW_PX // C.DS
    exclude[:b], exclude[-b:], exclude[:, :b], exclude[:, -b:] = True, True, True, True
    exclude |= cu_foil(raw["BSE"])
    chans = []
    for d in ("BSE", "Inlens"):
        lo, hi = np.percentile(raw[d][~exclude], [0.5, 99.5])
        chans.append(ndi.median_filter(np.clip((raw[d] - lo) / (hi - lo), 0, 1), size=3).astype(np.float32))
    return np.stack(chans), exclude


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bse", required=True)
    ap.add_argument("--inlens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true", help="use models/student.onnx with ONNX Runtime on CPU")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x, exclude = prepare(args.bse, args.inlens)
    t = time.time()
    if args.cpu:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(C.MODELS / "student.onnx"), providers=["CPUExecutionProvider"])
        probs = segment(x, lambda xb: sess.run(None, {"x": xb})[0], batch=4)
    else:
        import torch
        from .student import CKPT, build_model, torch_runner
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(pretrained=False).to(dev)
        model.load_state_dict(torch.load(CKPT, map_location=dev))
        model.eval()
        probs = segment(x, torch_runner(model, dev))
    from .physics import constrain_siox
    probs = constrain_siox(probs, x[0], exclude, class_axis=0)
    seconds = time.time() - t
    from .teacher import clean
    lab = clean(probs.argmax(0), exclude)
    Image.fromarray(lab).save(out / "labels.png")
    Image.fromarray(C.overlay_rgb(x[0], lab, alpha=0.45)).save(out / "overlay.jpg", quality=88)
    valid = lab != C.EXCLUDED
    fr = {c: round(100 * float((lab[valid] == k).mean()), 2) for k, c in enumerate(C.CLASSES)}
    json.dump({"phase_fractions_pct": fr, "inference_seconds": round(seconds, 2), "nm_per_px": C.NM_PER_PX},
              open(out / "fractions.json", "w"), indent=2)
    print(fr, f"({seconds:.2f}s)")


if __name__ == "__main__":
    main()
