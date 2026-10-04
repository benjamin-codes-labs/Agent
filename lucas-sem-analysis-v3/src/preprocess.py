"""Stage 1: read, downsample, exclude, normalise, denoise.

Per sample, writes data/processed/<sid>/:
  BSE.npy, Inlens.npy, SE2.npy   float32 in [0, 1], half resolution (50 nm/px)
  exclude.npy                    bool, True = not analysed (border, Cu foil, unpolished regions)
  meta.json                      shape, detector used for SE2, normalisation percentiles, excluded fraction

Usage: python -m src.preprocess
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import config as C

Image.MAX_IMAGE_PIXELS = None
EXCLUSIONS = json.load(open(C.ROOT / "data" / "exclusions.json"))


def read_half(path):
    """Green channel (the R/G/B channels are identical except a few edge columns), 2x block mean."""
    a = np.asarray(Image.open(path))[..., 1].astype(np.float32)
    h, w = a.shape[0] // C.DS * C.DS, a.shape[1] // C.DS * C.DS
    return a[:h, :w].reshape(h // C.DS, C.DS, w // C.DS, C.DS).mean((1, 3))


def cu_foil(bse_raw_half):
    """Saturated BSE regions (Cu current collector) connected to the top or bottom edge."""
    sat = bse_raw_half >= 240
    lab, n = ndi.label(sat)
    if not n:
        return np.zeros_like(sat)
    edge_ids = set(np.unique(lab[0])) | set(np.unique(lab[-1]))
    sizes = np.bincount(lab.ravel())
    keep = [i for i in edge_ids if i and sizes[i] >= 500]
    mask = np.isin(lab, keep)
    return ndi.binary_dilation(mask, iterations=20) if keep else mask


def process(item):
    (batch, sid), paths = item
    raw = {d: read_half(paths[d]) for d in C.DETECTORS}
    h, w = raw["BSE"].shape
    exclude = np.zeros((h, w), bool)
    b = C.BORDER_RAW_PX // C.DS
    exclude[:b], exclude[-b:], exclude[:, :b], exclude[:, -b:] = True, True, True, True
    cu = cu_foil(raw["BSE"])
    exclude |= cu
    for y0, y1, x0, x1 in EXCLUSIONS.get(sid, {}).get("boxes_halfres", []):
        exclude[y0:y1, x0:x1] = True

    d = C.proc_dir(sid)
    meta = {"batch": batch, "sample_id": sid, "shape": [h, w],
            "se2_detector": "ETD" if "ETD" in paths else "SE",
            "cu_foil_px": int(cu.sum()), "excluded_frac": round(float(exclude.mean()), 4), "percentiles": {}}
    valid = ~exclude
    for det, a in raw.items():
        lo, hi = np.percentile(a[valid], [0.5, 99.5])
        n = np.clip((a - lo) / (hi - lo), 0, 1)
        n = ndi.median_filter(n, size=3).astype(np.float32)
        np.save(d / f"{det}.npy", n)
        meta["percentiles"][det] = [round(float(lo), 2), round(float(hi), 2)]
    np.save(d / "exclude.npy", exclude)
    C.write_json(d / "meta.json", meta)
    return meta


def main():
    with ProcessPoolExecutor(C.WORKERS) as ex:
        metas = list(ex.map(process, C.discover().items()))
    for m in metas:
        flag = []
        if m["cu_foil_px"]:
            flag.append(f"Cu foil {m['cu_foil_px']} px")
        if m["sample_id"] in EXCLUSIONS:
            flag.append("manual exclusion")
        print(f"{m['batch']}/{m['sample_id']}: {m['shape']} SE2={m['se2_detector']} excluded {100 * m['excluded_frac']:.2f}% {' | '.join(flag)}")


if __name__ == "__main__":
    main()
