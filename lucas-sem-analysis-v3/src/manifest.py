"""Stage 0: inventory and QC of the raw data.

Writes
  data/manifest.csv         one row per TIFF (batch, sample_id, detector, path, size, pixel size, sha256, channel check)
  data/qc.csv               one row per sample (height group, BSE<->Inlens / BSE<->SE2 registration shifts)
  data/baseline/baseline.csv  earlier per-sample measurements (thr = BSE threshold, vis = vision-LLM estimate)

Usage: python -m src.manifest
"""
import glob
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

from . import config as C

Image.MAX_IMAGE_PIXELS = None


def file_row(args):
    (batch, sid), det, path = args
    raw = path.read_bytes()
    im = Image.open(path)
    xres = im.tag_v2.get(282)
    unit = im.tag_v2.get(296, 2)
    dpi = float(xres)
    nm = 25.4e6 / dpi if unit == 2 else 1e7 / dpi
    a = np.asarray(im)
    diff_cols = np.flatnonzero((a[..., 0] != a[..., 1]).any(0) | (a[..., 2] != a[..., 1]).any(0))
    return {
        "batch": batch, "sample_id": sid, "detector": det,
        "path": path.relative_to(C.ROOT).as_posix(),
        "width": im.width, "height": im.height, "mode": im.mode,
        "nm_per_px": round(nm, 4),
        "rgb_mismatch_columns": " ".join(map(str, diff_cols[:10])) + (" ..." if len(diff_cols) > 10 else ""),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def registration(args):
    (batch, sid), dets = args

    def grey_half(p):
        a = np.asarray(Image.open(p))[..., 1].astype(np.float32)
        h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))

    bse = grey_half(dets["BSE"])
    out = {"batch": batch, "sample_id": sid, "se2_detector": "ETD" if "ETD" in dets else "SE",
           "height_px": Image.open(dets["BSE"]).height}
    for det in ("Inlens", "SE2"):
        other = grey_half(dets[det])
        shift, _, _ = phase_cross_correlation(bse, other, upsample_factor=10, normalization=None)
        out[f"shift_BSE_{det}_y"] = round(float(shift[0]) * C.DS, 2)   # in full-resolution pixels
        out[f"shift_BSE_{det}_x"] = round(float(shift[1]) * C.DS, 2)
    return out


def build_baseline():
    """Collect the earlier per-sample numbers into one CSV used for validation."""
    rows = []
    si = re.compile(r"silicon|\bsio|\bsi\b", re.I)
    for f in sorted(glob.glob(str(C.ROOT / "prior_analysis" / "llm_run" / "samples" / "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        a, m = r["analysis"], r["metrics"]
        phase = next(p for p in a["phases"] if si.search(p["name"]))
        rows.append({"batch": r["batch"], "sample_id": r["sample_id"],
                     "pore_thr": m["pore_area_pct"], "pore_vis": a["porosity"]["estimated_percent"],
                     "siox_thr": m["bright_particle_area_pct"], "siox_vis": phase["estimated_area_percent"]})
    C.write_csv(C.BASELINE, rows)
    return rows


def main():
    samples = C.discover()
    jobs = [(key, det, p) for key, dets in samples.items() for det, p in dets.items() if det != "SE2"]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        rows = list(ex.map(file_row, jobs))
        qc = list(ex.map(registration, samples.items()))
    rows.sort(key=lambda r: (r["batch"], r["sample_id"], r["detector"]))
    C.write_csv(C.MANIFEST, rows)

    groups = defaultdict(list)
    for q in qc:
        groups[q["height_px"]].append(q)
    for h, qs in groups.items():
        for q in qs:
            q["height_group"] = f"h{h}"
            q["height_group_batches"] = "+".join(sorted({x["batch"][-1] for x in qs}))
    C.write_csv(C.QC, qc)
    base = build_baseline()

    # ---- summary
    n_files = len(rows)
    dets = defaultdict(int)
    for r in rows:
        dets[r["detector"]] += 1
    print(f"{n_files} TIFFs, {len(samples)} samples, detectors {dict(dets)}")
    print("batches:", {b: sum(1 for (bb, _) in samples if bb == b) for b in sorted({b for b, _ in samples})})
    print("pixel size nm:", sorted({r["nm_per_px"] for r in rows}))
    print("widths:", sorted({r["width"] for r in rows}), "heights:", sorted({r["height"] for r in rows}))
    mism = [f"{r['sample_id']}/{r['detector']}:[{r['rgb_mismatch_columns']}]" for r in rows if r["rgb_mismatch_columns"]]
    print(f"files with R!=G columns: {len(mism)}", mism[:6])
    shifts = np.array([[q[k] for k in q if k.startswith("shift_")] for q in qc])
    print(f"registration shifts (full-res px): mean {shifts.mean(0).round(2).tolist()}, max |shift| {np.abs(shifts).max():.2f}")
    print("height groups spanning >1 batch:")
    for h, qs in sorted(groups.items()):
        if len({q['batch'] for q in qs}) > 1 or len(qs) > 1:
            print(f"  h{h}: " + ", ".join(f"{q['sample_id']}/B{q['batch'][-1]}" for q in qs))
    print(f"baseline rows: {len(base)}")


if __name__ == "__main__":
    main()
