"""Paths, constants and small I/O helpers shared by every stage."""
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Raw TIFFs: SEM_RAW_DIR if set, else the hackathon repo's shared data/raw (batch_1..batch_3) when this
# folder sits inside it, else this project's own data/raw (Batch_1..Batch_3).
_SHARED_RAW = ROOT.parent / "data" / "raw"
RAW = Path(os.environ.get("SEM_RAW_DIR") or (_SHARED_RAW if _SHARED_RAW.is_dir() else ROOT / "data" / "raw"))
# Worker processes for the parallel stages; lower it (e.g. SEM_WORKERS=3) on machines with little RAM.
WORKERS = int(os.environ.get("SEM_WORKERS", "8"))
PROC = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifest.csv"
QC = ROOT / "data" / "qc.csv"
BASELINE = ROOT / "data" / "baseline" / "baseline.csv"
MODELS = ROOT / "models"
OUT_SEG = ROOT / "outputs" / "segmentation"
OUT_MET = ROOT / "outputs" / "metrics"
REPORTS = ROOT / "reports"

NM_PER_PX_RAW = 25.0
DS = 2                                  # block-mean downsampling factor
NM_PER_PX = NM_PER_PX_RAW * DS          # 50 nm/px working resolution
PX_UM2 = (NM_PER_PX / 1000) ** 2        # um^2 per working pixel
BORDER_RAW_PX = 8                       # excluded border at full resolution

CLASSES = ["pore", "graphite", "SiOx", "CBD"]
PORE, GRAPHITE, SIOX, CBD = range(4)
EXCLUDED = 255
COLOURS = np.array([[30, 110, 255], [170, 110, 255], [255, 150, 0], [40, 210, 120]], dtype=np.float32)

DETECTORS = ["BSE", "Inlens", "SE2"]    # SE2 = ETD, or SE in the 4 samples without ETD

for d in (PROC, MODELS, OUT_SEG, OUT_MET, REPORTS):
    d.mkdir(parents=True, exist_ok=True)


def discover():
    """{(batch, sample_id): {detector: path}} from RAW, with ETD/SE mapped to SE2. Batch folders may be
    named Batch_N or batch_N; the batch is always reported as Batch_N."""
    samples = {}
    for p in sorted(RAW.glob("[Bb]atch_*/img_*_*.tif")):
        m = re.match(r"img_([a-z0-9]+)_([A-Za-z]+)\.tif$", p.name)
        if not m:
            continue
        det = m.group(2)
        batch = "Batch_" + p.parent.name.split("_", 1)[1]
        samples.setdefault((batch, m.group(1)), {})[det] = p
    for dets in samples.values():
        dets["SE2"] = dets.get("ETD") or dets.get("SE")
    return dict(sorted(samples.items()))


def sample_ids():
    return [(b, s) for (b, s) in discover()]


def proc_dir(sid):
    d = PROC / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_channels(sid, dets=DETECTORS):
    return {d: np.load(proc_dir(sid) / f"{d}.npy") for d in dets}


def load_exclude(sid):
    return np.load(proc_dir(sid) / "exclude.npy")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def baseline():
    """{sample_id: {pore_thr, pore_vis, siox_thr, siox_vis}} as floats."""
    return {r["sample_id"]: {k: float(v) for k, v in r.items() if k not in ("batch", "sample_id")}
            for r in read_csv(BASELINE)}


def overlay_rgb(grey, labels, alpha=0.5):
    """Colour a label map over a 0-1 greyscale image; excluded pixels are tinted red."""
    g = np.repeat((np.clip(grey, 0, 1) * 255)[..., None], 3, -1).astype(np.float32)
    rgb = g.copy()
    for k in range(len(CLASSES)):
        m = labels == k
        rgb[m] = g[m] * (1 - alpha) + COLOURS[k] * alpha
    m = labels == EXCLUDED
    rgb[m] = g[m] * 0.4 + np.array([200, 0, 0]) * 0.6
    return rgb.clip(0, 255).astype(np.uint8)
