"""Batch-ID stage, step 2: per-sample features from the label maps, plus session (nuisance) probes.

Pre-registered classifier features (S*, fixed before any scoring; see docs/route_to_80 synthesis A2):
  siox_frac         SiOx area fraction (%)
  siox_ecd_aw       area-weighted mean equivalent diameter of SiOx particles (um)
  pore_open_excess  pore_all_frac - pore_deep_frac: open, grey-floored pores (%)
  pore_all_frac     all pores (%)
  pore_chord_aniso  median horizontal / median vertical pore chord; chords touching the border or an excluded
                    region are discarded (censoring fix)
Evidence-only features (reported, never fitted): pore_deep_frac, cbd_frac, cbd_share (CBD / (graphite + CBD)),
  graphite_frac, siox_count_density, pore_ps_r16, pore_large_frac, pore_count_density, pore_chord_H_p90,
  pore_chord_V_p90, pore_sv, pore_cv_25um, siox_pore_contact.
Each feature is also computed on the left and right halves; half_sd = |left - right| / sqrt(2).

Session probes (14, for the nuisance-only classifier; never used to predict batch): image height/width, BSE and
Inlens raw grey p1/p50/p99, BSE noise sigma, Inlens sharpness, Inlens horizontal-banding power, excluded
fraction, BSE TIFF file size.

Writes outputs/metrics/bid_features.csv and outputs/metrics/bid_session_probes.csv.

Usage: python -m src.bid_features [--labels final|teacher]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed

from . import config as C

Image.MAX_IMAGE_PIXELS = None
PX = C.NM_PER_PX / 1000            # um per pixel
S_FEATURES = ["siox_frac", "siox_ecd_aw", "pore_open_excess", "pore_all_frac", "pore_chord_aniso"]


def chords(mask, blocked, axis):
    """Run lengths of True along `axis`, discarding runs that touch a blocked pixel or the image edge."""
    m = np.moveaxis(mask, axis, 0)
    b = np.moveaxis(blocked, axis, 0)
    pad = np.pad(m.astype(np.int8), ((1, 1), (0, 0)))
    d = np.diff(pad, axis=0)
    s_r, s_c = np.nonzero(d == 1)
    e_r, e_c = np.nonzero(d == -1)
    order_s, order_e = np.lexsort((s_r, s_c)), np.lexsort((e_r, e_c))
    s_r, s_c, e_r = s_r[order_s], s_c[order_s], e_r[order_e]
    n = m.shape[0]
    bpad = np.pad(b, ((1, 1), (0, 0)), constant_values=True)
    ok = (s_r > 0) & (e_r < n) & ~bpad[s_r, s_c] & ~bpad[e_r + 1, s_c]
    return (e_r - s_r)[ok] * PX


def siox_particles(lab):
    mask = ndi.binary_fill_holes(lab == C.SIOX)
    dist = ndi.distance_transform_edt(mask)
    from skimage.morphology import h_maxima
    markers, _ = ndi.label(h_maxima(dist, 3))
    inst = watershed(-dist, markers, mask=mask)
    areas = np.bincount(inst.ravel())[1:] * PX ** 2
    return inst, areas[areas >= 0.1]


def features(lab, bse, t_low):
    valid = lab != C.EXCLUDED
    A = valid.sum()
    pore = (lab == C.PORE) & valid
    deep = pore & (bse < t_low)
    f = {}
    f["siox_frac"] = 100 * ((lab == C.SIOX) & valid).sum() / A
    f["pore_all_frac"] = 100 * pore.sum() / A
    f["pore_deep_frac"] = 100 * deep.sum() / A
    f["pore_open_excess"] = f["pore_all_frac"] - f["pore_deep_frac"]
    f["graphite_frac"] = 100 * ((lab == C.GRAPHITE) & valid).sum() / A
    f["cbd_frac"] = 100 * ((lab == C.CBD) & valid).sum() / A
    f["cbd_share"] = f["cbd_frac"] / max(f["graphite_frac"] + f["cbd_frac"], 1e-9)
    _, areas = siox_particles(lab)
    ecd = 2 * np.sqrt(areas / np.pi)
    f["siox_ecd_aw"] = float((ecd * areas).sum() / areas.sum()) if len(areas) else np.nan
    f["siox_count_density"] = 1000 * len(areas) / (A * PX ** 2)
    blocked = ~valid
    cH, cV = chords(pore, blocked, 1), chords(pore, blocked, 0)
    f["pore_chord_aniso"] = float(np.median(cH) / np.median(cV)) if len(cH) and len(cV) else np.nan
    f["pore_chord_H_p90"] = float(np.percentile(cH, 90)) if len(cH) else np.nan
    f["pore_chord_V_p90"] = float(np.percentile(cV, 90)) if len(cV) else np.nan
    f["pore_ps_r16"] = ndi.binary_opening(pore, structure=disk(16)).sum() / max(pore.sum(), 1)
    cc, n = ndi.label(pore)
    pa = np.bincount(cc.ravel())[1:] * PX ** 2
    f["pore_large_frac"] = pa[pa >= 20].sum() / max(pa.sum(), 1e-9)
    f["pore_count_density"] = 1000 * (pa >= 0.25).sum() / (A * PX ** 2)
    edges = (pore[:, 1:] != pore[:, :-1])[valid[:, 1:] & valid[:, :-1]].sum() + (pore[1:] != pore[:-1])[valid[1:] & valid[:-1]].sum()
    f["pore_sv"] = edges * PX / (A * PX ** 2)
    win = int(25 / PX)
    fr = [pore[y:y + win, x:x + win].mean() for y in range(0, lab.shape[0] - win + 1, win)
          for x in range(0, lab.shape[1] - win + 1, win) if valid[y:y + win, x:x + win].all()]
    f["pore_cv_25um"] = float(np.std(fr) / np.mean(fr)) if len(fr) > 1 and np.mean(fr) > 0 else np.nan
    near_pore = ndi.binary_dilation(pore, iterations=2)
    si = (lab == C.SIOX)
    rim = ndi.binary_dilation(si, iterations=2) & ~si
    f["siox_pore_contact"] = (rim & near_pore).sum() / max(rim.sum(), 1)
    return f


def session_probes(sid, paths, lab):
    bse_raw = np.asarray(Image.open(paths["BSE"]))[..., 1].astype(np.float32)
    inl_raw = np.asarray(Image.open(paths["Inlens"]))[..., 1].astype(np.float32)
    p = {"height_px": bse_raw.shape[0], "width_px": bse_raw.shape[1]}
    for name, a in (("bse", bse_raw), ("inlens", inl_raw)):
        p[f"{name}_p1"], p[f"{name}_p50"], p[f"{name}_p99"] = np.percentile(a, [1, 50, 99]).tolist()
    p["bse_noise_sigma"] = float(np.median(np.abs(bse_raw[:, 1:] - bse_raw[:, :-1])) / 0.954)
    lap = ndi.laplace(ndi.gaussian_filter(inl_raw, 1))
    p["inlens_sharpness"] = float(np.var(lap))
    rows = inl_raw.mean(1)
    p["inlens_banding_power"] = float(np.var(rows - ndi.uniform_filter1d(rows, 51)))
    p["excluded_frac"] = float((lab == C.EXCLUDED).mean())
    p["bse_file_mb"] = os.path.getsize(paths["BSE"]) / 1e6
    return p


def run(args):
    (batch, sid), paths, labels_name, t_low = args
    lab = np.load(C.proc_dir(sid) / f"{labels_name}_labels.npy")
    bse = C.load_channels(sid, ["BSE"])["BSE"]
    full = features(lab, bse, t_low)
    w = lab.shape[1] // 2
    left, right = features(lab[:, :w], bse[:, :w], t_low), features(lab[:, w:], bse[:, w:], t_low)
    row = {"batch": batch, "sample_id": sid, **{k: round(float(v), 5) for k, v in full.items()}}
    row.update({f"{k}_half_sd": round(float(abs(left[k] - right[k]) / np.sqrt(2)), 5) for k in full})
    probes = {"batch": batch, "sample_id": sid, **{k: round(float(v), 5) for k, v in session_probes(sid, paths, lab).items()}}
    return row, probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="final", choices=["final", "teacher", "student"])
    args = ap.parse_args()
    t_low = {r["sample_id"]: float(r["bse_t_low"]) for r in C.read_csv(C.OUT_MET / "seed_coverage.csv")}
    jobs = [(k, p, args.labels, t_low[k[1]]) for k, p in C.discover().items()]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        res = list(ex.map(run, jobs))
    C.write_csv(C.OUT_MET / "bid_features.csv", [r for r, _ in res])
    C.write_csv(C.OUT_MET / "bid_session_probes.csv", [p for _, p in res])
    print(f"features for {len(res)} samples from {args.labels} labels")
    for k in S_FEATURES + ["pore_deep_frac", "cbd_share"]:
        vals = {b: [r[k] for r, _ in res if r["batch"] == b] for b in ("Batch_1", "Batch_2", "Batch_3")}
        print(f"  {k:18s} " + "  ".join(f"{b[-1]}: {np.nanmean(v):7.3f}±{np.nanstd(v, ddof=1):6.3f}" for b, v in vals.items()))


if __name__ == "__main__":
    main()
