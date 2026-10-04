"""Robustness check for the one Batch_1 vs Batch_2 lead, ETD graphite-surface roughness.

Is it surface texture, or detector noise / normalisation? For each sample, on the raw ETD (full resolution,
un-normalised): pixel-noise sigma (median absolute difference of horizontal neighbours / 0.954, i.e. white noise),
graphite local SD at several scales (sigma 1, 2, 4, 8 full-res px), the same after subtracting the noise variance,
and the normalisation range used for the normalised KPI. Batch_1 vs Batch_2: Cliff's delta, Mann-Whitney p and the
three same-session pairs.

Usage: python -m src.roughness_check
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from skimage.morphology import disk

from . import config as C

Image.MAX_IMAGE_PIXELS = None
PAIRS = [("f1vzngrs", "epqdaau9", "h2148"), ("fzrt2k6r", "b3esycq1", "h2156"), ("ffwubibz", "r17byphk", "h2080")]


def one(item):
    (batch, sid), paths = item
    raw = np.asarray(Image.open(paths["SE2"]))[..., 1].astype(np.float32)
    lab = np.load(C.proc_dir(sid) / "final_labels.npy")
    lab_full = np.kron(lab, np.ones((2, 2), np.uint8))[:raw.shape[0] // 2 * 2, :raw.shape[1] // 2 * 2]
    raw = raw[:lab_full.shape[0], :lab_full.shape[1]]
    g = ndi.binary_erosion(lab_full == C.GRAPHITE, structure=disk(10))
    noise = float(np.median(np.abs(raw[:, 1:] - raw[:, :-1])[g[:, 1:]]) / 0.954)
    f = {"batch": batch, "sample_id": sid, "etd_raw_noise_sigma": noise,
         "etd_raw_graphite_median": float(np.median(raw[g])),
         "etd_norm_range": float(np.subtract(*np.percentile(raw[lab_full != C.EXCLUDED], [99.5, 0.5])))}
    for s in (1, 2, 4, 8):
        m = ndi.gaussian_filter(raw, s)
        sd = np.sqrt(np.maximum(ndi.gaussian_filter(raw * raw, s) - m * m, 0))
        f[f"etd_raw_rough_s{s}"] = float(np.median(sd[g]))
        # texture after removing the white-noise contribution (noise variance inside a Gaussian window ~ sigma^2 * (1 - 1/(4 pi s^2)))
        f[f"etd_raw_rough_s{s}_denoised"] = float(np.sqrt(max(np.median(sd[g]) ** 2 - noise ** 2 * (1 - 1 / (4 * np.pi * s * s)), 0)))
    return f


def main():
    with ProcessPoolExecutor(C.WORKERS) as ex:
        rows = list(ex.map(one, C.discover().items()))
    C.write_csv(C.OUT_MET / "roughness_check.csv", rows)
    norm = {r["sample_id"]: float(r["etd_graphite_roughness"]) for r in C.read_csv(C.OUT_MET / "kpi_topo_values.csv")}
    for r in rows:
        r["etd_graphite_roughness_normalised"] = norm[r["sample_id"]]
    by = {r["sample_id"]: r for r in rows}
    b1 = [r for r in rows if r["batch"] == "Batch_1"]
    b2 = [r for r in rows if r["batch"] == "Batch_2"]
    keys = ["etd_graphite_roughness_normalised", "etd_raw_noise_sigma", "etd_norm_range", "etd_raw_graphite_median"] + \
           [f"etd_raw_rough_s{s}{d}" for s in (1, 2, 4, 8) for d in ("", "_denoised")]
    print(f"{'measure':36s} {'B1':>8s} {'B2':>8s} {'delta':>6s} {'p':>7s} pairs(B1>B2)")
    for k in keys:
        a, b = np.array([r[k] for r in b1]), np.array([r[k] for r in b2])
        d = ((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / 49
        pairs = sum(by[x][k] > by[y][k] for x, y, _ in PAIRS)
        print(f"{k:36s} {a.mean():8.3f} {b.mean():8.3f} {d:6.2f} {stats.mannwhitneyu(a, b).pvalue:7.4f} {pairs}/3")
    v = lambda k: [r[k] for r in rows]
    print("\nSpearman over all 31 samples with the normalised roughness:")
    for k in ("etd_raw_noise_sigma", "etd_norm_range", "etd_raw_rough_s2", "etd_raw_rough_s2_denoised", "etd_raw_rough_s8_denoised"):
        print(f"  {k:30s} rho = {stats.spearmanr(v('etd_graphite_roughness_normalised'), v(k)).statistic:+.2f}")


if __name__ == "__main__":
    main()
