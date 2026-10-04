"""Exploratory: ETD/SE topography KPIs and grain-contrast KPIs, focused on Batch_1 vs Batch_2.

Topography (SE2 = ETD, or SE for 4 samples; per-image normalised, so mostly ratios and shapes):
  gradient energy at sigma 1/2/4 px, ETD / BSE gradient-energy ratio (topography vs composition), shading
  direction and strength (mean signed gradient), curtaining-streak amplitude (column-profile fluctuation),
  polished-graphite roughness (local SD inside eroded graphite), relief inside pores (SD and mean of ETD inside
  pores relative to graphite), edge brightening (ETD in a 2-px band on graphite edges vs graphite interior),
  ETD-Inlens and ETD-BSE pixel correlation, ETD saturation fraction (raw >= 250).
Grain contrast (from the delivered label map): coefficient of variation of the per-flake median Inlens and BSE
  (graphite channelling / orientation contrast), and SiOx particle size / shape quantiles.

Writes outputs/metrics/kpi_topo_values.csv.

Usage: python -m src.kpi_topo
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk

from . import config as C

Image.MAX_IMAGE_PIXELS = None


def grad_energy(a, s):
    return float(np.mean(ndi.gaussian_gradient_magnitude(a, s) ** 2))


def topo(item):
    (batch, sid), paths = item
    ch = C.load_channels(sid)
    bse, inl, se2 = ch["BSE"], ch["Inlens"], ch["SE2"]
    lab = np.load(C.proc_dir(sid) / "final_labels.npy")
    valid = lab != C.EXCLUDED
    graphite = lab == C.GRAPHITE
    g_in = ndi.binary_erosion(graphite, structure=disk(4))
    pore = lab == C.PORE
    f = {"se2_detector_is_SE": float("ETD" not in paths)}
    for s in (1, 2, 4):
        f[f"etd_grad_energy_s{s}"] = grad_energy(np.where(valid, se2, np.median(se2)), s)
    f["etd_over_bse_grad_ratio_s2"] = f["etd_grad_energy_s2"] / max(grad_energy(np.where(valid, bse, np.median(bse)), 2), 1e-9)
    gy, gx = ndi.gaussian_filter(se2, 2, order=(1, 0)), ndi.gaussian_filter(se2, 2, order=(0, 1))
    mag = np.sqrt(gx ** 2 + gy ** 2)[valid].mean()
    f["etd_shading_x"] = float(gx[valid].mean() / max(mag, 1e-9))
    f["etd_shading_y"] = float(gy[valid].mean() / max(mag, 1e-9))
    f["etd_shading_strength"] = float(np.hypot(f["etd_shading_x"], f["etd_shading_y"]))
    # curtaining: vertical streaks -> fluctuation of the column-mean profile after removing slow trends
    cols = np.where(g_in, se2, np.nan)
    prof = np.nanmean(cols, 0)
    prof = np.where(np.isfinite(prof), prof, np.nanmean(prof))
    f["etd_curtaining_amp"] = float(np.std(prof - ndi.uniform_filter1d(prof, 41)))
    prof_b = np.nanmean(np.where(g_in, bse, np.nan), 0)
    prof_b = np.where(np.isfinite(prof_b), prof_b, np.nanmean(prof_b))
    f["bse_curtaining_amp"] = float(np.std(prof_b - ndi.uniform_filter1d(prof_b, 41)))
    loc_sd = np.sqrt(np.maximum(ndi.gaussian_filter(se2 ** 2, 2) - ndi.gaussian_filter(se2, 2) ** 2, 0))
    f["etd_graphite_roughness"] = float(np.median(loc_sd[g_in]))
    loc_sd_i = np.sqrt(np.maximum(ndi.gaussian_filter(inl ** 2, 2) - ndi.gaussian_filter(inl, 2) ** 2, 0))
    f["inlens_graphite_roughness"] = float(np.median(loc_sd_i[g_in]))
    g_med = np.median(se2[g_in])
    f["etd_pore_relief_sd"] = float(np.std(se2[pore]))
    f["etd_pore_vs_graphite"] = float((np.median(se2[pore]) - g_med) / max(np.std(se2[g_in]), 1e-6))
    edge = graphite & ~ndi.binary_erosion(graphite, structure=disk(2))
    f["etd_edge_brightening"] = float((np.median(se2[edge]) - g_med) / max(np.std(se2[g_in]), 1e-6))
    v = valid
    f["corr_etd_inlens"] = float(np.corrcoef(se2[v], inl[v])[0, 1])
    f["corr_etd_bse"] = float(np.corrcoef(se2[v], bse[v])[0, 1])
    raw = np.asarray(Image.open(paths["SE2"]))[..., 1]
    f["etd_saturated_frac"] = float((raw >= 250).mean())
    # graphite grain (flake) contrast: CV of per-flake medians
    lab_g, n = ndi.label(g_in)
    if n > 5:
        ids = np.arange(1, n + 1)
        size = ndi.sum(np.ones_like(lab_g), lab_g, ids)
        big = ids[size >= int(4 / C.PX_UM2)]                      # flakes >= 4 um^2
        for name, a in (("inlens", inl), ("bse", bse), ("etd", se2)):
            med = np.array(ndi.median(a, lab_g, big))
            f[f"{name}_flake_contrast_cv"] = float(np.std(med) / max(np.mean(med), 1e-6))
        f["graphite_flake_count_ge4um2"] = float(len(big))
    return sid, batch, f


def main():
    with ProcessPoolExecutor(C.WORKERS) as ex:
        res = list(ex.map(topo, C.discover().items()))
    ps = C.read_csv(C.OUT_MET / "particle_sizes.csv")
    rows = []
    for sid, batch, f in res:
        d = np.array([float(r["eq_diam_um"]) for r in ps if r["sample_id"] == sid])
        ar = np.array([float(r["aspect_ratio"]) for r in ps if r["sample_id"] == sid])
        f.update({"siox_ecd_p25_um": float(np.percentile(d, 25)), "siox_ecd_p75_um": float(np.percentile(d, 75)),
                  "siox_ecd_p95_um": float(np.percentile(d, 95)), "siox_aspect_median": float(np.median(ar))})
        rows.append({"batch": batch, "sample_id": sid, **{k: round(v, 6) for k, v in f.items()}})
    C.write_csv(C.OUT_MET / "kpi_topo_values.csv", rows)
    print(f"{len(rows[0]) - 2} topography / grain KPIs for {len(rows)} samples")


if __name__ == "__main__":
    main()
