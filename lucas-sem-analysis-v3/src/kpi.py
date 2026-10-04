"""Exploratory non-ML KPIs: thresholds, shapes, contrast, orientation, spectra, heterogeneity.

Everything here is a plain image-processing function of the normalised half-resolution BSE / Inlens images
(50 nm/px; per-image normalisation is linear, so intensity RATIOS are unaffected by it). No trained model and no
label map is used: pores = BSE below the lower multi-Otsu threshold, SiOx = BSE above the upper one.

For each KPI: Kruskal-Wallis across batches, Batch_1 vs Batch_2 Mann-Whitney + Cliff's delta, Batch_3 vs rest
Cliff's delta, a session-stratified permutation p (batch labels permuted only within image-height groups),
Benjamini-Hochberg q across all KPIs, the largest |Spearman rho| with the acquisition (session) probes, and the
|Spearman rho| with the segmentation's pore and SiOx fractions (redundancy).

Exploratory classifier K: in each leave-one-out fold the 5 KPIs with the largest Kruskal-Wallis H on the
training fold are selected, robust-z scaled and fed to shrinkage LDA; its permutation null repeats the
selection inside every shuffle. Logged to the look ledger as exploratory.

Writes outputs/metrics/kpi_values.csv, kpi_stats.csv, kpi_classifier.json.

Usage: python -m src.kpi
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree
from skimage.feature import canny
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops, label as cc_label
from skimage.morphology import disk, remove_small_objects
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C

PX = C.NM_PER_PX / 1000
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
PROBES = ["height_px", "bse_noise_sigma", "inlens_sharpness", "inlens_banding_power", "bse_file_mb",
          "bse_p50", "inlens_p50"]


def shape_stats(mask, prefix, area_um2):
    lab = cc_label(mask)
    props = [p for p in regionprops(lab) if p.area * PX ** 2 >= 0.1]
    f = {}
    if not props:
        return f
    a = np.array([p.area for p in props]) * PX ** 2
    ecd = 2 * np.sqrt(a / np.pi)
    maj = np.array([p.major_axis_length for p in props])
    mnr = np.maximum(np.array([p.minor_axis_length for p in props]), 1)
    theta = np.array([p.orientation for p in props])              # radians from the row axis (vertical)
    per = np.maximum(np.array([p.perimeter for p in props]), 1)
    f[f"{prefix}_n_per_1000um2"] = 1000 * len(a) / area_um2
    f[f"{prefix}_median_ecd_um"] = float(np.median(ecd))
    f[f"{prefix}_p90_ecd_um"] = float(np.percentile(ecd, 90))
    f[f"{prefix}_aspect_aw"] = float(np.average(maj / mnr, weights=a))
    f[f"{prefix}_solidity_aw"] = float(np.average([p.solidity for p in props], weights=a))
    f[f"{prefix}_circularity_med"] = float(np.median(4 * np.pi * np.array([p.area for p in props]) / per ** 2))
    # orientation order vs the horizontal (in-plane) axis: +1 all horizontal, -1 all vertical
    # (regionprops orientation = angle of the major axis from the row axis; weights favour large, elongated objects)
    f[f"{prefix}_horizontal_order"] = float(np.average(-np.cos(2 * theta), weights=a * (maj / mnr - 1 + 1e-6)))
    cents = np.array([p.centroid for p in props]) * PX
    if len(cents) > 3:
        d, _ = cKDTree(cents).query(cents, k=2)
        r_obs = d[:, 1].mean()
        r_exp = 0.5 / np.sqrt(len(cents) / area_um2)
        f[f"{prefix}_clark_evans"] = float(r_obs / r_exp)            # <1 clustered, >1 dispersed
    return f


def box_count_dim(edge):
    sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for s in sizes:
        h, w = edge.shape[0] // s * s, edge.shape[1] // s * s
        counts.append(edge[:h, :w].reshape(h // s, s, w // s, s).any((1, 3)).sum())
    return float(-np.polyfit(np.log(sizes), np.log(np.maximum(counts, 1)), 1)[0])


def kpis(sid):
    ch = C.load_channels(sid, ["BSE", "Inlens"])
    ex = C.load_exclude(sid)
    bse, inl = ch["BSE"], ch["Inlens"]
    valid = ~ex
    area = valid.sum() * PX ** 2
    t_low, t_high = threshold_multiotsu(bse[valid], classes=3)
    dark = (bse < t_low) & valid
    bright = remove_small_objects(ndi.binary_opening((bse > t_high) & valid, structure=disk(2)), max_size=int(0.5 / PX ** 2))
    mid = (bse >= t_low) & (bse <= t_high) & valid
    f = {}
    # thresholds
    f["thr_dark_frac"] = 100 * dark.sum() / valid.sum()
    f["thr_dark_frac_fixed020"] = 100 * ((bse < 0.20) & valid).sum() / valid.sum()
    f["thr_bright_frac"] = 100 * bright.sum() / valid.sum()
    # contrast and grey-level shape (ratios are invariant to the linear per-image normalisation)
    m_dark, m_mid, m_bright = np.median(bse[dark]), np.median(bse[mid]), np.median(bse[bright]) if bright.any() else np.nan
    f["bse_siox_contrast_ratio"] = float((m_bright - m_mid) / max(m_mid - m_dark, 1e-6))
    f["bse_mid_skew"] = float(stats.skew(bse[mid]))
    f["bse_mid_kurtosis"] = float(stats.kurtosis(bse[mid]))
    hist = np.histogram(bse[valid], 64, (0, 1))[0] / valid.sum()
    f["bse_entropy"] = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    f["bse_edge_density"] = canny(bse, sigma=2)[valid].sum() / area
    f["inlens_edge_density"] = canny(inl, sigma=2)[valid].sum() / area
    # pore (dark) and SiOx (bright) shapes from thresholding
    f.update(shape_stats(dark, "dark", area))
    f.update(shape_stats(bright, "bright", area))
    f["dark_boundary_fractal_dim"] = box_count_dim(dark ^ ndi.binary_erosion(dark))
    rl = []                                                           # median pore chord, horizontal then vertical
    for axis in (1, 0):
        m = np.moveaxis(dark, axis, 0).astype(np.int8)
        d = np.diff(np.pad(m, ((1, 1), (0, 0))), axis=0)
        s = np.argwhere(d == 1); e = np.argwhere(d == -1)
        s = s[np.lexsort((s[:, 0], s[:, 1]))]; e = e[np.lexsort((e[:, 0], e[:, 1]))]
        runs = e[:, 0] - s[:, 0]
        runs = runs[runs >= 3]                                        # ignore 1-2 px noise runs
        rl.append(np.median(runs * PX) if len(runs) else np.nan)
    f["dark_chord_H_over_V"] = float(rl[0] / rl[1])
    # flake orientation from the BSE structure tensor (in-plane = horizontal)
    gy, gx = ndi.gaussian_filter(bse, 2, order=(1, 0)), ndi.gaussian_filter(bse, 2, order=(0, 1))
    jxx, jyy, jxy = (ndi.gaussian_filter(v, 8) for v in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(jxx + jyy, 1e-9)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)                       # gradient direction; flakes are normal to it
    w = coh * mid
    f["flake_coherence_mean"] = float(coh[mid].mean())
    f["flake_order_parameter"] = float(np.abs((w * np.exp(2j * ang)).sum()) / w.sum())
    flake_dir = np.degrees(np.abs(ang)) - 90                        # 0 = horizontal flake
    f["flake_tilted_frac"] = float((w * (np.abs(flake_dir) > 30)).sum() / w.sum())
    # spectral slope and anisotropy (BSE, valid area median-filled)
    img = np.where(valid, bse, np.median(bse[valid]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    hh, ww = P.shape
    fy = np.fft.fftshift(np.fft.fftfreq(hh, PX))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(ww, PX))[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    band = (fr > 1 / 10) & (fr < 1 / 0.5)
    bins = np.logspace(np.log10(1 / 10), np.log10(1 / 0.5), 16)
    idx = np.digitize(fr[band], bins)
    radial = np.array([P[band][idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()])
    centers = np.array([np.sqrt(bins[i - 1] * bins[i]) for i in range(1, len(bins)) if (idx == i).any()])
    f["psd_slope"] = float(np.polyfit(np.log(centers), np.log(radial), 1)[0])
    mid_band = (fr > 1 / 10) & (fr < 1 / 1)
    vert_energy = P[mid_band & (np.abs(fy) > np.abs(fx) * 2)].sum()   # variation along y = horizontal layering
    horz_energy = P[mid_band & (np.abs(fx) > np.abs(fy) * 2)].sum()
    f["psd_anisotropy"] = float(vert_energy / max(horz_energy, 1e-9))
    # heterogeneity
    def window_cv(mask, size_um):
        s = int(size_um / PX)
        v = [mask[y:y + s, x:x + s].mean() for y in range(0, mask.shape[0] - s + 1, s)
             for x in range(0, mask.shape[1] - s + 1, s) if valid[y:y + s, x:x + s].all()]
        return float(np.std(v) / np.mean(v)) if len(v) > 2 and np.mean(v) > 0 else np.nan
    f["dark_cv_10um"] = window_cv(dark, 10)
    f["bright_cv_25um"] = window_cv(bright, 25)
    h2 = valid.shape[0] // 2
    f["dark_top_over_bottom"] = float(dark[:h2].sum() / max(valid[:h2].sum(), 1) / max(dark[h2:].sum() / max(valid[h2:].sum(), 1), 1e-9))
    s = int(5 / PX)
    boxes = [dark[y:y + s, x:x + s].sum() for y in range(0, dark.shape[0] - s + 1, s // 2)
             for x in range(0, dark.shape[1] - s + 1, s // 2)]
    boxes = np.array(boxes, float)
    f["dark_lacunarity_5um"] = float(boxes.var() / max(boxes.mean() ** 2, 1e-9) + 1)
    return sid, {k: float(v) for k, v in f.items()}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / (len(a) * len(b)))


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty(len(p))
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * len(p) / rank)
        q[i] = prev
    return q


def kw_h(v, y):
    return stats.kruskal(*[v[y == k] for k in range(3)]).statistic


def stratified_p(v, y, strata, rng, n=5000):
    obs = kw_h(v, y)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    hits = 0
    for _ in range(n):
        yp = y.copy()
        for g in groups:
            yp[g] = rng.permutation(yp[g])
        hits += kw_h(v, yp) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def loo_select_lda(X, y, k=5):
    n = len(y)
    proba = np.zeros((n, 3))
    chosen = []
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        H = np.array([kw_h(X[tr, j], y[tr]) for j in range(X.shape[1])])
        sel = np.argsort(H)[::-1][:k]
        chosen.append(sel)
        med = np.median(X[tr][:, sel], 0)
        iqr = np.subtract(*np.percentile(X[tr][:, sel], [75, 25], 0)) / 1.349
        iqr[iqr == 0] = 1
        Z = np.clip((X[:, sel] - med) / iqr, -3, 3)
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(3) / 3).fit(Z[tr], y[tr])
        proba[i] = m.predict_proba(Z[[i]])[0]
    return proba, chosen


def bal_acc(y, pred):
    return float(np.mean([(pred[y == k] == k).mean() for k in range(3)]))


def main():
    sids = [s for _, s in C.sample_ids()]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        res = dict(ex.map(kpis, sids))
    names = sorted({k for v in res.values() for k in v})
    batch = {s: b for b, s in C.sample_ids()}
    C.write_csv(C.OUT_MET / "kpi_values.csv",
                [{"batch": batch[s], "sample_id": s, **{k: round(res[s].get(k, np.nan), 6) for k in names}} for s in sids])
    y = np.array([BATCHES.index(batch[s]) for s in sids])
    qc = {r["sample_id"]: r for r in C.read_csv(C.QC)}
    strata = np.array([qc[s]["height_group"] for s in sids])
    probes = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_session_probes.csv")}
    seg = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "phase_fractions.csv")}
    rng = np.random.default_rng(0)
    rows = []
    keep = [k for k in names if np.isfinite([res[s].get(k, np.nan) for s in sids]).all()]
    for k in keep:
        v = np.array([res[s][k] for s in sids])
        sess = {p: abs(stats.spearmanr(v, [float(probes[s][p]) for s in sids]).statistic) for p in PROBES}
        top_probe = max(sess, key=sess.get)
        rows.append({
            "kpi": k, "B1_mean": round(v[y == 0].mean(), 4), "B2_mean": round(v[y == 1].mean(), 4), "B3_mean": round(v[y == 2].mean(), 4),
            "kruskal_p": stats.kruskal(*[v[y == j] for j in range(3)]).pvalue,
            "B1_vs_B2_mw_p": stats.mannwhitneyu(v[y == 0], v[y == 1]).pvalue,
            "B1_vs_B2_cliffs_delta": round(cliffs_delta(v[y == 0], v[y == 1]), 3),
            "B3_vs_rest_cliffs_delta": round(cliffs_delta(v[y == 2], v[y != 2]), 3),
            "session_stratified_p": stratified_p(v, y, strata, rng, 2000),
            "max_abs_rho_session_probe": round(sess[top_probe], 3), "most_correlated_probe": top_probe,
            "abs_rho_seg_pore": round(abs(stats.spearmanr(v, [float(seg[s]["pore_pct"]) for s in sids]).statistic), 3),
            "abs_rho_seg_siox": round(abs(stats.spearmanr(v, [float(seg[s]["SiOx_pct"]) for s in sids]).statistic), 3)})
    q = bh([r["kruskal_p"] for r in rows])
    qs = bh([r["session_stratified_p"] for r in rows])
    for r, a, b in zip(rows, q, qs):
        r["kruskal_q_bh"], r["session_stratified_q_bh"] = round(a, 4), round(b, 4)
        r["kruskal_p"], r["B1_vs_B2_mw_p"], r["session_stratified_p"] = round(r["kruskal_p"], 4), round(r["B1_vs_B2_mw_p"], 4), round(r["session_stratified_p"], 4)
        r["flag"] = ("session-sensitive" if r["max_abs_rho_session_probe"] > 0.5 else "") + \
                    (" redundant-with-segmentation" if max(r["abs_rho_seg_pore"], r["abs_rho_seg_siox"]) > 0.8 else "")
    rows.sort(key=lambda r: r["kruskal_p"])
    C.write_csv(C.OUT_MET / "kpi_stats.csv", rows)

    X = np.array([[res[s][k] for k in keep] for s in sids])
    proba, chosen = loo_select_lda(X, y)
    ba = bal_acc(y, proba.argmax(1))
    null = []
    for _ in range(300):
        yp = rng.permutation(y)
        null.append(bal_acc(yp, loo_select_lda(X, yp)[0].argmax(1)))
    null = np.array(null)
    picks = np.bincount(np.concatenate(chosen), minlength=len(keep))
    out = {"config": "K (exploratory): top-5 KPIs by in-fold Kruskal H -> shrinkage LDA",
           "n_kpis": len(keep), "LOO_balanced_accuracy": round(ba, 3),
           "recall": {BATCHES[j]: f"{(proba.argmax(1)[y == j] == j).sum()}/{(y == j).sum()}" for j in range(3)},
           "null_mean": round(float(null.mean()), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
           "p_value": round(float((np.sum(null >= ba) + 1) / (len(null) + 1)), 4),
           "most_often_selected": {keep[j]: int(picks[j]) for j in np.argsort(picks)[::-1][:8]}}
    C.write_json(C.OUT_MET / "kpi_classifier.json", out)
    from .bid_model import ledger
    ledger({"config": "K_exploratory", "kind": "LOO", "balanced_accuracy": out["LOO_balanced_accuracy"], "detail": json.dumps(out["recall"])})
    print(f"{len(keep)} KPIs")
    print(f"{'kpi':32s} {'B1':>9s} {'B2':>9s} {'B3':>9s} {'KW p':>7s} {'q':>6s} {'sess p':>7s} {'d12':>6s} {'d3r':>6s} flag")
    for r in rows[:15]:
        print(f"{r['kpi']:32s} {r['B1_mean']:9.3f} {r['B2_mean']:9.3f} {r['B3_mean']:9.3f} {r['kruskal_p']:7.4f} {r['kruskal_q_bh']:6.3f} "
              f"{r['session_stratified_p']:7.4f} {r['B1_vs_B2_cliffs_delta']:6.2f} {r['B3_vs_rest_cliffs_delta']:6.2f} {r['flag']}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
