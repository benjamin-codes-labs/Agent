"""Exploratory: acquisition fingerprint for Batch_1 vs Batch_2 only, with an explanation.

Classifiers on the 14 Batch_1 / Batch_2 samples (fingerprint features from src.fingerprint):
  all      in-fold robust z of all fingerprint features -> shrinkage LDA
  top3     3 features with the largest |Cliff's delta| chosen inside each fold -> shrinkage LDA
  stump    the single best feature chosen inside each fold + a threshold at the midpoint of the class medians
Leave-one-out and leave-one-session-out (image-height groups); permutation null (2000 shuffles of the B1/B2
labels) with the feature choice inside every shuffle.

Explanation: for the most-selected features, B1 vs B2 values, effect size, the three same-session B1/B2 pairs, and
Spearman correlation with the segmented material fractions (is it the microscope, or does the material change the
signal?). Figure: reports/figures/fingerprint_b12.png.

Writes outputs/metrics/fingerprint_b12.json.

Usage: python -m src.fingerprint_b12
"""
import json
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C
from .bid_model import ledger

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
PAIRS = [("f1vzngrs", "epqdaau9", "h2148"), ("fzrt2k6r", "b3esycq1", "h2156"), ("ffwubibz", "r17byphk", "h2080")]
MEANING = {
    "col_pattern_sd": "faint vertical stripes: column-to-column brightness differences in the noise",
    "banding_peak_share": "how strongly one repeating horizontal banding period dominates (scan-line rhythm)",
    "row_banding_sd": "line-to-line brightness jitter (beam scan lines)",
    "hf_power_y": "fine-grain noise energy across scan lines",
    "hf_power_x": "fine-grain noise energy along scan lines",
    "noise_sd": "overall graininess",
    "ac_along_line": "how smeared the noise is along the scan direction (scan speed / filtering)",
    "ac_across_line": "how correlated neighbouring scan lines are",
    "p50": "median grey level (brightness setting)", "p1": "black level (brightness/offset setting)",
    "p99": "white level (contrast/gain setting)", "distinct_levels": "how many grey levels are used (dynamic range)",
    "clip_low": "share of pixels clipped to black", "clip_high": "share of pixels clipped to white",
    "lzw_bytes_per_px": "file compressibility (noise texture)", "noise_kurtosis": "spikiness of the noise",
    "banding_period_px": "dominant banding period",
}


def cliffs(a, b):
    return ((a[:, None, :] > b[None]).sum((0, 1)) - (a[:, None, :] < b[None]).sum((0, 1))) / (len(a) * len(b))


def zscale(X, tr):
    med = np.median(X[tr], 0)
    iqr = np.subtract(*np.percentile(X[tr], [75, 25], 0)) / 1.349
    iqr[iqr == 0] = 1
    return np.clip((X - med) / iqr, -3, 3)


def fit_predict(kind, X, y, tr, te):
    if kind == "all":
        Z = zscale(X, tr)
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5]).fit(Z[tr], y[tr]).predict(Z[te]), None
    d = cliffs(X[tr][y[tr] == 0], X[tr][y[tr] == 1])
    order = np.argsort(-np.abs(d))
    if kind == "top3":
        sel = order[:3]
        Z = zscale(X[:, sel], tr)
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5]).fit(Z[tr], y[tr]).predict(Z[te]), sel
    j = order[0]
    m0, m1 = np.median(X[tr][y[tr] == 0, j]), np.median(X[tr][y[tr] == 1, j])
    thr = (m0 + m1) / 2
    return np.where((X[te, j] > thr) == (m1 > m0), 1, 0), np.array([j])


def evaluate(kind, X, y, folds):
    pred = np.zeros(len(y), int)
    chosen = []
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        p, sel = fit_predict(kind, X, y, tr, te)
        pred[te] = p
        if sel is not None:
            chosen += list(sel)
    return pred, chosen


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in (0, 1)]))


def main():
    fp = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "fingerprint_values.csv")}
    names = [k for k in next(iter(fp.values())) if k not in ("batch", "sample_id", "height_px", "width_px")]
    sids = [s for b, s in C.sample_ids() if b in ("Batch_1", "Batch_2")]
    y = np.array([0 if fp[s]["batch"] == "Batch_1" else 1 for s in sids])
    X = np.array([[float(fp[s][k]) for k in names] for s in sids])
    qc = {r["sample_id"]: r["height_group"] for r in C.read_csv(C.QC)}
    groups = np.array([qc[s] for s in sids])
    loo = [np.array([i]) for i in range(len(y))]
    logo = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(0)
    out = {"n_samples": len(y), "n_features": len(names), "classifiers": {}}
    all_chosen = Counter()
    for kind in ("all", "top3", "stump"):
        p_loo, ch = evaluate(kind, X, y, loo)
        p_logo, _ = evaluate(kind, X, y, logo)
        all_chosen.update(names[j] for j in ch)
        obs = bal(y, p_loo)
        null = np.array([bal(yp, evaluate(kind, X, yp, loo)[0]) for yp in (rng.permutation(y) for _ in range(2000))])
        out["classifiers"][kind] = {
            "LOO_correct": f"{int((p_loo == y).sum())}/14", "LOO_balanced": round(obs, 3),
            "LOGO_correct": f"{int((p_logo == y).sum())}/14", "LOGO_balanced": round(bal(y, p_logo), 3),
            "perm_p": round(float((np.sum(null >= obs) + 1) / 2001), 4), "null_p95": round(float(np.percentile(null, 95)), 3),
            "features_chosen": dict(Counter(names[j] for j in ch).most_common(5))}
        print(kind, out["classifiers"][kind], flush=True)

    # explanation
    seg = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "phase_fractions.csv")}
    all_ids = [s for _, s in C.sample_ids()]
    d_all = cliffs(X[y == 0], X[y == 1])
    explain = []
    for name, _ in all_chosen.most_common(6):
        j = names.index(name)
        det, key = name.split("_", 1)
        v_all = np.array([float(fp[s][name]) for s in all_ids])
        corr = {c: round(float(stats.spearmanr(v_all, [float(seg[s][f"{c}_pct"]) for s in all_ids]).statistic), 2)
                for c in ("pore", "graphite", "SiOx", "CBD")}
        pairs = [{"session": g, "B1": round(float(fp[a][name]), 5), "B2": round(float(fp[b][name]), 5),
                  "B1_minus_B2_sign_matches_batch_difference": bool(np.sign(float(fp[a][name]) - float(fp[b][name])) == np.sign(d_all[j]))}
                 for a, b, g in PAIRS]
        explain.append({"feature": name, "detector": {"bse": "BSE", "inlens": "Inlens", "se2": "ETD/SE"}[det],
                        "meaning": MEANING.get(key, key), "times_selected": all_chosen[name],
                        "B1_values": [round(float(v), 5) for v in X[y == 0, j]], "B2_values": [round(float(v), 5) for v in X[y == 1, j]],
                        "cliffs_delta_B1_vs_B2": round(float(d_all[j]), 2), "mw_p": round(float(stats.mannwhitneyu(X[y == 0, j], X[y == 1, j]).pvalue), 4),
                        "same_session_pairs": pairs, "spearman_with_material_all31": corr})
    out["explanation"] = explain
    C.write_json(C.OUT_MET / "fingerprint_b12.json", out)
    ledger({"config": "fingerprint_B1vB2_exploratory", "kind": "LOO",
            "balanced_accuracy": max(v["LOO_balanced"] for v in out["classifiers"].values()),
            "detail": json.dumps({k: v["LOO_correct"] for k, v in out["classifiers"].items()})})
    figure(explain[:4], fp, sids, y)
    for e in explain[:4]:
        print(f"\n{e['feature']} ({e['detector']}): {e['meaning']}\n  selected {e['times_selected']}x | delta {e['cliffs_delta_B1_vs_B2']} p {e['mw_p']}"
              f"\n  same-session pairs: {[(p['session'], p['B1'], p['B2'], p['B1_minus_B2_sign_matches_batch_difference']) for p in e['same_session_pairs']]}"
              f"\n  correlation with material (all 31): {e['spearman_with_material_all31']}")


def column_profile(sid, det):
    paths = C.discover()[[k for k in C.discover() if k[1] == sid][0]]
    a = np.asarray(Image.open(paths[det]))[..., 1].astype(np.float32)
    r = a - ndi.median_filter(a, size=3)
    return r.mean(0)


def figure(explain, fp, sids, y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, e in zip(axes.ravel()[:4], explain):
        name = e["feature"]
        v = {s: float(fp[s][name]) for s in sids}
        b3 = [float(r[name]) for r in fp.values() if r["batch"] == "Batch_3"]
        ax.scatter(np.full(len(b3), 2) + np.random.default_rng(1).uniform(-.08, .08, len(b3)), b3, c="0.75", s=25, label="Batch 3")
        for k, (lab, col) in enumerate((("Batch 1", "tab:blue"), ("Batch 2", "tab:orange"))):
            vals = [v[s] for s, yy in zip(sids, y) if yy == k]
            ax.scatter(np.full(len(vals), k) + np.random.default_rng(k).uniform(-.08, .08, len(vals)), vals, c=col, s=40, label=lab)
        for a, b, g in PAIRS:
            ax.plot([0, 1], [v[a], v[b]], "k--", lw=0.8, alpha=0.6)
        ax.set_xticks([0, 1, 2], ["Batch 1", "Batch 2", "Batch 3"])
        ax.set_title(f"{name}\n(Cliff's δ B1 vs B2 = {e['cliffs_delta_B1_vs_B2']}, p = {e['mw_p']})", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 0].text(0.02, -0.18, "dashed lines = same imaging session (B1 ↔ B2)", transform=axes[0, 0].transAxes, fontsize=8)
    top = explain[0]["feature"]
    det = {"bse": "BSE", "inlens": "Inlens", "se2": "SE2"}[top.split("_", 1)[0]]
    for ax, (sid, lab, col) in zip(axes[1, 2:], [("f1vzngrs", "Batch 1", "tab:blue")]):
        pass
    ax = axes[1, 2]
    for sid, lab, col in (("fzrt2k6r", "Batch 1 (fzrt2k6r)", "tab:blue"), ("b3esycq1", "Batch 2 (b3esycq1)", "tab:orange")):
        prof = column_profile(sid, det)[:600]
        ax.plot(prof, color=col, lw=0.8, label=lab)
    ax.set_title(f"{det} column profile of the noise, first 600 columns\n(same imaging session h2156)", fontsize=10)
    ax.set_xlabel("column (px)"); ax.set_ylabel("mean noise residual (grey levels)")
    ax.legend(fontsize=8)
    for ax in axes.ravel()[len(explain):5]:
        ax.axis("off")
    fig.suptitle("Acquisition fingerprint: Batch 1 vs Batch 2", fontsize=13)
    fig.tight_layout()
    path = C.REPORTS / "figures" / "fingerprint_b12.png"
    fig.savefig(path, dpi=110)
    print("figure ->", path)


if __name__ == "__main__":
    main()
