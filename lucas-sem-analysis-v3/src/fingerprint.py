"""Exploratory: acquisition fingerprinting - can the instrument/session traces in the raw TIFFs predict batch?

Per detector (BSE, Inlens, ETD/SE), on the raw full-resolution grey values:
  noise residual (image - 3x3 median): SD, kurtosis; lag-1 autocorrelation along and across the scan lines;
  line-to-line banding (SD of row means of the residual) and column pattern (SD of column means);
  dominant banding period and its share of row-profile power; high-frequency power share horizontally vs
  vertically; grey-level usage (distinct levels, clipped fractions at 0 / 255, p1 / p50 / p99);
plus image height, width and LZW bytes per pixel.

Classifier: in-fold robust z -> shrinkage LDA (equal priors), leave-one-out and leave-one-session-out (sessions =
image-height groups), permutation null on the leave-one-out score, and Batch_1-vs-Batch_2 alone.
If the leave-session-out score collapses, the fingerprint recognises imaging sessions, not material.

Writes outputs/metrics/fingerprint_values.csv and fingerprint.json.

Usage: python -m src.fingerprint
"""
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C
from .bid_model import BATCHES, ledger, summarise

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


def det_features(a, prefix):
    a = a.astype(np.float32)
    r = a - ndi.median_filter(a, size=3)
    f = {}
    f[f"{prefix}_noise_sd"] = float(r.std())
    f[f"{prefix}_noise_kurtosis"] = float(stats.kurtosis(r[::4, ::4].ravel()))
    f[f"{prefix}_ac_along_line"] = float(np.corrcoef(r[:, :-1].ravel()[::7], r[:, 1:].ravel()[::7])[0, 1])
    f[f"{prefix}_ac_across_line"] = float(np.corrcoef(r[:-1].ravel()[::7], r[1:].ravel()[::7])[0, 1])
    rows, cols = r.mean(1), r.mean(0)
    f[f"{prefix}_row_banding_sd"] = float(rows.std())
    f[f"{prefix}_col_pattern_sd"] = float(cols.std())
    prof = a.mean(1) - ndi.uniform_filter1d(a.mean(1), 101)
    P = np.abs(np.fft.rfft(prof - prof.mean())) ** 2
    P[:3] = 0
    k = int(P.argmax())
    f[f"{prefix}_banding_period_px"] = float(len(prof) / k) if k else 0.0
    f[f"{prefix}_banding_peak_share"] = float(P[k] / max(P.sum(), 1e-9))
    F = np.abs(np.fft.rfft2(r[: r.shape[0] // 2 * 2, : 2048])) ** 2
    fy = np.abs(np.fft.fftfreq(F.shape[0]))[:, None]
    fx = np.fft.rfftfreq(2048)[None, :]
    tot = F.sum()
    f[f"{prefix}_hf_power_x"] = float(F[(fx > 0.3) & (fy < 0.1)].sum() / tot)
    f[f"{prefix}_hf_power_y"] = float(F[(fy > 0.3) & (fx < 0.1)].sum() / tot)
    f[f"{prefix}_distinct_levels"] = float(len(np.unique(a)))
    f[f"{prefix}_clip_low"] = float((a <= 0).mean())
    f[f"{prefix}_clip_high"] = float((a >= 255).mean())
    for q in (1, 50, 99):
        f[f"{prefix}_p{q}"] = float(np.percentile(a, q))
    return f


def one(item):
    (batch, sid), paths = item
    f = {}
    for det, name in (("BSE", "bse"), ("Inlens", "inlens"), ("SE2", "se2")):
        a = np.asarray(Image.open(paths[det]))[..., 1]
        f.update(det_features(a, name))
        f[f"{name}_lzw_bytes_per_px"] = os.path.getsize(paths[det]) / a.size
        if det == "BSE":
            f["height_px"], f["width_px"] = float(a.shape[0]), float(a.shape[1])
    return batch, sid, f


def lda_predict(X, y, folds):
    pred = np.zeros(len(y), int)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        med = np.median(X[tr], 0)
        iqr = np.subtract(*np.percentile(X[tr], [75, 25], 0)) / 1.349
        iqr[iqr == 0] = 1
        Z = np.clip((X - med) / iqr, -3, 3)
        k = len(np.unique(y[tr]))
        pred[te] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(k) / k).fit(Z[tr], y[tr]).predict(Z[te])
    return pred


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def main():
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, C.discover().items()))
    names = sorted(res[0][2])
    C.write_csv(C.OUT_MET / "fingerprint_values.csv",
                [{"batch": b, "sample_id": s, **{k: round(f[k], 6) for k in names}} for b, s, f in res])
    sids = [s for _, s, _ in res]
    y = np.array([BATCHES.index(b) for b, _, _ in res])
    X = np.array([[f[k] for k in names] for _, _, f in res])
    qc = {r["sample_id"]: r["height_group"] for r in C.read_csv(C.QC)}
    groups = np.array([qc[s] for s in sids])
    loo = [np.array([i]) for i in range(len(y))]
    logo = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    out = {"n_features": len(names)}
    # without image size (height is literally the session key) and with it
    for variant, cols in (("all_fingerprint", list(range(len(names)))),
                          ("without_image_size", [j for j, n in enumerate(names) if n not in ("height_px", "width_px")])):
        Xv = X[:, cols]
        p_loo = lda_predict(Xv, y, loo)
        p_logo = lda_predict(Xv, y, logo)
        rng = np.random.default_rng(0)
        null = np.array([bal(yp, lda_predict(Xv, yp, loo)) for yp in (rng.permutation(y) for _ in range(300))])
        m12 = y < 2
        p12_loo = lda_predict(Xv[m12], y[m12], [np.array([i]) for i in range(m12.sum())])
        g12 = groups[m12]
        p12_logo = lda_predict(Xv[m12], y[m12], [np.flatnonzero(g12 == g) for g in np.unique(g12)])
        out[variant] = {
            "LOO": summarise(y, np.eye(3)[p_loo]), "LOGO_session": summarise(y, np.eye(3)[p_logo]),
            "LOO_perm_p": round(float((np.sum(null >= bal(y, p_loo)) + 1) / 301), 4), "null_p95": round(float(np.percentile(null, 95)), 3),
            "B1_vs_B2_only": {"LOO_correct": f"{int((p12_loo == y[m12]).sum())}/14", "LOO_balanced": round(bal(y[m12], p12_loo), 3),
                              "LOGO_correct": f"{int((p12_logo == y[m12]).sum())}/14", "LOGO_balanced": round(bal(y[m12], p12_logo), 3)}}
        o = out[variant]
        print(f"{variant:20s} LOO {o['LOO']['balanced_accuracy']:.3f} {o['LOO']['recall']} (perm p {o['LOO_perm_p']}) | "
              f"LOGO {o['LOGO_session']['balanced_accuracy']:.3f} {o['LOGO_session']['recall']} | B1 vs B2: {o['B1_vs_B2_only']}")
    # which fingerprint features separate B1 and B2 most
    m12 = y < 2
    d = []
    for j, n in enumerate(names):
        a, b = X[y == 0, j], X[y == 1, j]
        delta = ((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / 49
        d.append((abs(delta), n, round(float(delta), 2), round(float(stats.mannwhitneyu(a, b).pvalue), 4)))
    out["top_B1_vs_B2_fingerprint_features"] = [{"feature": n, "cliffs_delta": dl, "mw_p": p} for _, n, dl, p in sorted(d, reverse=True)[:8]]
    C.write_json(C.OUT_MET / "fingerprint.json", out)
    print("top B1-vs-B2 fingerprint features:", out["top_B1_vs_B2_fingerprint_features"][:6])
    ledger({"config": "fingerprint_exploratory", "kind": "LOO", "balanced_accuracy": out["all_fingerprint"]["LOO"]["balanced_accuracy"],
            "detail": json.dumps(out["all_fingerprint"]["LOO"]["recall"])})


if __name__ == "__main__":
    main()
