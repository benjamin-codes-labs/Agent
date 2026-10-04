"""Acquisition-fingerprint batch model (the teammate's per-view design, rebuilt on src.fingerprint measurements).

One shrinkage-LDA per detector view (BSE, Inlens, ETD/SE) on that view's fingerprint measurements (noise level and
shape, scan-line correlations, banding and stripe strength, grey levels and clipping, compressibility); the view
probabilities are averaged. Views that are missing for a new image are skipped. Each prediction is explained by the
measurements that pushed it hardest (LDA coefficient x robust z), with a plain-language meaning.

  fit      train on all 31 locations -> models/fingerprint_model.pkl; also writes honest leave-one-location-out and
           leave-one-session-out probabilities (overall and per view) -> outputs/metrics/fingerprint_model_cv.json
  predict  python -m src.fingerprint_model predict --bse X_BSE.tif --inlens X_Inlens.tif [--etd X_ETD.tif]

Usage: python -m src.fingerprint_model fit
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C
from .bid_model import BATCHES
from .fingerprint import det_features

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
MODEL = C.MODELS / "fingerprint_model.pkl"
VIEWS = {"bse": "BSE", "inlens": "Inlens", "se2": "ETD/SE"}
MEANING = {
    "col_pattern_sd": "vertical stripe strength in the noise (often ion-milling curtaining)",
    "banding_peak_share": "how strongly one horizontal banding rhythm dominates (scan lines)",
    "banding_period_px": "spacing of the dominant horizontal banding",
    "row_banding_sd": "line-to-line brightness jitter",
    "hf_power_y": "fine-grain noise energy across scan lines",
    "hf_power_x": "fine-grain noise energy along scan lines",
    "noise_sd": "overall graininess", "noise_kurtosis": "spikiness of the noise",
    "ac_along_line": "how smeared the noise is along the scan direction",
    "ac_across_line": "how correlated neighbouring scan lines are",
    "p1": "black level", "p50": "median grey level", "p99": "white level",
    "distinct_levels": "number of grey levels used (dynamic range)",
    "clip_low": "share of pixels clipped to black", "clip_high": "share of pixels clipped to white",
    "lzw_bytes_per_px": "file compressibility (noise texture)",
}


def measure_view(path, prefix):
    a = np.asarray(Image.open(path))[..., 1]
    f = det_features(a, prefix)
    f[f"{prefix}_lzw_bytes_per_px"] = os.path.getsize(path) / a.size
    return f


def training_table():
    fp = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "fingerprint_values.csv")}
    sids = [s for _, s in C.sample_ids()]
    y = np.array([BATCHES.index(fp[s]["batch"]) for s in sids])
    names = {v: [k for k in fp[sids[0]] if k.startswith(v + "_")] for v in VIEWS}
    X = {v: np.array([[float(fp[s][k]) for k in names[v]] for s in sids]) for v in VIEWS}
    return sids, y, names, X


def fit_view(X, y):
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], 0)) / 1.349
    iqr[iqr == 0] = 1
    Z = np.clip((X - med) / iqr, -3, 3)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(3) / 3).fit(Z, y)
    return {"median": med, "iqr": iqr, "lda": lda}


def view_proba(m, x):
    z = np.clip((x - m["median"]) / m["iqr"], -3, 3)
    return m["lda"].predict_proba(z[None])[0], z


def cv_probs(X, y, folds):
    P = {v: np.zeros((len(y), 3)) for v in X}
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        for v in X:
            m = fit_view(X[v][tr], y[tr])
            for i in te:
                P[v][i] = view_proba(m, X[v][i])[0]
    return P


CAL_BINS = ((0.99, 1.01), (0.9, 0.99), (0.6, 0.9), (0.0, 0.6))


def calibration(P, y):
    """How often the top pick was right, by how sure the model claimed to be (LDA probabilities are over-confident)."""
    top, ok = P.max(1), P.argmax(1) == y
    return [{"claimed_top_probability": f"{lo:.2f}-{min(hi, 1):.2f}", "right": f"{int(ok[(top >= lo) & (top < hi)].sum())}/"
             f"{int(((top >= lo) & (top < hi)).sum())}"} for lo, hi in CAL_BINS]


def calibration_for(p_top, scheme="LOO"):
    cv = C.OUT_MET / "fingerprint_model_cv.json"
    if not cv.exists():
        return None
    cal = json.load(open(cv)).get(scheme + "_calibration")
    for (lo, hi), row in zip(CAL_BINS, cal or []):
        if lo <= p_top < hi:
            return {"claimed": round(float(p_top), 3), "in_leave_one_out_tests_at_this_level": row["right"] + " right",
                    "note": "the fingerprint model's raw probabilities are over-confident; use this empirical rate"}
    return None


def fit_model(exclude=()):
    sids, y, names, X = training_table()
    keep = np.array([i for i, s in enumerate(sids) if s not in set(exclude)])
    return {"views": {v: {"features": names[v], **fit_view(X[v][keep], y[keep])} for v in VIEWS},
            "batches": BATCHES, "trained_on": [sids[i] for i in keep]}


def cmd_fit(_):
    sids, y, names, X = training_table()
    model = {"views": {v: {"features": names[v], **fit_view(X[v], y)} for v in VIEWS}, "batches": BATCHES, "trained_on": sids}
    with open(MODEL, "wb") as f:
        pickle.dump(model, f)
    qc = {r["sample_id"]: r["height_group"] for r in C.read_csv(C.QC)}
    groups = np.array([qc[s] for s in sids])
    out = {}
    for scheme, folds in (("LOO", [np.array([i]) for i in range(len(y))]),
                          ("LOGO_session", [np.flatnonzero(groups == g) for g in np.unique(groups)])):
        P = cv_probs(X, y, folds)
        avg = np.mean(list(P.values()), 0)
        out[scheme] = {s: {"true": BATCHES[y[i]], "p": [round(float(v), 4) for v in avg[i]],
                           "per_view": {v: [round(float(x), 4) for x in P[v][i]] for v in P}} for i, s in enumerate(sids)}
        out[scheme + "_calibration"] = calibration(avg, y)
        acc = np.mean([(avg.argmax(1)[y == c] == c).mean() for c in range(3)])
        print(f"fingerprint {scheme}: balanced accuracy {acc:.3f}")
    C.write_json(C.OUT_MET / "fingerprint_model_cv.json", out)
    print(f"fingerprint model ({sum(len(v) for v in names.values())} measurements, 3 views) -> {MODEL}")


def predict(feats_by_view, model=None, top=3):
    """feats_by_view: {view: {feature: value}} for the available views. Returns probabilities and explanations."""
    model = model or pickle.load(open(MODEL, "rb"))
    per_view, expl = {}, []
    for v, m in model["views"].items():
        if v not in feats_by_view:
            continue
        x = np.array([feats_by_view[v][k] for k in m["features"]])
        p, z = view_proba(m, x)
        per_view[v] = p
    if not per_view:
        raise ValueError("no fingerprint views available")
    avg = np.mean(list(per_view.values()), 0)
    order = np.argsort(avg)[::-1]
    win, run = int(order[0]), int(order[1])
    for v, m in model["views"].items():
        if v not in feats_by_view:
            continue
        x = np.array([feats_by_view[v][k] for k in m["features"]])
        _, z = view_proba(m, x)
        push = (m["lda"].coef_[win] - m["lda"].coef_[run]) * z
        for j in np.argsort(-np.abs(push))[:top]:
            key = m["features"][j].split("_", 1)[1]
            expl.append({"view": VIEWS[v], "measurement": m["features"][j], "meaning": MEANING.get(key, key),
                         "robust_z": round(float(z[j]), 2), "direction": "high" if z[j] > 0 else "low",
                         "push_toward_winner": round(float(push[j]), 3)})
    expl.sort(key=lambda e: -abs(e["push_toward_winner"]))
    return {"probabilities": dict(zip(BATCHES, np.round(avg, 3).tolist())),
            "per_view": {VIEWS[v]: dict(zip(BATCHES, np.round(p, 3).tolist())) for v, p in per_view.items()},
            "views_used": [VIEWS[v] for v in per_view],
            "predicted_batch": BATCHES[win], "runner_up": BATCHES[run],
            "views_agree": len({int(np.argmax(p)) for p in per_view.values()}) == 1,
            "calibration": calibration_for(float(avg.max())),
            "top_measurements": expl[:5]}


def cmd_predict(args):
    feats = {"bse": measure_view(args.bse, "bse"), "inlens": measure_view(args.inlens, "inlens")}
    if args.etd:
        feats["se2"] = measure_view(args.etd, "se2")
    print(json.dumps(predict(feats), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "predict"])
    ap.add_argument("--bse")
    ap.add_argument("--inlens")
    ap.add_argument("--etd")
    args = ap.parse_args()
    {"fit": cmd_fit, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
