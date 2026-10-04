"""Candidate final batch model: per-view acquisition-fingerprint LDAs (BSE, Inlens, ETD/SE) + segmentation head.

Fixed before scoring (2026-10-03, after the teammate's fingerprint report and its independent replication):
  FP    = mean of three shrinkage LDAs, one per detector view, on that view's fingerprint measurements
          (the teammate's design, rebuilt on src.fingerprint measurements)
  FP+S  = mean of the three view LDAs and the segmentation head S (5 label-map features, shrinkage LDA)
Every component is a linear model on named measurements, so each prediction decomposes into per-feature pushes.

Evaluation: leave-one-location-out, leave-one-session-out (image-height groups), permutation null (1000 shuffles,
whole procedure inside), per-batch recall, and a two-level view ("Batch_3" vs "Batch_1 or 2").

Writes outputs/metrics/final_model.json.

Usage: python -m src.final_model
"""
import json
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C
from .bid_features import S_FEATURES
from .bid_model import BATCHES, ledger, summarise

warnings.filterwarnings("ignore")


def lda_proba(X, y, folds):
    P = np.zeros((len(y), 3))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        med = np.median(X[tr], 0)
        iqr = np.subtract(*np.percentile(X[tr], [75, 25], 0)) / 1.349
        iqr[iqr == 0] = 1
        Z = np.clip((X - med) / iqr, -3, 3)
        P[te] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(3) / 3).fit(Z[tr], y[tr]).predict_proba(Z[te])
    return P


def blocks():
    fp = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "fingerprint_values.csv")}
    feats = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_features.csv")}
    sids = [s for _, s in C.sample_ids()]
    y = np.array([BATCHES.index(fp[s]["batch"]) for s in sids])
    views = {v: np.array([[float(fp[s][k]) for k in fp[sids[0]] if k.startswith(v + "_")] for s in sids])
             for v in ("bse", "inlens", "se2")}
    S = np.array([[float(feats[s][k]) for k in S_FEATURES] for s in sids])
    qc = {r["sample_id"]: r["height_group"] for r in C.read_csv(C.QC)}
    return sids, y, views, S, np.array([qc[s] for s in sids])


def predict(views, S, y, folds, with_s):
    parts = [lda_proba(X, y, folds) for X in views.values()]
    if with_s:
        parts.append(lda_proba(S, y, folds))
    return np.mean(parts, 0)


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def two_level(y, P):
    b3 = P[:, 2] >= 0.5
    t = y == 2
    return {"Batch_3_recall": f"{int((b3 & t).sum())}/{int(t.sum())}", "Batch_1or2_recall": f"{int((~b3 & ~t).sum())}/{int((~t).sum())}",
            "balanced": round(((b3 & t).sum() / t.sum() + (~b3 & ~t).sum() / (~t).sum()) / 2, 3)}


def main():
    sids, y, views, S, groups = blocks()
    loo = [np.array([i]) for i in range(len(y))]
    logo = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(0)
    out = {}
    for name, with_s in (("FP", False), ("FP+S", True)):
        P_loo, P_logo = predict(views, S, y, loo, with_s), predict(views, S, y, logo, with_s)
        obs = bal(y, P_loo.argmax(1))
        null = np.array([bal(yp, predict(views, S, yp, loo, with_s).argmax(1)) for yp in (rng.permutation(y) for _ in range(1000))])
        out[name] = {"LOO": summarise(y, P_loo), "LOGO_session": summarise(y, P_logo),
                     "locations_correct_LOO": f"{int((P_loo.argmax(1) == y).sum())}/31",
                     "perm_p": round(float((np.sum(null >= obs) + 1) / 1001), 4),
                     "null_p95": round(float(np.percentile(null, 95)), 3), "null_p99": round(float(np.percentile(null, 99)), 3),
                     "two_level_LOO": two_level(y, P_loo), "two_level_LOGO": two_level(y, P_logo),
                     "per_sample_LOO": {s: {"true": BATCHES[y[i]], "pred": BATCHES[int(P_loo[i].argmax())],
                                            "p": [round(float(v), 3) for v in P_loo[i]]} for i, s in enumerate(sids)}}
        o = out[name]
        print(f"{name:5s} LOO bal {o['LOO']['balanced_accuracy']:.3f} {o['LOO']['recall']} ({o['locations_correct_LOO']}) "
              f"perm p {o['perm_p']} | LOGO bal {o['LOGO_session']['balanced_accuracy']:.3f} {o['LOGO_session']['recall']} | "
              f"B3-vs-B1/2: LOO {o['two_level_LOO']['balanced']} LOGO {o['two_level_LOGO']['balanced']}", flush=True)
        ledger({"config": f"final_{name}", "kind": "LOO", "balanced_accuracy": o["LOO"]["balanced_accuracy"],
                "detail": json.dumps(o["LOO"]["recall"])})
    C.write_json(C.OUT_MET / "final_model.json", out)


if __name__ == "__main__":
    main()
