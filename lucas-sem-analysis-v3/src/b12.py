"""Exploratory: is there any Batch_1 vs Batch_2 signal, and does a two-step classifier help?

Part 1 - univariate B1 vs B2 screen over every KPI (non-ML KPIs, ETD/SE topography + grain KPIs, segmentation
features): Mann-Whitney p, Cliff's delta, Benjamini-Hochberg q; the same with the one SE-detector Batch_2 sample
(rxax5ozo) removed; the sign of B1 - B2 in the three same-session B1/B2 pairs (h2148, h2156, h2080), which is the
most direct evidence against an imaging-session explanation; and |Spearman rho| with the acquisition probes.

Part 2 - two-step classifier:
  step 1: "is it Batch_3?" = argmax of the original fused model F (its leave-one-out / leave-session-out
          probabilities, so step 1 is honest);
  step 2: "Batch_1 or Batch_2?" = trained only on the training fold's B1/B2 samples; the 3 KPIs with the largest
          |Cliff's delta| ON THE TRAINING FOLD are selected, robust-z scaled, shrinkage LDA (equal priors).
Variants: pool = all KPIs, or only KPIs not flagged as session-sensitive (|rho| with an acquisition probe > 0.5).
Step 2 alone is also scored on the 14 B1/B2 samples (leave-one-out), against a permutation null that shuffles
the B1/B2 labels with the feature selection inside every shuffle.

Writes outputs/metrics/b12_screen.csv and outputs/metrics/b12_two_step.json.

Usage: python -m src.b12 [--n 2000]
"""
import argparse
import json

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C
from .bid_model import BATCHES, bal_acc, ledger, load, procedure, summarise

PAIRS = [("f1vzngrs", "epqdaau9", "h2148"), ("fzrt2k6r", "b3esycq1", "h2156"), ("ffwubibz", "r17byphk", "h2080")]
SE_SAMPLE = "rxax5ozo"
PROBES = ["height_px", "bse_noise_sigma", "inlens_sharpness", "inlens_banding_power", "bse_file_mb", "bse_p50", "inlens_p50"]


def table():
    def rd(name, drop=()):
        return {r["sample_id"]: {k: float(v) for k, v in r.items() if k not in ("batch", "sample_id") + tuple(drop)
                                 and not k.endswith("_half_sd")}
                for r in C.read_csv(C.OUT_MET / name)}
    parts = [rd("kpi_values.csv"), rd("kpi_topo_values.csv", drop=("se2_detector_is_SE",)), rd("bid_features.csv")]
    sids = [s for _, s in C.sample_ids()]
    names = sorted({k for p in parts for k in next(iter(p.values()))})
    X = np.array([[next(p[s][k] for p in parts if k in p[s]) for k in names] for s in sids])
    keep = np.isfinite(X).all(0) & (X.std(0) > 0)
    return sids, [n for n, k in zip(names, keep) if k], X[:, keep]


def cliffs(a, b):
    """Cliff's delta for every column: a (n1, F), b (n2, F)."""
    return ((a[:, None, :] > b[None]).sum((0, 1)) - (a[:, None, :] < b[None]).sum((0, 1))) / (len(a) * len(b))


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty(len(p))
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * len(p) / rank)
        q[i] = prev
    return q


def step2_predict(X, y12, tr, te, cols, k=3):
    """Train B1-vs-B2 on rows tr (labels 0 / 1), select top-k |delta| among `cols`, predict rows te."""
    a, b = X[tr][y12[tr] == 0][:, cols], X[tr][y12[tr] == 1][:, cols]
    sel = np.array(cols)[np.argsort(-np.abs(cliffs(a, b)))[:k]]
    med = np.median(X[tr][:, sel], 0)
    iqr = np.subtract(*np.percentile(X[tr][:, sel], [75, 25], 0)) / 1.349
    iqr[iqr == 0] = 1
    Z = np.clip((X[:, sel] - med) / iqr, -3, 3)
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(Z[tr], y12[tr])
    return m.predict(Z[te]), sel


def step2_loo(X, y12, idx, cols):
    """Leave-one-out over the B1/B2 rows idx; returns predictions and selected features per fold."""
    pred, chosen = {}, []
    for i in idx:
        tr = np.array([j for j in idx if j != i])
        p, sel = step2_predict(X, y12, tr, np.array([i]), cols)
        pred[i] = int(p[0]); chosen.append(sel)
    return pred, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    sids, names, X = table()
    batch = {s: b for b, s in C.sample_ids()}
    y = np.array([BATCHES.index(batch[s]) for s in sids])
    probes = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_session_probes.csv")}
    P = np.array([[float(probes[s][p]) for p in PROBES] for s in sids])

    # ---------------- part 1: univariate screen
    b1, b2 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    b2_noSE = np.array([i for i in b2 if sids[i] != SE_SAMPLE])
    d_all = cliffs(X[b1], X[b2])
    d_noSE = cliffs(X[b1], X[b2_noSE])
    p_all = np.array([stats.mannwhitneyu(X[b1, j], X[b2, j]).pvalue for j in range(len(names))])
    p_noSE = np.array([stats.mannwhitneyu(X[b1, j], X[b2_noSE, j]).pvalue for j in range(len(names))])
    q_all = bh(p_all)
    sess_rho = np.array([max(abs(stats.spearmanr(X[:, j], P[:, k]).statistic) for k in range(P.shape[1])) for j in range(len(names))])
    rows = []
    for j, n in enumerate(names):
        signs = [np.sign(X[sids.index(a), j] - X[sids.index(b), j]) for a, b, _ in PAIRS]
        agree = sum(s == np.sign(d_all[j]) and s != 0 for s in signs)
        rows.append({"kpi": n, "B1_mean": round(X[b1, j].mean(), 5), "B2_mean": round(X[b2, j].mean(), 5),
                     "cliffs_delta_B1_vs_B2": round(float(d_all[j]), 3), "mw_p": round(float(p_all[j]), 4), "bh_q": round(float(q_all[j]), 4),
                     "cliffs_delta_without_SE_sample": round(float(d_noSE[j]), 3), "mw_p_without_SE_sample": round(float(p_noSE[j]), 4),
                     "same_session_pairs_agreeing": f"{agree}/3", "max_abs_rho_acquisition": round(float(sess_rho[j]), 3),
                     "session_sensitive": bool(sess_rho[j] > 0.5)})
    rows.sort(key=lambda r: r["mw_p"])
    C.write_csv(C.OUT_MET / "b12_screen.csv", rows)
    print(f"{len(names)} KPIs screened for Batch_1 vs Batch_2 (n = 7 vs 7)")
    print(f"{'kpi':34s} {'B1':>9s} {'B2':>9s} {'delta':>6s} {'p':>7s} {'q':>6s} {'delta-SE':>8s} {'pairs':>5s} {'acq rho':>7s}")
    for r in rows[:15]:
        print(f"{r['kpi']:34s} {r['B1_mean']:9.4g} {r['B2_mean']:9.4g} {r['cliffs_delta_B1_vs_B2']:6.2f} {r['mw_p']:7.4f} {r['bh_q']:6.3f} "
              f"{r['cliffs_delta_without_SE_sample']:8.2f} {r['same_session_pairs_agreeing']:>5s} {r['max_abs_rho_acquisition']:7.2f}{' S' if r['session_sensitive'] else ''}")

    # ---------------- part 2: two-step classifier
    d = load()
    assert list(d["sids"]) == sids
    loo_F = procedure(d, y)["F"]
    groups = d["groups"]
    folds = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    logo_F = procedure(d, y, folds)["F"]
    y12 = np.where(y == 1, 1, 0)                       # 0 = Batch_1, 1 = Batch_2 (only used on B1/B2 rows)
    b12_idx = np.flatnonzero(y < 2)
    pools = {"all_kpis": list(range(len(names))),
             "non_session_kpis": [j for j in range(len(names)) if not sess_rho[j] > 0.5]}
    out = {"n_kpis": len(names), "variants": {}}
    for vname, cols in pools.items():
        res = {}
        for scheme, pF, fold_of in (("LOO", loo_F, {i: np.array([i]) for i in range(len(y))}),
                                    ("LOGO_session", logo_F, {i: f for f in folds for i in f})):
            pred = np.zeros(len(y), int)
            for i in range(len(y)):
                if pF[i].argmax() == 2:
                    pred[i] = 2
                    continue
                held = fold_of[i]
                tr = np.array([j for j in b12_idx if j not in held])
                p, _ = step2_predict(X, y12, tr, np.array([i]), cols)
                pred[i] = int(p[0])
            onehot = np.eye(3)[pred]
            res[scheme] = summarise(y, onehot)
        # step 2 alone on the 14 B1/B2 samples + permutation null with selection inside
        p2, chosen = step2_loo(X, y12, b12_idx, cols)
        acc2 = float(np.mean([p2[i] == y12[i] for i in b12_idx]))
        rng = np.random.default_rng(3)
        null = []
        for _ in range(args.n):
            yp = y12.copy()
            yp[b12_idx] = rng.permutation(y12[b12_idx])
            pp, _ = step2_loo(X, yp, b12_idx, cols)
            null.append(np.mean([pp[i] == yp[i] for i in b12_idx]))
        null = np.array(null)
        picks = np.bincount(np.concatenate(chosen), minlength=len(names))
        res["step2_B1_vs_B2_only"] = {"accuracy": round(acc2, 3), "correct": f"{int(round(acc2 * 14))}/14",
                                      "null_mean": round(float(null.mean()), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                                      "p_value": round(float((np.sum(null >= acc2) + 1) / (len(null) + 1)), 4),
                                      "most_selected": {names[j]: int(picks[j]) for j in np.argsort(-picks)[:6] if picks[j]}}
        out["variants"][vname] = res
        ledger({"config": f"two_step_{vname}_exploratory", "kind": "LOO", "balanced_accuracy": res["LOO"]["balanced_accuracy"],
                "detail": json.dumps(res["LOO"]["recall"])})
    out["reference_original_F"] = {"LOO": summarise(y, loo_F), "LOGO_session": summarise(y, logo_F)}
    C.write_json(C.OUT_MET / "b12_two_step.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
