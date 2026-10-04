"""Side-by-side comparison requested after inspection: the pre-registered fused model F vs F + a brightness head.

  A  = F: mean(S head, DINO head)                          (pre-registered, unchanged)
  B  = F + C: mean(S head, DINO head, brightness head C)   (exploratory, added after looking at the images)

Brightness head C: raw (un-normalised) BSE grey of the graphite background, of the SiOx particles ("dots"), and
the contrast ratio (SiOx - graphite) / (graphite - pore black level), from the delivered label maps; in-fold robust
z (clipped at |z| = 3) -> shrinkage LDA with equal priors, like the S head.

Both are evaluated identically: leave-one-sample-out (LOO), leave-one-height-group-out (session control, LOGO)
and a permutation null with the whole procedure (incl. the DINO falsifier choice) inside each shuffle.

Writes outputs/metrics/bid_brightness.csv and outputs/metrics/bid_compare.json.

Usage: python -m src.bid_compare [--n 200]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from . import config as C
from .bid_model import bal_acc, head_S, ledger, load, loo, procedure, summarise, texton_fold_hists
from .preprocess import read_half

BRIGHT = ["bse_background_grey", "bse_dots_grey", "bse_contrast_ratio"]
_W = {}


def brightness(item):
    (batch, sid), paths = item
    lab = np.load(C.proc_dir(sid) / "final_labels.npy")
    raw = read_half(paths["BSE"])
    h, w = min(raw.shape[0], lab.shape[0]), min(raw.shape[1], lab.shape[1])
    r, l = raw[:h, :w], lab[:h, :w]
    g, s, p = np.median(r[l == C.GRAPHITE]), np.median(r[l == C.SIOX]), np.percentile(r[l == C.PORE], 25)
    return {"batch": batch, "sample_id": sid, "bse_background_grey": round(float(g), 3),
            "bse_dots_grey": round(float(s), 3), "bse_pore_black_level": round(float(p), 3),
            "bse_contrast_ratio": round(float((s - g) / max(g - p, 1e-6)), 4)}


def load_all():
    d = load()
    b = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_brightness.csv")}
    d["C"] = np.array([[float(b[s][k]) for k in BRIGHT] for s in d["sids"]])
    return d


def both(d, y, folds=None, texton_cache=None):
    res = procedure(d, y, folds, texton_cache)
    pC = loo(lambda tr, te: head_S(d["C"], y, tr, te), y, folds)
    return {"A": res["F"], "B": (res["S"] + res["D"] + pC) / 3, "C_alone": pC, "dino": res["dino_head"]}


def _perm(args):
    seed, hists = args
    if "d" not in _W:
        _W["d"] = load_all()
    d = _W["d"]
    yp = np.random.default_rng(seed).permutation(d["y"])
    r = both(d, yp, texton_cache=hists)
    return bal_acc(yp, r["A"].argmax(1)), bal_acc(yp, r["B"].argmax(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(brightness, C.discover().items()))
    C.write_csv(C.OUT_MET / "bid_brightness.csv", rows)

    d = load_all()
    y = d["y"]
    hists = texton_fold_hists(d, [np.array([i]) for i in range(len(y))])
    res = both(d, y, texton_cache=hists)
    groups = d["groups"]
    res_g = both(d, y, [np.flatnonzero(groups == g) for g in np.unique(groups)])
    print(f"DINO head chosen by the falsifier: {res['dino']}", flush=True)

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    null_A, null_B = [], []
    with ProcessPoolExecutor(C.WORKERS) as ex:
        for i, (a, b) in enumerate(ex.map(_perm, [(5000 + i, hists) for i in range(args.n)])):
            null_A.append(a); null_B.append(b)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{args.n} permutations", flush=True)

    out = {}
    for key, name, null in (("A", "A: original (segmentation + DINO)", null_A),
                            ("B", "B: original + brightness/contrast head", null_B)):
        obs = bal_acc(y, res[key].argmax(1))
        null = np.array(null)
        out[key] = {"name": name, "LOO": summarise(y, res[key]), "LOGO_session": summarise(y, res_g[key]),
                    "permutation": {"null_mean": round(float(null.mean()), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                                    "null_p99": round(float(np.percentile(null, 99)), 3),
                                    "p_value": round(float((np.sum(null >= obs) + 1) / (len(null) + 1)), 4), "n": len(null)}}
    out["C_alone"] = {"LOO": summarise(y, res["C_alone"]), "LOGO_session": summarise(y, res_g["C_alone"])}
    C.write_json(C.OUT_MET / "bid_compare.json", out)
    ledger({"config": "F+C_exploratory", "kind": "LOO", "balanced_accuracy": out["B"]["LOO"]["balanced_accuracy"],
            "detail": json.dumps(out["B"]["LOO"]["recall"])})
    for k in ("A", "B"):
        o = out[k]
        print(f"{o['name']}\n  LOO  {o['LOO']['balanced_accuracy']:.3f} {o['LOO']['recall']}\n"
              f"  LOGO {o['LOGO_session']['balanced_accuracy']:.3f} {o['LOGO_session']['recall']}\n"
              f"  permutation p {o['permutation']['p_value']} (null 95th pct {o['permutation']['null_p95']})")
    print(f"brightness head alone: LOO {out['C_alone']['LOO']['balanced_accuracy']:.3f}, LOGO {out['C_alone']['LOGO_session']['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
