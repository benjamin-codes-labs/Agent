"""Batch-ID stage, step 3: classifiers and the validation harness (docs/route_to_80 synthesis, A2-A5).

Two pre-registered scored configurations, no others:
  S      the 5 pre-registered label-map features -> in-fold robust z (clipped at |z| = 3) -> shrinkage LDA
         with equal class priors.
  F      mean of the S probabilities and one DINO head. The DINO head is chosen INSIDE the procedure by the
         synthesis's falsifier: phase-conditioned token means (head R: one linear head per region type
         SiOx / pore / CBD, votes averaged) must beat the whole-image mean token (head G) by >= 2 correct
         samples in LOO; otherwise a texton histogram head (T: k-means K = 24 on tokens, fitted in-fold,
         Hellinger -> linear) is used. Every linear DINO head = StandardScaler -> PCA(k) -> logistic
         regression (balanced), averaged over a fixed grid k in {4, 8} x C in {0.03, 0.1, 0.3} (never selected).

Validation: leave-one-sample-out (LOO) balanced accuracy and per-class recall; permutation null with the
whole procedure (including the falsifier choice) inside each shuffle; leave-one-height-group-out (session
control, LOGO); nuisance-only classifier on the 14 session probes (must score <= 0.45); the 2080-px triplet
(one sample per batch imaged together). Every scored run is appended to outputs/metrics/look_ledger.csv.

Usage: python -m src.bid_model cache | score | null [--n 300] | fit
"""
import argparse
import datetime
import hashlib
import json
import pickle
import warnings

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config as C
from .bid_features import S_FEATURES

warnings.filterwarnings("ignore")
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
GRID = [(k, c) for k in (4, 8) for c in (0.03, 0.1, 0.3)]
REGIONS = {"SiOx": C.SIOX, "pore": C.PORE, "CBD": C.CBD}
CACHE = C.PROC / "bid_cache.npz"
LEDGER = C.OUT_MET / "look_ledger.csv"
TRIPLET = ["ffwubibz", "r17byphk", "cfe5vt7s"]
K_TEXTON = 24


# ---------------------------------------------------------------- data

def token_phase_weights(lab, hp, wp, p=14):
    """Per-token fraction of each class (and of valid pixels) in its 14 x 14 pixel block."""
    blk = lab[:hp * p, :wp * p].reshape(hp, p, wp, p)
    frac = np.stack([(blk == k).mean((1, 3)) for k in range(4)], -1)
    return frac                                  # excluded share = 1 - frac.sum(-1)


def cmd_cache(args):
    sids = [s for _, s in C.sample_ids()]
    glob_m, phase_m, tokens = [], [], []
    rng = np.random.default_rng(0)
    for sid in sids:
        lab = np.load(C.proc_dir(sid) / f"{args.labels}_labels.npy")
        t = np.concatenate([np.load(C.proc_dir(sid) / f"dino_{d}_50nm.npy").astype(np.float32) for d in ("BSE", "Inlens")], -1)
        hp, wp, _ = t.shape
        w = token_phase_weights(lab, hp, wp)
        valid = w.sum(-1)
        glob_m.append((t * valid[..., None]).sum((0, 1)) / valid.sum())
        phase_m.append(np.stack([(t * w[..., k, None]).sum((0, 1)) / max(w[..., k].sum(), 1e-6) for k in range(4)]))
        flat = t[valid > 0.5]
        tokens.append(flat[rng.choice(len(flat), min(10000, len(flat)), replace=False)])
    np.savez_compressed(CACHE, sids=np.array(sids), glob=np.stack(glob_m), phase=np.stack(phase_m),
                        tokens=np.stack(tokens), labels_source=args.labels)
    print(f"cached DINO means for {len(sids)} samples from {args.labels} labels -> {CACHE}")


def load():
    feats = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_features.csv")}
    probes = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_session_probes.csv")}
    qc = {r["sample_id"]: r for r in C.read_csv(C.QC)}
    z = np.load(CACHE)
    sids = list(z["sids"])
    y = np.array([BATCHES.index(feats[s]["batch"]) for s in sids])
    S = np.array([[float(feats[s][k]) for k in S_FEATURES] for s in sids])
    P = np.array([[float(v) for k, v in probes[s].items() if k not in ("batch", "sample_id")] for s in sids])
    groups = np.array([qc[s]["height_group"] for s in sids])
    return {"sids": sids, "y": y, "S": S, "P": P, "groups": groups,
            "glob": z["glob"], "phase": z["phase"], "tokens": z["tokens"]}


# ---------------------------------------------------------------- heads (fit on train idx, predict test idx)

def head_S(X, y, tr, te):
    med = np.median(X[tr], 0)
    iqr = np.subtract(*np.percentile(X[tr], [75, 25], 0)) / 1.349
    iqr[iqr == 0] = 1
    Z = np.clip((X - med) / iqr, -3, 3)
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(3) / 3).fit(Z[tr], y[tr])
    return m.predict_proba(Z[te])


def linear_grid(X, y, tr, te):
    out = []
    for k, c in GRID:
        sc = StandardScaler().fit(X[tr])
        pca = PCA(min(k, len(tr) - 1), random_state=0).fit(sc.transform(X[tr]))
        f = lambda a: pca.transform(sc.transform(a))
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(f(X[tr]), y[tr])
        out.append(lr.predict_proba(f(X[te])))
    return np.mean(out, 0)


def head_G(d, y, tr, te):
    return linear_grid(d["glob"], y, tr, te)


def head_R(d, y, tr, te):
    return np.mean([linear_grid(d["phase"][:, k], y, tr, te) for k in REGIONS.values()], 0)


def head_T(d, y, tr, te):
    km = MiniBatchKMeans(K_TEXTON, random_state=0, n_init=3, batch_size=4096).fit(d["tokens"][tr].reshape(-1, d["tokens"].shape[-1]))
    H = np.stack([np.bincount(km.predict(t), minlength=K_TEXTON) / len(t) for t in d["tokens"]])
    return linear_grid(np.sqrt(H), y, tr, te)


def texton_fold_hists(d, folds):
    """Per-fold texton histograms. The vocabulary is fitted on the fold's training tokens without labels, so
    it is identical under label permutation and can be computed once and reused by the permutation null."""
    out = []
    n = len(d["tokens"])
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        km = MiniBatchKMeans(K_TEXTON, random_state=0, n_init=3, batch_size=4096).fit(d["tokens"][tr].reshape(-1, d["tokens"].shape[-1]))
        out.append(np.sqrt(np.stack([np.bincount(km.predict(t), minlength=K_TEXTON) / len(t) for t in d["tokens"]])))
    return out


def loo(fn, y, folds=None):
    n = len(y)
    folds = folds if folds is not None else [np.array([i]) for i in range(n)]
    proba = np.zeros((n, 3))
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        proba[te] = fn(tr, te)
    return proba


def bal_acc(y, pred):
    return float(np.mean([(pred[y == k] == k).mean() for k in range(3)]))


def procedure(d, y, folds=None, texton_cache=None):
    """The full pre-registered procedure for both configs. Returns dict of LOO probabilities + choice."""
    pS = loo(lambda tr, te: head_S(d["S"], y, tr, te), y, folds)
    pG = loo(lambda tr, te: head_G(d, y, tr, te), y, folds)
    pR = loo(lambda tr, te: head_R(d, y, tr, te), y, folds)
    passes = (pR.argmax(1) == y).sum() >= (pG.argmax(1) == y).sum() + 2
    if passes:
        pD, dino = pR, "R (phase-conditioned token means)"
    elif texton_cache is not None:                      # per-fold histograms (label-free), reused by the null
        n = len(y)
        fl = folds if folds is not None else [np.array([i]) for i in range(n)]
        pD = np.zeros((n, 3))
        for H, te in zip(texton_cache, fl):
            pD[te] = linear_grid(H, y, np.setdiff1d(np.arange(n), te), te)
        dino = "T (texton histogram)"
    else:
        pD = loo(lambda tr, te: head_T(d, y, tr, te), y, folds)
        dino = "T (texton histogram)"
    return {"S": pS, "G": pG, "R": pR, "D": pD, "F": (pS + pD) / 2, "dino_head": dino, "falsifier_pass": bool(passes)}


def code_hash():
    h = hashlib.sha256()
    for p in sorted((C.ROOT / "src").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def ledger(entry):
    rows = C.read_csv(LEDGER) if LEDGER.exists() else []
    rows.append({"utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "code_hash": code_hash(), **entry})
    C.write_csv(LEDGER, rows, fields=list(rows[-1]))


def summarise(y, proba):
    pred = proba.argmax(1)
    rec = {BATCHES[k]: f"{(pred[y == k] == k).sum()}/{(y == k).sum()}" for k in range(3)}
    conf = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    return {"balanced_accuracy": round(bal_acc(y, pred), 3), "recall": rec, "confusion_rows_true": conf.tolist()}


# ---------------------------------------------------------------- commands

def cmd_score(args):
    d = load()
    y = d["y"]
    res = procedure(d, y)
    out = {"n": len(y), "dino_head_chosen": res["dino_head"], "falsifier_pass": res["falsifier_pass"],
           "falsifier_correct": {"G": int((res["G"].argmax(1) == y).sum()), "R": int((res["R"].argmax(1) == y).sum())}}
    for cfg in ("S", "F"):
        out[f"LOO_{cfg}"] = summarise(y, res[cfg])
    # session control: leave-one-height-group-out
    groups = d["groups"]
    folds = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    res_g = procedure(d, y, folds)
    for cfg in ("S", "F"):
        out[f"LOGO_{cfg}"] = summarise(y, res_g[cfg])
    out["triplet_2080px_LOGO_F"] = {s: BATCHES[int(res_g["F"][d["sids"].index(s)].argmax())] + f" (true {BATCHES[y[d['sids'].index(s)]]})" for s in TRIPLET}
    # nuisance-only classifier: session probes alone must not predict batch
    pN = loo(lambda tr, te: linear_grid(d["P"], y, tr, te), y)
    out["nuisance_only_LOO"] = summarise(y, pN)
    # pair-out: the two 2316-px Batch_1 samples (one session, the whole Batch_1 SiOx signal) held out together
    out["per_sample_LOO"] = {s: {"true": BATCHES[y[i]], "pred_S": BATCHES[int(res["S"][i].argmax())],
                                 "pred_F": BATCHES[int(res["F"][i].argmax())],
                                 "p_F": [round(float(v), 3) for v in res["F"][i]]} for i, s in enumerate(d["sids"])}
    C.write_json(C.OUT_MET / "bid_scores.json", out)
    for cfg in ("S", "F"):
        ledger({"config": cfg, "kind": "LOO", "balanced_accuracy": out[f"LOO_{cfg}"]["balanced_accuracy"],
                "detail": json.dumps(out[f"LOO_{cfg}"]["recall"])})
    print(json.dumps({k: v for k, v in out.items() if k != "per_sample_LOO"}, indent=1))


_WORKER = {}


def _perm(args):
    seed, hists = args
    if "d" not in _WORKER:
        _WORKER["d"] = load()
    d = _WORKER["d"]
    yp = np.random.default_rng(seed).permutation(d["y"])
    res = procedure(d, yp, texton_cache=hists)
    return bal_acc(yp, res["S"].argmax(1)), bal_acc(yp, res["F"].argmax(1))


def cmd_null(args):
    from concurrent.futures import ProcessPoolExecutor
    d = load()
    y = d["y"]
    obs = json.load(open(C.OUT_MET / "bid_scores.json"))
    hists = texton_fold_hists(d, [np.array([i]) for i in range(len(y))])
    print("per-fold texton histograms ready", flush=True)
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"                      # inherited by the spawned workers: one thread each
    null_S, null_F = [], []
    with ProcessPoolExecutor(C.WORKERS) as ex:
        for i, (s_, f_) in enumerate(ex.map(_perm, [(1000 + i, hists) for i in range(args.n)])):
            null_S.append(s_); null_F.append(f_)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{args.n} permutations", flush=True)
    out = {}
    for cfg, null in (("S", null_S), ("F", null_F)):
        o = obs[f"LOO_{cfg}"]["balanced_accuracy"]
        null = np.array(null)
        out[cfg] = {"observed": o, "null_mean": round(float(null.mean()), 3), "null_p95": round(float(np.percentile(null, 95)), 3),
                    "null_p99": round(float(np.percentile(null, 99)), 3), "p_value": round(float((np.sum(null >= o) + 1) / (len(null) + 1)), 4),
                    "n_permutations": len(null)}
    np.savez(C.OUT_MET / "bid_null.npz", S=np.array(null_S), F=np.array(null_F))
    C.write_json(C.OUT_MET / "bid_null.json", out)
    print(json.dumps(out, indent=1))


def fit_model(exclude=()):
    """Fit S + the chosen DINO head on all samples except `exclude` (used by the demo hold-out mode)."""
    d = load()
    keep = np.array([i for i, s in enumerate(d["sids"]) if s not in set(exclude)])
    d = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(d["sids"]) else v) for k, v in d.items()}
    d["sids"] = [d["sids"][i] for i in keep] if isinstance(d["sids"], list) else list(np.array(d["sids"])[keep])
    return _fit(d)


def cmd_fit(args):
    """Fit both configs on all 31 samples for predicting new images."""
    model = fit_model()
    with open(C.MODELS / "bid_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"fitted S + DINO head {model['dino_head']} on {len(model['trained_on'])} samples -> models/bid_model.pkl")


def _fit(d):
    y = d["y"]
    obs = json.load(open(C.OUT_MET / "bid_scores.json"))
    allidx = np.arange(len(y))
    med = np.median(d["S"], 0)
    iqr = np.subtract(*np.percentile(d["S"], [75, 25], 0)) / 1.349
    iqr[iqr == 0] = 1
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(3) / 3).fit(np.clip((d["S"] - med) / iqr, -3, 3), y)
    dino = obs["dino_head_chosen"][0]
    heads = []
    sources = [d["phase"][:, k] for k in REGIONS.values()] if dino == "R" else [None]
    if dino == "T":
        km = MiniBatchKMeans(K_TEXTON, random_state=0, n_init=3, batch_size=4096).fit(d["tokens"].reshape(-1, d["tokens"].shape[-1]))
        H = np.sqrt(np.stack([np.bincount(km.predict(t), minlength=K_TEXTON) / len(t) for t in d["tokens"]]))
        sources = [H]
    for X in sources:
        for k, c in GRID:
            sc = StandardScaler().fit(X)
            pca = PCA(k, random_state=0).fit(sc.transform(X))
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(pca.transform(sc.transform(X)), y)
            heads.append((sc, pca, lr))
    return {"S_features": S_FEATURES, "S_median": med, "S_iqr": iqr, "lda": lda, "dino_head": dino,
            "regions": list(REGIONS) if dino == "R" else None, "kmeans": km if dino == "T" else None,
            "dino_heads": heads, "grid": GRID, "batches": BATCHES, "trained_on": list(d["sids"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "score", "null", "fit"])
    ap.add_argument("--labels", default="final")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    {"cache": cmd_cache, "score": cmd_score, "null": cmd_null, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
