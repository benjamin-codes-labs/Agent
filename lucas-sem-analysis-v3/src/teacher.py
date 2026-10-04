"""Stage 3b: teacher = LightGBM pixel classifier trained on automatic labels, with self-training.

Training labels = rule seeds (src.seeds) + AI-annotated superpixels, train split (src.ai_labels).
Self-training: after each fit, pixels predicted with max-prob >= 0.9 that agree with a 5x5 majority filter
are added as labels and the model is refit (2 rounds).

Subcommands
  train      fit + self-train on all samples, predict every pixel, post-process, run sanity gates
  ablation   BSE vs BSE+Inlens vs BSE+Inlens+SE2 (round-0 labels, stride-3 evaluation)
  stability  refit on 3 random 20-sample subsets x 2 seeds with the final labels (stride-3 evaluation)

Writes data/processed/<sid>/teacher_labels.npy, teacher_maxp.npy, models/teacher.txt,
outputs/metrics/teacher_fractions.csv, ablation.csv, stability.csv.

Usage: python -m src.teacher train|ablation|stability [--features bse_inlens]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
from scipy import ndimage as ndi

from . import config as C
from .physics import constrain_siox, fill_siox_holes

FEATURE_SETS = {"bse": ["BSE"], "bse_inlens": ["BSE", "Inlens"], "all": ["BSE", "Inlens", "SE2"]}
PARAMS = dict(objective="multiclass", num_class=4, learning_rate=0.12, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, num_threads=16,
              verbose=-1)
AI_WEIGHT = 4.0          # AI-labelled superpixels are few but cover the hard regions
MIN_OBJECT_PX = int(round(0.1 / C.PX_UM2))   # 0.1 um^2


# ---------------------------------------------------------------- data

def load_features(sid, fset):
    return np.concatenate([np.load(C.proc_dir(sid) / f"feat_{d}.npy", mmap_mode="r") for d in FEATURE_SETS[fset]], -1)


def base_labels(sid, use_ai=True):
    """Rule seeds + AI labels (train split). Returns (labels, is_ai).

    With AI labels, the rule-based CBD seeds are dropped: on blind seeded superpixels the annotator agreed
    with the pore / graphite / SiOx rules 100 % of the time but with the CBD rule only 19/41 times (the rule
    also fires on pore walls and thin graphite slivers), so CBD is learned from the AI labels only.
    """
    lab = np.load(C.proc_dir(sid) / "seeds.npy").copy()
    is_ai = np.zeros(lab.shape, bool)
    p = C.proc_dir(sid) / "ai_labels.npy"
    if use_ai and p.exists():
        lab[lab == C.CBD] = C.EXCLUDED
        ai = np.load(p)
        m = (ai != C.EXCLUDED) & (lab == C.EXCLUDED)
        lab[m], is_ai[m] = ai[m], True
    return lab, is_ai


def sample_pixels(labels, is_ai, per_class, rng):
    idx, w = [], []
    flat, ai = labels.ravel(), is_ai.ravel()
    for k in range(4):
        cand = np.flatnonzero((flat == k) & ~ai)
        if len(cand):
            idx.append(rng.choice(cand, min(per_class, len(cand)), replace=False))
            w.append(np.ones(len(idx[-1]), np.float32))
        cand = np.flatnonzero((flat == k) & ai)
        if len(cand):
            idx.append(rng.choice(cand, min(per_class // 4, len(cand)), replace=False))
            w.append(np.full(len(idx[-1]), AI_WEIGHT, np.float32))
    idx = np.concatenate(idx)
    return idx, flat[idx], np.concatenate(w)


def build_training(sids, fset, label_fn, per_class, rng):
    X, y, w = [], [], []
    for sid in sids:
        labels, is_ai = label_fn(sid)
        idx, yi, wi = sample_pixels(labels, is_ai, per_class, rng)
        F = load_features(sid, fset)
        X.append(np.asarray(F.reshape(-1, F.shape[-1])[np.sort(idx)], dtype=np.float32))
        order = np.argsort(idx)
        y.append(yi[order]); w.append(wi[order])
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def fit(X, y, w, seed=0, rounds=300):
    rng = np.random.default_rng(seed)
    val = rng.random(len(y)) < 0.1
    params = dict(PARAMS, seed=seed)
    dtr = lgb.Dataset(X[~val], y[~val], weight=w[~val])
    dva = lgb.Dataset(X[val], y[val], weight=w[val], reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def predict_proba(model, sid, fset, stride=1, chunk=600_000):
    F = load_features(sid, fset)
    if stride > 1:
        F = F[::stride, ::stride]
    h, w, nf = F.shape
    flat = F.reshape(-1, nf)
    out = np.empty((h * w, 4), np.float32)
    for i in range(0, h * w, chunk):
        out[i:i + chunk] = model.predict(np.asarray(flat[i:i + chunk], dtype=np.float32), num_threads=16)
    bse = C.load_channels(sid, ["BSE"])["BSE"]
    return constrain_siox(out.reshape(h, w, 4), bse, C.load_exclude(sid), stride=stride)


# ---------------------------------------------------------------- post-processing / metrics

def majority5(pred):
    onehot = np.stack([ndi.uniform_filter((pred == k).astype(np.float32), 5) for k in range(4)], -1)
    return onehot.argmax(-1)


def clean(pred, exclude):
    """Remove objects < 0.1 um^2 (refilled from the nearest remaining pixel) and apply the exclusion mask."""
    out = pred.astype(np.uint8).copy()
    drop = np.zeros(out.shape, bool)
    for k in range(4):
        lab, n = ndi.label(out == k)
        if n:
            small = np.flatnonzero(np.bincount(lab.ravel()) < MIN_OBJECT_PX)
            drop |= np.isin(lab, small[small > 0])
    if drop.any():
        _, (iy, ix) = ndi.distance_transform_edt(drop, return_indices=True)
        out = out[iy, ix]
    out[exclude] = C.EXCLUDED
    return fill_siox_holes(out)


def siox_thrlike(labels, stride=1):
    """SiOx % after the same filter the baseline threshold used (opening r = 0.15 um, objects >= 2 um^2),
    so the SiOx gate compares like with like; the unfiltered SiOx % also counts particles < 2 um^2."""
    from skimage.morphology import disk, remove_small_objects
    si = ndi.binary_opening(labels == C.SIOX, structure=disk(max(1, round(3 / stride))))
    si = remove_small_objects(si, max_size=int(2 / C.PX_UM2 / stride ** 2) - 1)
    valid = labels != C.EXCLUDED
    return 100 * float(si[valid].mean())


def fractions(labels, exclude=None):
    valid = labels != C.EXCLUDED if exclude is None else ~exclude & (labels != C.EXCLUDED)
    n = valid.sum()
    return {c: 100 * float((labels[valid] == k).sum()) / n for k, c in enumerate(C.CLASSES)}


def gates(fr):
    """Sanity gates against the baseline. fr = {sid: {class: pct}}. Returns (all_ok, lines)."""
    base = C.baseline()
    lines, ok = [], True
    for sid, f in fr.items():
        b = base[sid]
        d_si = f.get("SiOx_thrlike", f["SiOx"]) - b["siox_thr"]
        si_ok = abs(d_si) <= 1.5 or sid == "epqdaau9"
        pore_ok = f["pore"] >= b["pore_thr"] - 0.5
        ok &= si_ok and pore_ok
        if not (si_ok and pore_ok):
            lines.append(f"  FAIL {sid}: SiOx (thr-like) {f.get('SiOx_thrlike', f['SiOx']):.1f} vs thr {b['siox_thr']} ({d_si:+.1f}); pore {f['pore']:.1f} vs thr {b['pore_thr']}")
    by_pore = sorted(fr, key=lambda s: fr[s]["pore"])
    by_si = sorted(fr, key=lambda s: fr[s]["SiOx"])
    checks = {
        "f1vzngrs among 3 least porous": "f1vzngrs" in by_pore[:3],
        "hzumfsms & 0grcilhi among 5 most porous": {"hzumfsms", "0grcilhi"} <= set(by_pore[-5:]),
        "5n1q8atc & 4ih2ggld among 3 highest SiOx": {"5n1q8atc", "4ih2ggld"} <= set(by_si[-3:]),
        "avn74qx1, hawkfj64, r17byphk among 6 lowest SiOx": {"avn74qx1", "hawkfj64", "r17byphk"} <= set(by_si[:6]),
    }
    for name, passed in checks.items():
        lines.append(f"  {'ok  ' if passed else 'FAIL'} ranking: {name}")
        ok &= passed
    return ok, lines


# ---------------------------------------------------------------- commands

def cmd_train(args):
    sids = [s for _, s in C.sample_ids()]
    rng = np.random.default_rng(0)
    extra = {}
    for rnd in range(3):
        t = time.time()
        X, y, w = build_training(sids, args.features, lambda sid: _with_extra(sid, extra, not args.no_ai), args.per_class, rng)
        model = fit(X, y, w)
        print(f"round {rnd}: {len(y)} training px {np.bincount(y, minlength=4).tolist()}, "
              f"{model.best_iteration} trees, {time.time() - t:.0f}s", flush=True)
        fr = {}
        # Self-training rounds only need labels at the pixels that get sampled for training, so they are
        # predicted on a stride-2 grid (4x faster); the final round predicts every pixel.
        stride = 1 if rnd == 2 else 2
        for sid in sids:
            proba = predict_proba(model, sid, args.features, stride)
            pred, maxp = proba.argmax(-1), proba.max(-1)
            exclude = C.load_exclude(sid)
            if rnd < 2:
                confident = (maxp >= 0.9) & (pred == majority5(pred)) & ~exclude[::stride, ::stride]
                full = np.full(exclude.shape, C.EXCLUDED, np.uint8)
                full[::stride, ::stride] = np.where(confident, pred, C.EXCLUDED)
                extra[sid] = full
            final = clean(pred, exclude[::stride, ::stride])
            fr[sid] = fractions(final)
            fr[sid]["SiOx_thrlike"] = siox_thrlike(final, stride)
            if rnd == 2:
                np.save(C.proc_dir(sid) / f"teacher{args.tag}_labels.npy", final)
                np.save(C.proc_dir(sid) / f"teacher{args.tag}_maxp.npy", (maxp * 255).astype(np.uint8))
        print(f"round {rnd}: predicted {len(sids)} samples at stride {stride} ({time.time() - t:.0f}s)", flush=True)
        print(f"round {rnd} mean fractions: " + ", ".join(f"{c} {np.mean([f[c] for f in fr.values()]):.1f}%" for c in C.CLASSES), flush=True)
        ok, lines = gates(fr)
        print(f"round {rnd} gates: {'PASS' if ok else 'FAIL'}")
        print("\n".join(lines), flush=True)
    model.save_model(str(C.MODELS / f"teacher{args.tag}.txt"))
    base = C.baseline()
    rows = [{"sample_id": s, **{f"{c}_pct": round(v, 2) for c, v in fr[s].items()},  # incl. SiOx_thrlike
             "pore_thr": base[s]["pore_thr"], "pore_vis": base[s]["pore_vis"],
             "siox_thr": base[s]["siox_thr"], "siox_vis": base[s]["siox_vis"]} for s in sids]
    C.write_csv(C.OUT_MET / f"teacher{args.tag}_fractions.csv", rows)
    imp = model.feature_importance("gain")
    print("teacher saved; top features by gain:", np.argsort(imp)[::-1][:10].tolist())


def _with_extra(sid, extra, use_ai=True):
    labels, is_ai = base_labels(sid, use_ai)
    if sid in extra:
        m = (labels == C.EXCLUDED) & (extra[sid] != C.EXCLUDED)
        labels = labels.copy()
        labels[m] = extra[sid][m]
    return labels, is_ai


def eval_fractions(model, sids, fset, stride=3):
    fr = {}
    for sid in sids:
        pred = predict_proba(model, sid, fset, stride).argmax(-1)
        ex = C.load_exclude(sid)[::stride, ::stride]
        pred = pred.astype(np.uint8)
        pred[ex] = C.EXCLUDED
        fr[sid] = fractions(pred)
        fr[sid]["SiOx_thrlike"] = siox_thrlike(pred, stride)
    return fr


def cmd_ablation(args):
    sids = [s for _, s in C.sample_ids()]
    base = C.baseline()
    rows = []
    for fset in FEATURE_SETS:
        t = time.time()
        X, y, w = build_training(sids, fset, base_labels, args.per_class // 2, np.random.default_rng(1))
        model = fit(X, y, w, seed=1)
        fr = eval_fractions(model, sids, fset)
        ok, lines = gates(fr)
        si_err = np.mean([abs(fr[s]["SiOx"] - base[s]["siox_thr"]) for s in sids if s != "epqdaau9"])
        pore_below = sum(fr[s]["pore"] < base[s]["pore_thr"] - 0.5 for s in sids)
        rows.append({"features": fset, "n_features": X.shape[1], "SiOx_mean_abs_err_vs_thr": round(si_err, 2),
                     "samples_pore_below_thr": pore_below, "gates": "PASS" if ok else "FAIL",
                     **{f"mean_{c}_pct": round(np.mean([fr[s][c] for s in sids]), 2) for c in C.CLASSES},
                     "minutes": round((time.time() - t) / 60, 1)})
        print(rows[-1], flush=True)
        print("\n".join(lines), flush=True)
    C.write_csv(C.OUT_MET / "ablation.csv", rows)


def cmd_stability(args):
    sids = [s for _, s in C.sample_ids()]
    runs = []
    for subset in range(3):
        rng = np.random.default_rng(100 + subset)
        train = list(rng.choice(sids, 20, replace=False))
        for seed in range(2):
            def labels_fn(sid):
                lab = np.load(C.proc_dir(sid) / "teacher_labels.npy")
                maxp = np.load(C.proc_dir(sid) / "teacher_maxp.npy")
                lab = np.where(maxp >= 230, lab, C.EXCLUDED).astype(np.uint8)   # confident teacher pixels
                return lab, np.zeros(lab.shape, bool)
            X, y, w = build_training(train, args.features, labels_fn, args.per_class // 2, np.random.default_rng(seed))
            model = fit(X, y, w, seed=seed)
            runs.append(eval_fractions(model, sids, args.features))
            print(f"stability run subset {subset} seed {seed} done", flush=True)
    rows = []
    for s in sids:
        row = {"sample_id": s}
        for c in C.CLASSES:
            v = [r[s][c] for r in runs]
            row[f"{c}_mean"], row[f"{c}_sd"] = round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)), 2)
        rows.append(row)
    C.write_csv(C.OUT_MET / "stability.csv", rows)
    for c in C.CLASSES:
        sd = [r[f"{c}_sd"] for r in rows]
        print(f"{c}: stability SD mean {np.mean(sd):.2f} pp, max {np.max(sd):.2f} pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "ablation", "stability"])
    ap.add_argument("--features", default="bse_inlens", choices=list(FEATURE_SETS))
    ap.add_argument("--per-class", type=int, default=16000)
    ap.add_argument("--no-ai", action="store_true", help="train on rule seeds only (comparison run)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _seedsonly")
    args = ap.parse_args()
    {"train": cmd_train, "ablation": cmd_ablation, "stability": cmd_stability}[args.cmd](args)


if __name__ == "__main__":
    main()
