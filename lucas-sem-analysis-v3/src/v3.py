"""v3 batch decision = v2 + the teammate's v3 additions.

Kept from v2: two sides that must agree (the teammate's imaging side vs our material side), answers "Batch_k",
"Batch_1 or Batch_2" or "unsure", confidence tiers, prediction set, track record, text-only explanations.
Added (teammate's design):
  - training-image guard and known-location matcher (src/known_spot.py), applied before the models;
  - tile texture model (src/texture_model.py). In the teammate's design it is averaged with the fingerprint model
    on the imaging side; in our nested test that lowered accuracy (80 % vs 85 % when answering), so by default it is
    reported in facts.json but does not vote (`--with-texture` restores it);
  - calibration: every model's probabilities are temperature-scaled on held-out predictions;
  - GET4 material-range rule (src/get4.py error bars on our pore / carbon / SiOx fractions): turns a
    "Batch_1 or Batch_2" answer into one batch when the fractions, with their 95 % error bars, fit only that
    batch's range;
  - High tier = the average calibrated probability is above the level at which held-out answers were >= 90 % right;
  - forced mode: fingerprint decides "Batch_3 or not"; then the range rule, else a Batch_1-vs-Batch_2 fingerprint
    specialist.
Everything that is chosen from data (calibration temperatures, the High threshold, batch ranges) is chosen inside
each fold without the held-out location (nested). The DINOv2 texton vocabulary is fitted per outer fold without the
held-out location (label-free) and shared by the inner folds.

  prepare   GET4 error bars for the 31 locations -> outputs/metrics/v3_material_se.csv; per-fold texton
            histograms -> data/processed/v3_texton_hists.npz
  evaluate  nested leave-one-location-out and leave-one-session-out -> outputs/metrics/v3_eval.json
  fit       final models on all 31 locations -> models/v3_model.pkl

Usage: python -m src.v3 prepare ; python -m src.v3 evaluate [--with-texture] ; python -m src.v3 fit
"""
import argparse
import json
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config as C
from . import texture_model as TX
from .bid_model import BATCHES, head_S, ledger, linear_grid, load as load_mat, texton_fold_hists
from .fingerprint_model import VIEWS as FP_VIEWS, fit_view as fp_fit_view, training_table as fp_table, view_proba as fp_view_proba
from .validate import wilson

warnings.filterwarnings("ignore")
SE_CSV = C.OUT_MET / "v3_material_se.csv"
HISTS = C.PROC / "v3_texton_hists.npz"
EVAL = C.OUT_MET / "v3_eval.json"
MODEL = C.MODELS / "v3_model.pkl"
PHASES3 = ("pore", "carbon", "SiOx")
TX_VIEWS = TX.VIEWS
TARGET = 0.90
TGRID = np.geomspace(0.05, 20, 81)
USE_TEXTURE = False     # --with-texture: imaging side = fingerprint + texture averaged (the teammate's spec). It lowered
                        # held-out accuracy (outputs/metrics/v3_eval_with_texture.json), so the texture model is reported
                        # in facts.json but does not vote.


# ---------------------------------------------------------------- material error bars (GET4) and the range rule

def material_se(lab):
    """Pore / carbon / SiOx % of valid pixels, with GET4's image standard error (pp)."""
    from .get4 import analyse_phase
    l = lab[8:-8, 8:-8]
    valid = l != C.EXCLUDED
    h, w = l.shape
    max_lag = min(h, w) // 2
    sizes = np.unique(np.geomspace(4, min(h, w) // 2, 16).astype(int))
    out = {}
    for name, m in (("pore", l == 0), ("carbon", (l == 1) | (l == 3)), ("SiOx", l == 2)):
        r = analyse_phase(m & valid, max_lag, None, sizes, C.NM_PER_PX / 1000)
        out[name] = {"pct": round(100 * float(m[valid].mean()), 3), "se_pp": round(100 * float(r["se_image"]), 3)}
    return out


def batch_ranges(ms_by_sid, batch_by_sid):
    return {b: {ph: [min(ms_by_sid[s][ph]["pct"] for s in ms_by_sid if batch_by_sid[s] == b),
                     max(ms_by_sid[s][ph]["pct"] for s in ms_by_sid if batch_by_sid[s] == b)] for ph in PHASES3}
            for b in BATCHES}


def range_rule(ms, ranges):
    """A phase votes for a batch when its 95 % interval overlaps only that batch's training range. The rule answers
    Batch_1 or Batch_2 when at least one phase votes and all votes agree."""
    votes, detail = [], []
    for ph in PHASES3:
        pct, se = ms[ph]["pct"], ms[ph]["se_pp"]
        lo, hi = pct - 1.96 * se, pct + 1.96 * se
        fits = [b for b in BATCHES if lo <= ranges[b][ph][1] and hi >= ranges[b][ph][0]]
        detail.append({"phase": ph, "pct": round(pct, 2), "ci95": [round(lo, 2), round(hi, 2)],
                       "fits_ranges_of": fits, "ranges": {b: [round(v, 2) for v in ranges[b][ph]] for b in BATCHES}})
        if len(fits) == 1:
            votes.append(fits[0])
    b12 = [v for v in votes if v != "Batch_3"]
    ans = b12[0] if b12 and len(set(votes)) == 1 else None
    return {"answer": ans, "votes": votes, "phases": detail}


# ---------------------------------------------------------------- calibration helpers

def logp(P):
    return np.log(np.clip(P, 1e-6, 1))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def apply_T(P, T):
    return softmax(logp(P) / T)


def fit_T(P, y):
    nll = [-np.log(apply_T(P, T)[np.arange(len(y)), y] + 1e-12).mean() for T in TGRID]
    return float(TGRID[int(np.argmin(nll))])


def geomean(Ps):
    return softmax(np.mean([logp(p) for p in Ps], 0))


def choose_threshold(top, correct, target=TARGET, min_n=5):
    """Smallest probability t such that answers with top >= t were >= target right (at least min_n of them)."""
    order = np.argsort(-top)
    best = None
    for k in range(min_n, len(order) + 1):
        if correct[order[:k]].mean() >= target:
            best = float(top[order[k - 1]])
    return best if best is not None else 1.01


# ---------------------------------------------------------------- data and per-fold model fits

def load_data():
    d = load_mat()
    sids, y = list(d["sids"]), d["y"]
    fs, _, names, X = fp_table()
    order = [fs.index(s) for s in sids]
    tiles = TX.load_tiles()
    ms = {r["sample_id"]: {ph: {"pct": float(r[f"{ph}_pct"]), "se_pp": float(r[f"{ph}_se_pp"])} for ph in PHASES3}
          for r in C.read_csv(SE_CSV)}
    return {"sids": sids, "y": y, "S": d["S"], "groups": d["groups"], "Xfp": {v: X[v][order] for v in FP_VIEWS},
            "fp_names": names, "tiles": tiles, "ms": ms}


def fit_b12(X, y):
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], 0)) / 1.349
    iqr[iqr == 0] = 1
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(np.clip((X - med) / iqr, -3, 3), y)
    return {"median": med, "iqr": iqr, "lda": lda}


def b12_proba(m, x):
    return m["lda"].predict_proba(np.clip((x - m["median"]) / m["iqr"], -3, 3)[None])[0]


def raw_probs(D, tr, te, H, with_b12=False):
    """Raw (uncalibrated) probabilities for locations `te` from models fitted on locations `tr`."""
    y, sids = D["y"], D["sids"]
    tr, te = np.asarray(tr), np.asarray(te)
    out = {"fp_views": {}, "tx_views": {}}
    for v in FP_VIEWS:
        m = fp_fit_view(D["Xfp"][v][tr], y[tr])
        out["fp_views"][v] = np.array([fp_view_proba(m, D["Xfp"][v][i])[0] for i in te])
    for v in TX_VIEWS:
        m = TX.fit_view({sids[i]: D["tiles"][v][sids[i]] for i in tr}, {sids[i]: int(y[i]) for i in tr})
        out["tx_views"][v] = np.array([TX.view_proba(m, D["tiles"][v][sids[i]]) for i in te])
    out["mat"] = (head_S(D["S"], y, tr, te) + linear_grid(H, y, tr, te)) / 2
    if with_b12:
        t12 = tr[y[tr] < 2]
        ms = {v: fit_b12(D["Xfp"][v][t12], y[t12]) for v in FP_VIEWS}
        out["b12"] = np.array([np.mean([b12_proba(ms[v], D["Xfp"][v][i]) for v in FP_VIEWS], 0) for i in te])
    return out


def loc_level(r):
    """Location-level raw probabilities: fingerprint and texture combine their views by geometric mean."""
    return {"fp": geomean(list(r["fp_views"].values())), "tx": geomean(list(r["tx_views"].values())), "mat": r["mat"]}


_D = None


def _init():
    global _D
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _D = load_data()


def _fold(args):
    te, H = args
    n = len(_D["y"])
    tr = np.setdiff1d(np.arange(n), te)
    inner = {int(j): raw_probs(_D, np.setdiff1d(tr, [j]), [j], H) for j in tr}
    outer = raw_probs(_D, tr, te, H, with_b12=True)
    return te, inner, outer


# ---------------------------------------------------------------- the decision

def sides(lv, T):
    p_img = (apply_T(lv["fp"], T["fp"]) + apply_T(lv["tx"], T["tx"])) / 2 if USE_TEXTURE else apply_T(lv["fp"], T["fp"])
    return p_img, apply_T(lv["mat"], T["mat"])


def decide(p_img, p_mat, fp_views_agree=True, flags=(), t_high=1.01, rng=None):
    """v2's agreement rule on calibrated probabilities, plus the range rule and the calibrated High threshold."""
    a, b = np.asarray(p_img, float), np.asarray(p_mat, float)
    fa, fb = int(a.argmax()), int(b.argmax())
    avg = (a + b) / 2
    reasons = list(flags)
    if fa == fb:
        kind, answer = "specific", BATCHES[fa]
    elif fa < 2 and fb < 2:
        kind, answer = "pair", "Batch_1 or Batch_2"
        reasons.append("the imaging side and the material side split between Batch_1 and Batch_2, which are "
                       "statistically indistinguishable in this data")
        if rng and rng.get("answer"):
            kind, answer = "specific_by_material_range", rng["answer"]
            reasons.append(f"GET4 material-range rule: the composition with its 95 % error bars fits only "
                           f"{rng['answer']}'s range (phases voting: {', '.join(rng['votes'])})")
    else:
        kind, answer = "unsure", "unsure"
        reasons.append(f"the sides disagree (imaging side: {BATCHES[fa]}, material side: {BATCHES[fb]})")
    if not fp_views_agree:
        reasons.append("the fingerprint model's detector views disagree")
    if kind == "specific":
        strong = avg.max() >= t_high
        if not strong:
            reasons.append(f"agreement, but the average calibrated probability {avg.max():.2f} is below the "
                           f"{TARGET:.0%}-accuracy level {t_high:.2f}")
        tier = "High" if strong and fp_views_agree and not flags else "Medium"
    else:
        tier = "Medium" if kind != "unsure" else "Low"
    lean = BATCHES[int(avg.argmax())]
    return {"answer": answer if kind != "unsure" else f"unsure (leaning {lean})", "answer_type": kind,
            "confidence": tier, "leaning": lean,
            "imaging_side_pick": BATCHES[fa], "material_side_pick": BATCHES[fb], "sides_agree": fa == fb,
            "prediction_set": [BATCHES[k] for k in range(3) if avg[k] >= 0.2],
            "average_calibrated_probabilities": dict(zip(BATCHES, np.round(avg, 3).tolist())),
            "high_tier_threshold": round(float(t_high), 3),
            "reasons": reasons or [f"both sides agree and the average calibrated probability is above the "
                                   f"{TARGET:.0%}-accuracy level; no warning flags"]}


def forced(p_fp_cal, rng, p_b12):
    if int(np.argmax(p_fp_cal)) == 2:
        return "Batch_3", "fingerprint model: Batch_3"
    if rng and rng.get("answer"):
        return rng["answer"], "fingerprint: not Batch_3; GET4 material-range rule"
    return BATCHES[int(np.argmax(p_b12))], "fingerprint: not Batch_3; Batch_1-vs-Batch_2 fingerprint specialist"


def is_correct(dec, true):
    k = dec["answer_type"]
    if k in ("specific", "specific_by_material_range"):
        return dec["answer"] == true
    if k == "pair":
        return true != "Batch_3"
    return False


def views_agree(r, i):
    return len({int(np.argmax(r["fp_views"][v][i])) for v in FP_VIEWS}) == 1


def fold_result(D, te, inner, outer):
    """Calibrate and choose the High threshold on the inner (held-out) predictions, then decide the outer ones."""
    y, sids = D["y"], D["sids"]
    J = sorted(inner)
    lv_in = {k: np.concatenate([loc_level(inner[j])[k] for j in J]) for k in ("fp", "tx", "mat")}
    T = {k: fit_T(lv_in[k], y[J]) for k in lv_in}
    pi, pm = sides(lv_in, T)
    tr_ms = {sids[j]: D["ms"][sids[j]] for j in J}
    ranges = batch_ranges(tr_ms, {sids[j]: BATCHES[y[j]] for j in J})
    inner_dec = [decide(pi[k], pm[k], views_agree(inner[j], 0)) for k, j in enumerate(J)]
    spec = np.array([d["answer_type"] == "specific" for d in inner_dec])
    avg_top = np.array([max(d["average_calibrated_probabilities"].values()) for d in inner_dec])
    corr = np.array([is_correct(d, BATCHES[y[j]]) for d, j in zip(inner_dec, J)])
    t_high = choose_threshold(avg_top[spec], corr[spec]) if spec.sum() >= 5 else 1.01
    lv = loc_level(outer)
    po, pmo = sides(lv, T)
    res = {}
    for k, i in enumerate(te):
        rng = range_rule(D["ms"][sids[i]], ranges)
        dec = decide(po[k], pmo[k], views_agree(outer, k), (), t_high, rng)
        fa, why = forced(apply_T(lv["fp"][k:k + 1], T["fp"])[0], rng, outer["b12"][k])
        res[sids[i]] = {"true": BATCHES[y[i]], "decision": dec, "range_rule": rng, "forced": {"answer": fa, "route": why},
                        "calibrated": {"imaging_side": po[k].round(4).tolist(), "material_side": pmo[k].round(4).tolist(),
                                       "fingerprint": apply_T(lv["fp"][k:k + 1], T["fp"])[0].round(4).tolist(),
                                       "texture": apply_T(lv["tx"][k:k + 1], T["tx"])[0].round(4).tolist()},
                        "raw": {"fingerprint_views": {v: outer["fp_views"][v][k].round(4).tolist() for v in FP_VIEWS},
                                "texture_views": {v: outer["tx_views"][v][k].round(4).tolist() for v in TX_VIEWS},
                                "material": outer["mat"][k].round(4).tolist(), "b12_specialist": outer["b12"][k].round(4).tolist()},
                        "fold": {"temperatures": T, "high_threshold": t_high, "ranges": ranges}}
    return res


# ---------------------------------------------------------------- commands

def cmd_prepare(_):
    rows = []
    for batch, sid in C.sample_ids():
        ms = material_se(np.load(C.proc_dir(sid) / "final_labels.npy"))
        rows.append({"batch": batch, "sample_id": sid, **{f"{ph}_{k}": ms[ph][v] for ph in PHASES3
                                                          for k, v in (("pct", "pct"), ("se_pp", "se_pp"))}})
    C.write_csv(SE_CSV, rows)
    print(f"GET4 error bars -> {SE_CSV}; median SE pore {np.median([r['pore_se_pp'] for r in rows]):.2f} pp")
    d = load_mat()
    n = len(d["y"])
    loo = [np.array([i]) for i in range(n)]
    logo = [np.flatnonzero(d["groups"] == g) for g in np.unique(d["groups"])]
    H_loo, H_logo = texton_fold_hists(d, loo), texton_fold_hists(d, logo)
    np.savez_compressed(HISTS, loo=np.stack(H_loo), logo=np.stack(H_logo))
    print(f"texton histograms for {len(loo)} + {len(logo)} folds -> {HISTS}")


def bal(y, pred):
    return round(float(np.mean([(pred[y == k] == k).mean() for k in range(3)])), 3)


def summarise(D, res):
    y = np.array([BATCHES.index(res[s]["true"]) for s in D["sids"]])
    decs = [res[s]["decision"] for s in D["sids"]]
    kinds = np.array([d["answer_type"] for d in decs])
    corr = np.array([is_correct(d, res[s]["true"]) for d, s in zip(decs, D["sids"])])
    ans = kinds != "unsure"

    def ci(k, n):
        p, lo, hi = wilson(k, n)
        return {"correct": f"{k}/{n}", "accuracy": round(p, 3), "ci95": [round(lo, 3), round(hi, 3)]} if n else None

    per_batch = {}
    for s, d in zip(D["sids"], decs):
        t = res[s]["true"]
        key = ("right" if is_correct(d, t) else "wrong") if d["answer_type"] != "unsure" else "unsure"
        key = "pair_" + key if d["answer_type"] == "pair" else key
        per_batch.setdefault(t, {}).setdefault(key, 0)
        per_batch[t][key] += 1
    models = {}
    for name, key in (("fingerprint", "fingerprint"), ("texture", "texture"), ("imaging_side", "imaging_side"),
                      ("material_side", "material_side")):
        P = np.array([res[s]["calibrated"][key] for s in D["sids"]])
        models[name] = bal(y, P.argmax(1))
    for v in TX_VIEWS:
        models[f"texture_{v}_only"] = bal(y, np.array([res[s]["raw"]["texture_views"][v] for s in D["sids"]]).argmax(1))
    allavg = np.array([(np.array(res[s]["calibrated"]["imaging_side"]) + res[s]["calibrated"]["material_side"]) / 2 for s in D["sids"]])
    models["imaging_plus_material_always_answer"] = bal(y, allavg.argmax(1))
    fpred = np.array([BATCHES.index(res[s]["forced"]["answer"]) for s in D["sids"]])
    rr = [(res[s]["range_rule"]["answer"], res[s]["true"]) for s in D["sids"] if res[s]["range_rule"]["answer"]]
    used = [s for s, d in zip(D["sids"], decs) if d["answer_type"] == "specific_by_material_range"]
    tiers = {t: ci(int(sum(c for c, d in zip(corr, decs) if d["confidence"] == t)), sum(1 for d in decs if d["confidence"] == t))
             for t in ("High", "Medium", "Low")}
    return {"coverage": f"{int(ans.sum())}/{len(ans)} ({ans.mean():.0%})",
            "accuracy_when_answering": ci(int(corr[ans].sum()), int(ans.sum())),
            "by_type": {k: ci(int(corr[kinds == k].sum()), int((kinds == k).sum())) for k in
                        ("specific", "specific_by_material_range", "pair")},
            "confidence_tiers": tiers, "outcome_by_true_batch": per_batch,
            "unsure_true_batches": [res[s]["true"] for s, k in zip(D["sids"], kinds) if k == "unsure"],
            "model_balanced_accuracy": models,
            "forced_mode": {"balanced_accuracy": bal(y, fpred),
                            "recall": {BATCHES[k]: f"{int((fpred[y == k] == k).sum())}/{int((y == k).sum())}" for k in range(3)}},
            "range_rule": {"fired": f"{len(rr)}/{len(D['sids'])}", "right_when_fired": f"{sum(a == t for a, t in rr)}/{len(rr)}",
                           "used_to_resolve_pair": f"{len(used)} ({sum(res[s]['decision']['answer'] == res[s]['true'] for s in used)} right)"},
            "high_threshold_per_fold": {"median": round(float(np.median([res[s]["fold"]["high_threshold"] for s in D["sids"]])), 3),
                                        "min": round(float(min(res[s]["fold"]["high_threshold"] for s in D["sids"])), 3),
                                        "max": round(float(max(res[s]["fold"]["high_threshold"] for s in D["sids"])), 3)}}


def cmd_evaluate(args):
    global EVAL
    if args.with_texture:
        EVAL = C.OUT_MET / "v3_eval_with_texture.json"
    D = load_data()
    z = np.load(HISTS)
    n = len(D["y"])
    schemes = {"LOO": ([np.array([i]) for i in range(n)], z["loo"]),
               "LOGO_session": ([np.flatnonzero(D["groups"] == g) for g in np.unique(D["groups"])], z["logo"])}
    out = {}
    for name, (folds, H) in schemes.items():
        res = {}
        with ProcessPoolExecutor(args.workers, initializer=_init) as ex:  # workers only fit raw models; sides() runs here
            for te, inner, outer in ex.map(_fold, [(f, H[k]) for k, f in enumerate(folds)]):
                res.update(fold_result(D, te, inner, outer))
        s = summarise(D, res)
        out[name] = {**s, "per_sample": res}
        print(f"{name}: coverage {s['coverage']}, right when answering {s['accuracy_when_answering']}, tiers "
              f"{ {k: (v['correct'] if v else None) for k, v in s['confidence_tiers'].items()} }, forced "
              f"{s['forced_mode']['balanced_accuracy']}, models {s['model_balanced_accuracy']}, range {s['range_rule']}", flush=True)
    C.write_json(EVAL, out)
    o = out["LOO"]
    ledger({"config": "v3_v2_plus_teammate_nested" + ("_with_texture" if args.with_texture else "_no_texture") + "_exploratory", "kind": "LOO",
            "balanced_accuracy": o["accuracy_when_answering"]["accuracy"],
            "detail": json.dumps({"metric": "accuracy when answering (not balanced)", "coverage": o["coverage"],
                                  "forced_balanced": o["forced_mode"]["balanced_accuracy"],
                                  "LOGO_coverage": out["LOGO_session"]["coverage"],
                                  "LOGO_accuracy_when_answering": out["LOGO_session"]["accuracy_when_answering"]["accuracy"]})})


def fit_parts(D, keep):
    """Texture views and the B1-vs-B2 specialist fitted on locations `keep`."""
    y, sids = D["y"], D["sids"]
    keep = np.asarray(keep)
    tx = {v: TX.fit_view({sids[i]: D["tiles"][v][sids[i]] for i in keep}, {sids[i]: int(y[i]) for i in keep}) for v in TX_VIEWS}
    k12 = keep[y[keep] < 2]
    b12 = {v: fit_b12(D["Xfp"][v][k12], y[k12]) for v in FP_VIEWS}
    ranges = batch_ranges({sids[i]: D["ms"][sids[i]] for i in keep}, {sids[i]: BATCHES[y[i]] for i in keep})
    return {"texture": tx, "b12": b12, "ranges": ranges, "trained_on": [sids[i] for i in keep]}


def cmd_fit(_):
    D = load_data()
    ev = json.load(open(EVAL))["LOO"]["per_sample"]
    y = D["y"]
    lv = {"fp": geomean_rows([ev[s]["raw"]["fingerprint_views"] for s in D["sids"]]),
          "tx": geomean_rows([ev[s]["raw"]["texture_views"] for s in D["sids"]]),
          "mat": np.array([ev[s]["raw"]["material"] for s in D["sids"]])}
    T = {k: fit_T(lv[k], y) for k in lv}
    pi, pm = sides(lv, T)
    decs = [decide(pi[k], pm[k], len({int(np.argmax(v)) for v in ev[s]["raw"]["fingerprint_views"].values()}) == 1)
            for k, s in enumerate(D["sids"])]
    spec = np.array([d["answer_type"] == "specific" for d in decs])
    top = np.array([max(d["average_calibrated_probabilities"].values()) for d in decs])
    corr = np.array([is_correct(d, BATCHES[y[k]]) for k, d in enumerate(decs)])
    model = {**fit_parts(D, np.arange(len(y))), "temperatures": T, "high_threshold": choose_threshold(top[spec], corr[spec])}
    with open(MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"v3 model (temperatures {T}, High threshold {model['high_threshold']:.3f}) -> {MODEL}")


def forced_loo(D, y):
    """Leave-one-location-out forced mode under labels `y` (only fingerprint models and the range rule are involved,
    so it is cheap enough for a permutation null)."""
    n, sids = len(y), D["sids"]
    pred = np.zeros(n, int)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        fp = geomean([np.array([fp_view_proba(fp_fit_view(D["Xfp"][v][tr], y[tr]), D["Xfp"][v][i])[0]]) for v in FP_VIEWS])[0]
        if fp.argmax() == 2:
            pred[i] = 2
            continue
        rng = range_rule(D["ms"][sids[i]], batch_ranges({sids[j]: D["ms"][sids[j]] for j in tr}, {sids[j]: BATCHES[y[j]] for j in tr}))
        if rng["answer"]:
            pred[i] = BATCHES.index(rng["answer"])
            continue
        t12 = tr[y[tr] < 2]
        pred[i] = int(np.argmax(np.mean([b12_proba(fit_b12(D["Xfp"][v][t12], y[t12]), D["Xfp"][v][i]) for v in FP_VIEWS], 0)))
    return pred


def _perm_forced(seed):
    y = np.random.default_rng(seed).permutation(_D["y"])
    return bal(y, forced_loo(_D, y))


def cmd_forced_null(args):
    D = load_data()
    obs = bal(D["y"], forced_loo(D, D["y"]))
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        null = np.array(list(ex.map(_perm_forced, range(1, args.n + 1))))
    out = {"observed_balanced_accuracy": obs, "n_permutations": args.n, "null_mean": round(float(null.mean()), 3),
           "null_p95": round(float(np.percentile(null, 95)), 3), "null_p99": round(float(np.percentile(null, 99)), 3),
           "p_value": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)}
    C.write_json(C.OUT_MET / "v3_forced_null.json", out)
    print(out)


def track_record(dec):
    """Honest track record of this answer type and confidence tier (nested leave-one-out and session-out)."""
    if not EVAL.exists():
        return None
    ev = json.load(open(EVAL))
    out = {}
    for sch in ("LOO", "LOGO_session"):
        o = ev[sch]
        out[sch] = {"this_confidence_tier": o["confidence_tiers"].get(dec["confidence"]),
                    "this_answer_type": o["by_type"].get(dec["answer_type"]),
                    "overall": {"coverage": o["coverage"], "right_when_answering": o["accuracy_when_answering"]}}
    return out


def texture_block(tx_models, tiles, p_cal_loc, raw_views, win, run):
    m = tx_models.get("bse")
    tops = TX.top_features(m, tiles["bse"], BATCHES.index(win), BATCHES.index(run)) if m is not None and "bse" in tiles else []
    return {"what_this_is": "tile texture model: 12.8 um tiles, local binary patterns + co-occurrence texture + "
                            "power spectrum, logistic regression per detector view, averaged over tiles",
            "used_in_decision": USE_TEXTURE,
            "why_not_used": None if USE_TEXTURE else "in nested leave-one-out it lowered accuracy when answering from "
                                                     "85 % to 80 %, so it is shown for information only",
            "calibrated_probabilities": dict(zip(BATCHES, np.round(p_cal_loc, 3).tolist())),
            "per_view_raw": {TX_VIEWS[v]: dict(zip(BATCHES, np.round(p, 3).tolist())) for v, p in raw_views.items()},
            "n_tiles": {TX_VIEWS[v]: int(len(t)) for v, t in tiles.items()},
            "top_features": tops}


def geomean_rows(rows):
    return np.array([geomean([np.array(v)[None] for v in r.values()])[0] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "evaluate", "fit", "forced_null"])
    ap.add_argument("--n", type=int, default=500, help="permutations for forced_null")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--with-texture", action="store_true", help="the teammate's spec: imaging side = fingerprint + texture")
    args = ap.parse_args()
    global USE_TEXTURE
    USE_TEXTURE = args.with_texture
    {"prepare": cmd_prepare, "evaluate": cmd_evaluate, "fit": cmd_fit, "forced_null": cmd_forced_null}[args.cmd](args)


if __name__ == "__main__":
    main()
