"""Decision and uncertainty layer: combine the fingerprint model and the material model (segmentation + DINO).

Rule (fixed before the blind set; no tuned thresholds):
  both models pick the same batch                      -> that batch            (type "specific")
  they disagree, but both pick Batch_1 or Batch_2      -> "Batch_1 or Batch_2"  (type "pair")
  anything else                                        -> "unsure"
Confidence tier: High = specific, both models >= 0.6, detector views agree, no novelty / shortcut flags;
Medium = other specific answers and pair answers; Low = unsure. A prediction set (every batch whose average
probability >= 0.2) is reported alongside.

  evaluate   honest track record from leave-one-location-out and leave-one-session-out probabilities of both models
             -> outputs/metrics/decision_eval.json (coverage, accuracy per answer type and per batch, 95 % CIs)

Usage: python -m src.decision evaluate
"""
import argparse
import json

import numpy as np

from . import config as C
from .bid_model import BATCHES, ledger, load, procedure
from .validate import wilson

EVAL = C.OUT_MET / "decision_eval.json"


def decide(p_fp, p_mat, fp_views_agree=True, flags=()):
    """p_fp, p_mat: probability dicts or arrays over BATCHES. flags: reasons that lower confidence (novelty etc.)."""
    a = np.array([p_fp[b] for b in BATCHES]) if isinstance(p_fp, dict) else np.asarray(p_fp, float)
    b = np.array([p_mat[x] for x in BATCHES]) if isinstance(p_mat, dict) else np.asarray(p_mat, float)
    fa, fb = int(a.argmax()), int(b.argmax())
    reasons = list(flags)
    if fa == fb:
        kind, answer = "specific", BATCHES[fa]
    elif fa < 2 and fb < 2:
        kind, answer = "pair", "Batch_1 or Batch_2"
        reasons.append("the two models split between Batch_1 and Batch_2, which are statistically indistinguishable in this data")
    else:
        kind, answer = "unsure", "unsure"
        reasons.append(f"the models disagree (fingerprint: {BATCHES[fa]}, material model: {BATCHES[fb]})")
    if not fp_views_agree:
        reasons.append("the three detector views of the fingerprint model disagree")
    if kind == "specific":
        strong = a.max() >= 0.6 and b.max() >= 0.6
        if not strong:
            reasons.append(f"agreement but modest confidence (fingerprint {a.max():.2f}, material model {b.max():.2f})")
        tier = "High" if strong and fp_views_agree and not flags else "Medium"
    else:
        tier = "Medium" if kind == "pair" else "Low"
    avg = (a + b) / 2
    out = {"answer": answer, "answer_type": kind, "confidence": tier,
           "fingerprint_pick": BATCHES[fa], "material_model_pick": BATCHES[fb], "models_agree": fa == fb,
           "prediction_set": [BATCHES[k] for k in range(3) if avg[k] >= 0.2],
           "average_probabilities": dict(zip(BATCHES, np.round(avg, 3).tolist())),
           "reasons": reasons or ["both models agree with confidence >= 0.6; no warning flags"]}
    if EVAL.exists():
        ev = json.load(open(EVAL))["LOO"]
        out["track_record_for_this_answer_type"] = ev["by_type"].get(kind)
        out["track_record_overall"] = {k: ev[k] for k in ("coverage", "accuracy_when_answering")}
    return out


def score(y, P_fp, P_mat):
    rows = [decide(P_fp[i], P_mat[i]) for i in range(len(y))]
    kinds = np.array([r["answer_type"] for r in rows])
    correct = np.array([(r["answer"] == BATCHES[y[i]]) if r["answer_type"] == "specific" else
                        (y[i] < 2 if r["answer_type"] == "pair" else False) for i, r in enumerate(rows)])
    ans = kinds != "unsure"
    def ci(k, n):
        p, lo, hi = wilson(k, n)
        return {"correct": f"{k}/{n}", "accuracy": round(p, 3), "ci95": [round(lo, 3), round(hi, 3)]} if n else None
    out = {"coverage": f"{int(ans.sum())}/{len(y)} ({ans.mean():.0%})",
           "accuracy_when_answering": ci(int(correct[ans].sum()), int(ans.sum())),
           "by_type": {k: ci(int(correct[kinds == k].sum()), int((kinds == k).sum())) for k in ("specific", "pair")},
           "unsure": int((kinds == "unsure").sum()),
           "specific_by_true_batch": {BATCHES[c]: f"{int((correct & (kinds == 'specific') & (y == c)).sum())}/{int(((kinds == 'specific') & (y == c)).sum())}" for c in range(3)},
           "unsure_true_batches": [BATCHES[y[i]] for i in np.flatnonzero(kinds == "unsure")],
           "confidence_tiers": {t: ci(int(sum(correct[i] for i in range(len(y)) if rows[i]['confidence'] == t)),
                                      sum(1 for r in rows if r["confidence"] == t)) for t in ("High", "Medium", "Low")}}
    return out


def cmd_evaluate(_):
    cv = json.load(open(C.OUT_MET / "fingerprint_model_cv.json"))
    d = load()
    sids, y = d["sids"], d["y"]
    groups = d["groups"]
    mat = {"LOO": procedure(d, y)["F"],
           "LOGO_session": procedure(d, y, [np.flatnonzero(groups == g) for g in np.unique(groups)])["F"]}
    out = {}
    for scheme in ("LOO", "LOGO_session"):
        P_fp = np.array([cv[scheme][s]["p"] for s in sids])
        out[scheme] = score(y, P_fp, mat[scheme])
        out[scheme]["per_sample"] = {s: {"true": BATCHES[y[i]], "fingerprint_p": cv[scheme][s]["p"],
                                         "fingerprint_views": cv[scheme][s]["per_view"],
                                         "material_p": [round(float(v), 4) for v in mat[scheme][i]]} for i, s in enumerate(sids)}
        o = out[scheme]
        print(f"{scheme:13s} coverage {o['coverage']}, accuracy when answering {o['accuracy_when_answering']}, "
              f"by type {o['by_type']}, unsure {o['unsure']} {o['unsure_true_batches']}, tiers {o['confidence_tiers']}")
    C.write_json(EVAL, out)
    o = out["LOO"]
    ledger({"config": "decision_rule_fingerprint_x_F_exploratory", "kind": "LOO",
            "balanced_accuracy": o["accuracy_when_answering"]["accuracy"],
            "detail": json.dumps({"metric": "accuracy when answering (not balanced)", "coverage": o["coverage"],
                                  "LOGO_coverage": out["LOGO_session"]["coverage"],
                                  "LOGO_accuracy_when_answering": out["LOGO_session"]["accuracy_when_answering"]["accuracy"]})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate"])
    args = ap.parse_args()
    {"evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
