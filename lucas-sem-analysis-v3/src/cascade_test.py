"""Exploratory: cascade "Batch 3?" -> "Batch 1 or 2?" with the step-2 head chosen per class, tested honestly.

  reference  F = mean(S, D), plain 3-class argmax
  fixed      step 1: F says Batch_3 -> Batch_3; step 2: DINO head decides Batch_1 vs Batch_2.
             (Designed AFTER seeing per-class recalls, so its leave-one-out score is optimistic.)
  nested     step 1 as above; step 2 head (S, D or F) chosen INSIDE each outer fold by an inner leave-one-out on
             the 30 training samples (best Batch_1-vs-Batch_2 accuracy among the training B1/B2 samples). This is the
             honest estimate of the strategy "pick the head that is best at Batch 1 vs 2".
The DINO texton vocabulary for each outer fold is fitted without the held-out sample and without labels.

Writes outputs/metrics/cascade_test.json.

Usage: python -m src.cascade_test
"""
import json
from collections import Counter

import numpy as np

from . import config as C
from .bid_model import BATCHES, head_S, ledger, linear_grid, load, summarise, texton_fold_hists


def two_way(p):
    """Batch_1 (0) vs Batch_2 (1) decision from 3-class probabilities."""
    return 0 if p[0] >= p[1] else 1


def main():
    d = load()
    y = d["y"]
    n = len(y)
    hists = texton_fold_hists(d, [np.array([i]) for i in range(n)])
    pred = {"reference": np.zeros(n, int), "fixed": np.zeros(n, int), "nested": np.zeros(n, int)}
    chosen = []
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        H = hists[i]
        pS = head_S(d["S"], y, tr, np.array([i]))[0]
        pD = linear_grid(H, y, tr, np.array([i]))[0]
        pF = (pS + pD) / 2
        pred["reference"][i] = int(pF.argmax())
        step1_b3 = pF.argmax() == 2
        pred["fixed"][i] = 2 if step1_b3 else two_way(pD)
        # inner leave-one-out on the training fold to pick the step-2 head
        inner = {"S": [], "D": [], "F": []}
        for j in tr:
            if y[j] == 2:
                continue
            tr2 = np.setdiff1d(tr, [j])
            s = head_S(d["S"], y, tr2, np.array([j]))[0]
            dd = linear_grid(H, y, tr2, np.array([j]))[0]
            for k, p in (("S", s), ("D", dd), ("F", (s + dd) / 2)):
                inner[k].append(two_way(p) == y[j])
        best = max(inner, key=lambda k: (np.mean(inner[k]), k == "F"))      # ties -> fused
        chosen.append(best)
        head_p = {"S": pS, "D": pD, "F": pF}[best]
        pred["nested"][i] = 2 if step1_b3 else two_way(head_p)
    out = {k: summarise(y, np.eye(3)[v]) for k, v in pred.items()}
    out["nested_step2_head_choices"] = dict(Counter(chosen))
    C.write_json(C.OUT_MET / "cascade_test.json", out)
    for k in ("reference", "fixed", "nested"):
        print(f"{k:10s} LOO balanced accuracy {out[k]['balanced_accuracy']:.3f}  {out[k]['recall']}")
    print("nested step-2 head chosen per fold:", out["nested_step2_head_choices"])
    for k in ("fixed", "nested"):
        ledger({"config": f"cascade_{k}_exploratory", "kind": "LOO", "balanced_accuracy": out[k]["balanced_accuracy"],
                "detail": json.dumps(out[k]["recall"])})


if __name__ == "__main__":
    main()
