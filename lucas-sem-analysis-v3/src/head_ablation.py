"""Diagnostic ablation: each head alone vs the fused model (leave-one-out and leave-one-session-out).

S = segmentation-feature head (shrinkage LDA on 5 label-map features), D = the DINO head chosen by the
pre-registered falsifier (texton histogram), F = mean(S, D). Logged to the look ledger as a diagnostic.

Writes outputs/metrics/head_ablation.json.

Usage: python -m src.head_ablation
"""
import json

import numpy as np

from . import config as C
from .bid_model import ledger, load, procedure, summarise


def main():
    d = load()
    y = d["y"]
    res = procedure(d, y)
    groups = d["groups"]
    res_g = procedure(d, y, [np.flatnonzero(groups == g) for g in np.unique(groups)])
    out = {"dino_head": res["dino_head"]}
    for k, name in (("S", "segmentation head only"), ("D", "DINO head only"), ("F", "fused S + D")):
        out[k] = {"name": name, "LOO": summarise(y, res[k]), "LOGO_session": summarise(y, res_g[k])}
        print(f"{name:24s} LOO {out[k]['LOO']['balanced_accuracy']:.3f} {out[k]['LOO']['recall']} | "
              f"LOGO {out[k]['LOGO_session']['balanced_accuracy']:.3f} {out[k]['LOGO_session']['recall']}")
    C.write_json(C.OUT_MET / "head_ablation.json", out)
    ledger({"config": "D_alone_diagnostic", "kind": "LOO", "balanced_accuracy": out["D"]["LOO"]["balanced_accuracy"],
            "detail": json.dumps(out["D"]["LOO"]["recall"])})


if __name__ == "__main__":
    main()
