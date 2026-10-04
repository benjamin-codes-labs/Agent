"""v1 vs v2 (active-learning round) comparison on identical evaluation data.

Independent AI points (uniform + class-stratified) are re-scored on both versions' label maps; points that fall in a
superpixel used for TRAINING in round 2 are dropped from both, so v2 gets no leaked advantage. Also: agreement on
held-out AI superpixels (round-1 val split = same as v1; round-2 val split = new, targeted at uncertain regions),
pixel agreement v1 vs v2, phase fractions per batch, and the batch-ID numbers from both versions' metric files.

Writes outputs/metrics/v1_vs_v2.json.

Usage: python -m src.compare_versions
"""
import json

import numpy as np

from . import config as C
from .validate import wilson

V1 = C.OUT_MET / "v1"


def label_maps(version):
    name = "student_labels_v1.npy" if version == "v1" else "student_labels.npy"
    return {s: np.load(C.proc_dir(s) / name) for _, s in C.sample_ids()}


def r2_train_superpixels():
    p = C.PROC / "ai_labels" / "items_r2.csv"
    out = {}
    if p.exists():
        for r in C.read_csv(p):
            if r["split"] == "train":
                out.setdefault(r["sample_id"], set()).add(int(r["superpixel"]))
    return out


def point_scores(maps, rows):
    decided = [r for r in rows if r["ai_label"] in C.CLASSES]
    def model(r):
        k = maps[r["sample_id"]][int(r["y"]), int(r["x"])]
        return C.CLASSES[k] if k != C.EXCLUDED else "excluded"
    hits = [model(r) == r["ai_label"] for r in decided]
    acc, lo, hi = wilson(sum(hits), len(hits))
    per = {}
    for c in C.CLASSES:
        sub = [model(r) == c for r in decided if r["ai_label"] == c]           # recall per AI class
        prec = [r["ai_label"] == c for r in decided if model(r) == c]         # precision per model class
        per[c] = {"recall": round(float(np.mean(sub)), 3) if sub else None, "n_ai": len(sub),
                  "precision": round(float(np.mean(prec)), 3) if prec else None, "n_model": len(prec)}
    return {"n": len(hits), "accuracy": round(acc, 3), "ci95": [round(lo, 3), round(hi, 3)], "per_class": per}


def superpixel_agreement(maps, rows):
    sps = {}
    hits = []
    for r in rows:
        if r["ai_label"] not in C.CLASSES or r["confidence"] not in ("medium", "high"):
            continue
        sid = r["sample_id"]
        if sid not in sps:
            sps[sid] = np.load(C.proc_dir(sid) / "superpixels.npy")
        vals = maps[sid][sps[sid] == int(r["superpixel"])]
        vals = vals[vals != C.EXCLUDED]
        if len(vals):
            hits.append(C.CLASSES[np.bincount(vals, minlength=4).argmax()] == r["ai_label"])
    acc, lo, hi = wilson(sum(hits), len(hits))
    return {"n": len(hits), "agreement": round(acc, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def main():
    maps = {v: label_maps(v) for v in ("v1", "v2")}
    used = r2_train_superpixels()
    sps = {s: np.load(C.proc_dir(s) / "superpixels.npy") for _, s in C.sample_ids()}
    out = {"points": {}, "held_out_superpixels": {}}
    for name in ("uniform", "stratified"):
        rows = C.read_csv(C.OUT_MET / f"ai_points_{name}.csv")
        clean = [r for r in rows if int(sps[r["sample_id"]][int(r["y"]), int(r["x"])]) not in used.get(r["sample_id"], set())]
        out["points"][name] = {"n_dropped_overlapping_round2_training": len(rows) - len(clean),
                               **{v: point_scores(maps[v], clean) for v in ("v1", "v2")}}
    r1 = [r for r in C.read_csv(C.OUT_MET / "ai_superpixel_labels.csv") if r["split"] == "val"]
    out["held_out_superpixels"]["round1_val"] = {v: superpixel_agreement(maps[v], r1) for v in ("v1", "v2")}
    p2 = C.OUT_MET / "ai_superpixel_labels_r2.csv"
    if p2.exists():
        r2 = [r for r in C.read_csv(p2) if r["split"] == "val"]
        out["held_out_superpixels"]["round2_val_uncertain_regions"] = {v: superpixel_agreement(maps[v], r2) for v in ("v1", "v2")}
    # pixel agreement and per-batch fractions
    agree, conf = [], np.zeros((4, 4), int)
    frac = {"v1": {}, "v2": {}}
    for b, s in C.sample_ids():
        a, c = maps["v1"][s], maps["v2"][s]
        m = (a != C.EXCLUDED) & (c != C.EXCLUDED)
        agree.append((a[m] == c[m]).mean())
        conf += np.bincount(4 * a[m].astype(int) + c[m].astype(int), minlength=16).reshape(4, 4)
        for v, lab in (("v1", a), ("v2", c)):
            vv = lab[lab != C.EXCLUDED]
            frac[v].setdefault(b, []).append([100 * (vv == k).mean() for k in range(4)])
    out["pixel_agreement_v1_v2"] = round(float(np.mean(agree)), 4)
    out["confusion_rows_v1_cols_v2_fraction"] = (conf / conf.sum()).round(4).tolist()
    out["phase_fractions_by_batch"] = {v: {b: dict(zip(C.CLASSES, np.round(np.mean(x, 0), 2).tolist()))
                                           for b, x in sorted(frac[v].items())} for v in ("v1", "v2")}
    # batch-ID numbers from both versions' metric files
    def bid(folder):
        r = {}
        for f, keys in (("bid_scores.json", ("LOO_S", "LOO_F", "LOGO_F", "nuisance_only_LOO")),):
            if (folder / f).exists():
                j = json.load(open(folder / f))
                r.update({k: j[k]["balanced_accuracy"] for k in keys})
        if (folder / "bid_null.json").exists():
            r["perm_p_F"] = json.load(open(folder / "bid_null.json"))["F"]["p_value"]
        if (folder / "b12_two_step.json").exists():
            j = json.load(open(folder / "b12_two_step.json"))["variants"]["all_kpis"]
            r.update({"two_step_LOO": j["LOO"]["balanced_accuracy"], "two_step_LOGO": j["LOGO_session"]["balanced_accuracy"],
                      "two_step_B1vB2_p": j["step2_B1_vs_B2_only"]["p_value"]})
        return r
    out["batch_id"] = {"v1": bid(V1), "v2": bid(C.OUT_MET)}
    C.write_json(C.OUT_MET / "v1_vs_v2.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
