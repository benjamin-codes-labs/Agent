"""Stage 6b: validation without hand labels.

Evidence combined into outputs/metrics/validation.csv (per sample) and validation_summary.json:
  1. baseline agreement: model pore / SiOx vs thr (BSE threshold) and vis (vision-LLM estimate); Spearman
  2. expected rankings and sanity gates
  3. stability: SD of phase fractions over teacher refits on random sample subsets and seeds
  4. student vs teacher on the 6 held-out samples (pixel agreement, confusion, fraction differences)
  5. independent AI point check: accuracy on uniformly sampled points (Wilson 95 % CI), confusion,
     AI-estimated phase fractions; per-class precision on class-stratified points
  6. AI superpixel labels held out from training (val split): agreement
  7. inference timings

Usage: python -m src.validate
"""
import json

import numpy as np
from scipy import stats

from . import config as C
from .teacher import gates


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def confusion(truth, pred, labels):
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(truth, pred):
        m[labels.index(t), labels.index(p)] += 1
    return m.tolist()


def point_check(name):
    path = C.OUT_MET / f"ai_points_{name}.csv"
    if not path.exists():
        return None
    rows = C.read_csv(path)
    maps = {}
    for r in rows:
        sid = r["sample_id"]
        if sid not in maps:
            maps[sid] = np.load(C.proc_dir(sid) / "final_labels.npy")
        k = maps[sid][int(r["y"]), int(r["x"])]
        r["model"] = C.CLASSES[k] if k != C.EXCLUDED else "excluded"
    decided = [r for r in rows if r["ai_label"] in C.CLASSES and r["model"] in C.CLASSES]
    hits = sum(r["ai_label"] == r["model"] for r in decided)
    acc, lo, hi = wilson(hits, len(decided))
    res = {"n_points": len(rows), "n_decided": len(decided),
           "n_uncertain": sum(r["ai_label"] == "uncertain" for r in rows),
           "accuracy": round(acc, 3), "accuracy_ci95": [round(lo, 3), round(hi, 3)],
           "confusion_rows_ai_cols_model": {"labels": C.CLASSES,
                                            "matrix": confusion([r["ai_label"] for r in decided], [r["model"] for r in decided], C.CLASSES)}}
    hc = [r for r in decided if r["confidence"] == "high"]
    if hc:
        a, l, h = wilson(sum(r["ai_label"] == r["model"] for r in hc), len(hc))
        res["accuracy_high_confidence_only"] = {"n": len(hc), "accuracy": round(a, 3), "ci95": [round(l, 3), round(h, 3)]}
    if name == "uniform":
        ai = [r["ai_label"] for r in rows if r["ai_label"] in C.CLASSES]
        model = [r["model"] for r in rows if r["model"] in C.CLASSES]
        res["phase_fractions_ai_points_pct"] = {c: [round(100 * x, 1) for x in wilson(ai.count(c), len(ai))] for c in C.CLASSES}
        res["phase_fractions_model_at_points_pct"] = {c: round(100 * model.count(c) / len(model), 1) for c in C.CLASSES}
    if name == "stratified":
        res["precision_by_model_class"] = {}
        for c in C.CLASSES:
            sub = [r for r in decided if r["model"] == c]
            p, l, h = wilson(sum(r["ai_label"] == c for r in sub), len(sub))
            res["precision_by_model_class"][c] = {"n": len(sub), "precision": round(p, 3), "ci95": [round(l, 3), round(h, 3)]}
    return res


def superpixel_val():
    path = C.OUT_MET / "ai_superpixel_labels.csv"
    if not path.exists():
        return None
    rows = [r for r in C.read_csv(path) if r["split"] == "val" and r["ai_label"] in C.CLASSES
            and r["confidence"] in ("medium", "high")]
    hits = n = 0
    per = {c: [0, 0] for c in C.CLASSES}
    for r in rows:
        sp = np.load(C.proc_dir(r["sample_id"]) / "superpixels.npy")
        lab = np.load(C.proc_dir(r["sample_id"]) / "final_labels.npy")
        vals = lab[sp == int(r["superpixel"])]
        vals = vals[vals != C.EXCLUDED]
        if not len(vals):
            continue
        major = C.CLASSES[np.bincount(vals, minlength=4).argmax()]
        hits += major == r["ai_label"]; n += 1
        per[r["ai_label"]][0] += major == r["ai_label"]; per[r["ai_label"]][1] += 1
    acc, lo, hi = wilson(hits, n)
    return {"n": n, "agreement": round(acc, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "recall_by_ai_class": {c: (round(h / t, 3) if t else None, t) for c, (h, t) in per.items()}}


def student_vs_teacher():
    sp = json.load(open(C.MODELS / "student_split.json"))
    agree_all = agree_conf = n_all = n_conf = 0
    conf = np.zeros((4, 4), int)
    diffs = []
    for sid in sp["held_out"]:
        t = np.load(C.proc_dir(sid) / "teacher_labels.npy")
        tp = np.load(C.proc_dir(sid) / "teacher_maxp.npy")
        s = np.load(C.proc_dir(sid) / "student_labels.npy")
        v = t != C.EXCLUDED
        c = v & (tp >= int(0.7 * 255))
        agree_all += int((t[v] == s[v]).sum()); n_all += int(v.sum())
        agree_conf += int((t[c] == s[c]).sum()); n_conf += int(c.sum())
        conf += np.bincount(4 * t[v].astype(int) + s[v].astype(int), minlength=16).reshape(4, 4)
        diffs.append([100 * ((s[v] == k).mean() - (t[v] == k).mean()) for k in range(4)])
    diffs = np.array(diffs)
    return {"held_out": sp["held_out"], "agreement_all_px": round(agree_all / n_all, 4),
            "agreement_confident_px": round(agree_conf / n_conf, 4),
            "confusion_rows_teacher_cols_student": conf.tolist(),
            "max_abs_fraction_diff_pp": {c: round(float(np.abs(diffs[:, k]).max()), 2) for k, c in enumerate(C.CLASSES)}}


def main():
    base = C.baseline()
    fr = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "phase_fractions.csv")}
    stab = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "stability.csv")} if (C.OUT_MET / "stability.csv").exists() else {}
    rows = []
    for sid, r in fr.items():
        b = base[sid]
        row = {"sample_id": sid, "batch": r["batch"],
               "pore_model": float(r["pore_pct"]), "pore_thr": b["pore_thr"], "pore_vis": b["pore_vis"],
               "siox_model": float(r["SiOx_pct"]), "siox_thr": b["siox_thr"], "siox_vis": b["siox_vis"],
               "siox_thrlike_model": float(r["siox_thrlike_pct"]),
               "graphite_model": float(r["graphite_pct"]), "cbd_model": float(r["CBD_pct"])}
        row["siox_minus_thr"] = round(row["siox_thrlike_model"] - b["siox_thr"], 2)
        row["pore_minus_thr"] = round(row["pore_model"] - b["pore_thr"], 2)
        if sid in stab:
            row.update({f"{c}_stability_sd": stab[sid][f"{c}_sd"] for c in C.CLASSES})
        rows.append(row)
    C.write_csv(C.OUT_MET / "validation.csv", rows)

    ok, gate_lines = gates({r["sample_id"]: {"pore": r["pore_model"], "SiOx": r["siox_model"],
                                             "SiOx_thrlike": r["siox_thrlike_model"]} for r in rows})
    sp = lambda a, b: round(float(stats.spearmanr([r[a] for r in rows], [r[b] for r in rows]).statistic), 3)
    summary = {
        "gates_pass": bool(ok), "gate_details": gate_lines,
        "spearman": {"pore_model_vs_thr": sp("pore_model", "pore_thr"), "pore_model_vs_vis": sp("pore_model", "pore_vis"),
                     "siox_model_vs_thr": sp("siox_model", "siox_thr"), "siox_model_vs_vis": sp("siox_model", "siox_vis")},
        "siox_mean_abs_diff_vs_thr_pp": round(float(np.mean([abs(r["siox_minus_thr"]) for r in rows if r["sample_id"] != "epqdaau9"])), 2),
        "samples_pore_below_thr": [r["sample_id"] for r in rows if r["pore_minus_thr"] < 0],
        "pore_between_thr_and_vis": sum(r["pore_thr"] <= r["pore_model"] <= r["pore_vis"] for r in rows),
    }
    if stab:
        summary["stability_sd_pp"] = {c: {"mean": round(float(np.mean([float(s[f"{c}_sd"]) for s in stab.values()])), 2),
                                          "max": round(float(np.max([float(s[f"{c}_sd"]) for s in stab.values()])), 2)}
                                      for c in C.CLASSES}
    if (C.MODELS / "student_split.json").exists() and (C.PROC / rows[0]["sample_id"] / "student_labels.npy").exists():
        summary["student_vs_teacher"] = student_vs_teacher()
    for name in ("uniform", "stratified"):
        res = point_check(name)
        if res:
            summary[f"ai_points_{name}"] = res
    spv = superpixel_val()
    if spv:
        summary["ai_superpixels_held_out"] = spv
    t = C.OUT_MET / "inference_timing.json"
    if t.exists():
        summary["inference_timing"] = json.load(open(t))
    C.write_json(C.OUT_MET / "validation_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
