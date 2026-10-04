"""Batch-ID stage, step 4: evidence maps, per-sample facts JSON, and end-to-end prediction for new images.

Evidence (exact, not attention): every DINO head is StandardScaler -> PCA -> logistic regression on a weighted
mean of patch tokens, so it folds into logits = A x + c0, and token i contributes A t_i w_i / sum(w). The map of
contribution[winner] - contribution[runner-up] (mean over the fixed head grid and, for head R, over region
types) is the evidence map; per-token contributions sum to the head's logits (reconstruction error reported).
The S head's evidence is per feature: LDA coefficient x robust z.

Usage:
  python -m src.bid_explain samples                    # JSON + evidence PNG for the 31 training samples
  python -m src.bid_explain predict --bse X_BSE.tif --inlens X_Inlens.tif --out outputs/batchid/new/X
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import config as C
from .bid_features import S_FEATURES, features as label_features
from .bid_model import BATCHES, token_phase_weights

OUT = C.ROOT / "outputs" / "batchid"
PATCH = 14


def fold_head(sc, pca, lr):
    """logits = A @ x + c0 for x in raw feature space."""
    W = lr.coef_ @ pca.components_ / sc.scale_            # (3, F)
    c0 = lr.intercept_ - lr.coef_ @ pca.components_ @ (sc.mean_ / sc.scale_ + pca.mean_)
    return W, c0


def dino_evidence(model, tokens, weights):
    """tokens (hp, wp, F); weights (hp, wp, 4) phase fractions. Returns logits (3,), per-token contrib (hp, wp, 3)."""
    hp, wp, nf = tokens.shape
    heads = model["dino_heads"]
    if model["dino_head"] == "R":
        per = len(model["grid"])
        regions = [C.CLASSES.index(r) for r in model["regions"]]
        contrib = np.zeros((hp, wp, 3))
        logits = np.zeros(3)
        for j, k in enumerate(regions):
            w = weights[..., k]
            s = max(w.sum(), 1e-6)
            for sc, pca, lr in heads[j * per:(j + 1) * per]:
                A, c0 = fold_head(sc, pca, lr)
                contrib += (tokens @ A.T) * (w / s)[..., None] / (per * len(regions))
                logits += (A @ ((tokens * w[..., None]).sum((0, 1)) / s) + c0) / (per * len(regions))
        return logits, contrib
    # texton head T: logits = A sqrt(h) + c0 with h_j = n_j / N. Giving each token in texton j the share
    # A[:, j] / (N sqrt(h_j)) makes the per-token contributions sum exactly to A sqrt(h).
    km = model["kmeans"]
    valid = weights.sum(-1) > 0.5
    assign = km.predict(tokens.reshape(-1, nf).astype(km.cluster_centers_.dtype)).reshape(hp, wp)
    n_valid = max(int(valid.sum()), 1)
    h = np.bincount(assign[valid], minlength=km.n_clusters) / n_valid
    contrib = np.zeros((hp, wp, 3))
    logits = np.zeros(3)
    for sc, pca, lr in heads:
        A, c0 = fold_head(sc, pca, lr)
        share = A[:, assign] / (n_valid * np.sqrt(np.maximum(h[assign], 1e-12)))     # (3, hp, wp)
        contrib += np.where(valid[..., None], share.transpose(1, 2, 0), 0) / len(heads)
        logits += (A @ np.sqrt(h) + c0) / len(heads)
    return logits, contrib


def s_evidence(model, f):
    x = np.array([f[k] for k in model["S_features"]])
    z = np.clip((x - model["S_median"]) / model["S_iqr"], -3, 3)
    lda = model["lda"]
    proba = lda.predict_proba(z[None])[0]
    contrib = lda.coef_ * z[None]                         # (3, 5)
    return proba, z, contrib


def explain(model, sid, lab, tokens, f, extra):
    weights = token_phase_weights(lab, *tokens.shape[:2])
    logits_d, contrib = dino_evidence(model, tokens, weights)
    e = np.exp(logits_d - logits_d.max())
    p_d = e / e.sum()
    p_s, z, s_contrib = s_evidence(model, f)
    p_f = (p_s + p_d) / 2
    order = np.argsort(p_f)[::-1]
    win, run = int(order[0]), int(order[1])
    ev = contrib[..., win] - contrib[..., run]
    recon = float(np.abs(contrib.sum((0, 1)) - (logits_d - _c0_mean(model))).max())
    # top evidence regions
    valid_tok = weights.sum(-1) > 0.5
    thr = np.percentile(ev[valid_tok], 90)
    top = (ev >= thr) & valid_tok
    cc, n = ndi.label(top)
    regions = []
    for i in np.argsort([-ev[cc == j].sum() for j in range(1, n + 1)])[:3]:
        ys, xs = np.nonzero(cc == i + 1)
        y0, y1, x0, x1 = ys.min() * PATCH, (ys.max() + 1) * PATCH, xs.min() * PATCH, (xs.max() + 1) * PATCH
        sub = lab[y0:y1, x0:x1]
        v = sub != C.EXCLUDED
        regions.append({"bbox_um": [round(v_ * C.NM_PER_PX / 1000, 2) for v_ in (x0, y0, x1, y1)],
                        "evidence_share": round(float(ev[cc == i + 1].sum() / max(ev[ev > 0].sum(), 1e-9)), 3),
                        "local_phase_pct": {c: round(100 * float((sub[v] == k).mean()), 1) for k, c in enumerate(C.CLASSES)} if v.any() else {}})
    top_w = weights[top].sum(0)
    all_w = weights[valid_tok].sum(0)
    excluded_share = float((1 - weights.sum(-1))[ev > thr].mean()) if (ev > thr).any() else 0.0
    from .map_text import describe
    map_text = describe(lab, ev, weights, BATCHES[win], BATCHES[run])
    return {
        "sample_id": sid,
        "predicted_batch": BATCHES[win], "runner_up": BATCHES[run],
        "probabilities": {"fused_F": dict(zip(BATCHES, np.round(p_f, 3).tolist())),
                          "S_head": dict(zip(BATCHES, np.round(p_s, 3).tolist())),
                          "dino_head": dict(zip(BATCHES, np.round(p_d, 3).tolist()))},
        "features_S": {k: {"value": round(float(f[k]), 4), "robust_z": round(float(zz), 2),
                           "push_toward_winner_vs_runner_up": round(float(s_contrib[win, i] - s_contrib[run, i]), 3)}
                       for i, (k, zz) in enumerate(zip(model["S_features"], z))},
        "evidence": {"phase_share_in_top10pct_evidence_pct": {c: round(100 * float(top_w[k] / max(top_w.sum(), 1e-9)), 1) for k, c in enumerate(C.CLASSES)},
                     "phase_share_whole_image_pct": {c: round(100 * float(all_w[k] / max(all_w.sum(), 1e-9)), 1) for k, c in enumerate(C.CLASSES)},
                     "top_regions": regions, "reconstruction_error": recon,
                     "shortcut_alarm": excluded_share > 0.3},
        "map_text": map_text,
        **extra,
    }, ev


def _c0_mean(model):
    return np.mean([fold_head(*h)[1] for h in model["dino_heads"]], 0)


PROBE_SKIP = ("batch", "sample_id")


def qc_and_confidence(facts, lab, probes, cu_foil, manual_exclusion, self_id=None):
    """QC flags, acquisition (session) check, novelty flags and a confidence tier, added to `facts` in place."""
    train = C.read_csv(C.OUT_MET / "bid_session_probes.csv")
    keys = [k for k in train[0] if k not in PROBE_SKIP]
    T = np.array([[float(r[k]) for k in keys] for r in train])
    mu, sd = T.mean(0), T.std(0) + 1e-9
    Z = (T - mu) / sd
    z = (np.array([float(probes[k]) for k in keys]) - mu) / sd
    ids = [r["sample_id"] for r in train]
    dist = np.sqrt(((Z - z) ** 2).sum(1))
    if self_id in ids:
        dist[ids.index(self_id)] = np.inf
    nn = int(np.argmin(dist))
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn_max = float(D.min(1).max())
    qc_groups = {r["sample_id"]: r["height_group"] for r in C.read_csv(C.QC)}
    acq_novel = bool(dist[nn] > nn_max)
    feat_novel = [k for k, v in facts["features_S"].items() if abs(v["robust_z"]) >= 3]
    valid = lab != C.EXCLUDED
    facts["qc"] = {"excluded_pct": round(100 * float((~valid).mean()), 2), "cu_foil_detected": bool(cu_foil),
                   "manual_exclusion": bool(manual_exclusion), "detectors_used": ["BSE", "Inlens"]}
    facts["acquisition"] = {
        "probes": {k: round(float(probes[k]), 3) for k in keys},
        "nearest_training_acquisition": {"sample_id": ids[nn], "batch": train[nn]["batch"],
                                         "height_group": qc_groups.get(ids[nn]), "distance_z": round(float(dist[nn]), 2)},
        "novel_acquisition": acq_novel,
        "warning": "acquisition properties alone predict batch at 0.68 balanced accuracy in training; "
                   "a prediction that matches the nearest-acquisition batch may reflect imaging session, not material"}
    facts["novelty"] = {"unusual_features_abs_robust_z_ge_3": feat_novel, "novel_acquisition": acq_novel}
    p = sorted(facts["probabilities"]["fused_F"].values(), reverse=True)
    top, margin = p[0], p[0] - p[1]
    reasons = []
    if facts["evidence"]["shortcut_alarm"]:
        reasons.append("evidence sits on excluded / junk regions")
    if acq_novel:
        reasons.append("acquisition settings unlike any training image")
    if feat_novel:
        reasons.append(f"unusual feature values (|robust z| >= 3): {', '.join(feat_novel)}")
    if reasons or top < 0.5:
        tier = "Low"
        if top < 0.5:
            reasons.append(f"top probability {top:.2f} < 0.5")
    elif top >= 0.7 and margin >= 0.3:
        tier = "High"
    else:
        tier = "Medium"
        reasons.append(f"top probability {top:.2f}, margin {margin:.2f}")
    pair = {facts["predicted_batch"], facts["runner_up"]}
    if pair == {"Batch_1", "Batch_2"} and tier == "High":
        tier = "Medium"
        reasons.append("Batch_1 vs Batch_2 is not reliably separable (leave-one-out recall Batch_2 2/7)")
    facts["confidence"] = {"tier": tier, "reasons": reasons or ["top probability >= 0.7 and margin >= 0.3, no alarms"]}
    return facts


def evidence_png(bse, ev, path):
    h, w = bse.shape
    up = np.kron(ev, np.ones((PATCH, PATCH)))[:h, :w]
    up = np.pad(up, ((0, h - up.shape[0]), (0, w - up.shape[1])))
    m = np.abs(up).max() or 1
    g = np.repeat((bse * 255)[..., None], 3, -1).astype(np.float32) * 0.6
    pos, neg = np.clip(up / m, 0, 1), np.clip(-up / m, 0, 1)
    g[..., 0] += 255 * 0.4 * pos
    g[..., 2] += 255 * 0.4 * neg
    Image.fromarray(g.clip(0, 255).astype(np.uint8)).save(path, quality=88)


def phase_block(lab):
    v = lab[lab != C.EXCLUDED]
    four = {c: round(100 * float((v == k).mean()), 2) for k, c in enumerate(C.CLASSES)}
    return {"four_class_pct": four,
            "three_class_pct": {"pore": four["pore"], "carbon (graphite + binder)": round(four["graphite"] + four["CBD"], 2),
                                "SiOx": four["SiOx"]},
            "reliability": "pore / carbon / SiOx are reliable (87 % agreement with an independent annotator); "
                           "the graphite-vs-binder split is experimental (binder precision ~50 %)"}


def summary_sentences(facts):
    """A short plain-English summary built only from numbers already in `facts` (for a text-only agent)."""
    d, ph, fp = facts["decision"], facts["phases"], facts.get("fingerprint_model", {})
    tx, rr = facts.get("texture_model", {}), facts.get("material_range_rule")
    k = d["answer_type"]
    out = []
    if k == "refused":
        out.append("Answer: refused; " + d["reasons"][0] + ".")
    elif k == "known_location":
        out.append(f"Answer: {d['answer']} (certain); " + d["reasons"][0] + ".")
    elif k in ("specific", "forced"):
        out.append(f"Answer: {d['answer']} ({d['confidence']} confidence); the imaging side (fingerprint) "
                   f"picked {d['imaging_side_pick']} and the material side (segmentation + DINOv2) {d['material_side_pick']}.")
    elif k == "specific_by_material_range":
        out.append(f"Answer: {d['answer']} ({d['confidence']} confidence); the two sides split between Batch_1 and "
                   f"Batch_2, and the GET4 material-range rule decided.")
    elif k == "pair":
        out.append(f"Answer: Batch_1 or Batch_2 ({d['confidence']} confidence); the imaging side picked "
                   f"{d['imaging_side_pick']} and the material side {d['material_side_pick']}.")
    else:
        out.append(f"Answer: unsure, leaning {d['leaning']}; the imaging side picked {d['imaging_side_pick']} and the "
                   f"material side {d['material_side_pick']}.")
    if k not in ("refused", "known_location"):
        out.append("Reasons: " + "; ".join(d["reasons"]) + ".")
        tr = ((d.get("track_record") or {}).get("LOO") or {}).get("this_confidence_tier")
        if tr:
            out.append(f"Track record of {d['confidence']}-confidence answers: {tr['correct']} right ({tr['accuracy']:.0%}) "
                       f"in nested leave-one-location-out tests.")
    t = ph["three_class_pct"]
    out.append(f"Composition: pore {t['pore']:.1f} %, carbon (graphite + binder) {t['carbon (graphite + binder)']:.1f} %, "
               f"SiOx {t['SiOx']:.1f} % (binder alone {ph['four_class_pct']['CBD']:.1f} %, experimental).")
    if fp.get("top_measurements"):
        m = fp["top_measurements"][0]
        out.append(f"Strongest imaging-fingerprint cue ({m['view']} image): {m['meaning']} is "
                   f"{'higher' if m['direction'] == 'high' else 'lower'} than typical (robust z {m['robust_z']:+.1f}).")
    if tx.get("top_features"):
        m = tx["top_features"][0]
        out.append(f"Strongest texture cue (BSE tiles): {m['meaning']} ({m['feature']}) is "
                   f"{'higher' if m['mean_z'] > 0 else 'lower'} than typical (z {m['mean_z']:+.1f}).")
    if rr:
        out.append("Material-range check (GET4 95 % error bars): " + "; ".join(
            f"{q['phase']} {q['pct']:.1f} % [{q['ci95'][0]:.1f}, {q['ci95'][1]:.1f}] fits the range of "
            f"{', '.join(q['fits_ranges_of']) or 'no batch'}" for q in rr["phases"]) + ".")
    out += facts.get("map_text", {}).get("sentences", [])[:4]
    out.append("Batch_1 and Batch_2 are statistically indistinguishable in this dataset; treat any Batch_1-vs-Batch_2 "
               "call as low-certainty.")
    return out


FIELD_GUIDE = ("Read `decision` first: it is the final answer (a batch, 'Batch_1 or Batch_2', 'unsure (leaning X)', "
               "a known location, or refused), with its confidence tier and track record. `known_location_check` says "
               "whether the image is a training image or shows a known location. The decision combines two sides: the "
               "imaging side (`fingerprint_model`) and the material side (`material_model`: segmentation features + DINOv2 "
               "texture); `material_range_rule` can settle a Batch_1-or-Batch_2 split. `texture_model` (tile texture) is "
               "reported for information only: it lowered held-out accuracy, so it does not vote. The individual "
               "models' picks are NOT the final answer. `phases` is the composition. `material_model.map_text` describes the "
               "segmentation and evidence maps in words. `forced_mode` is the always-answer variant and `decision_v2` the "
               "previous rule, for comparison. All numbers are measured; only `summary_sentences` is generated text, "
               "filled in from these numbers.")
MATERIAL_KEYS = ("predicted_batch", "runner_up", "probabilities", "confidence", "features_S", "evidence", "map_text",
                 "validation_reference", "loo_prediction", "note", "features_half_split_sd")


def v2_layout(facts):
    """Order fields for a text-only reader: final decision first, the two models' details after."""
    mat = {k: facts.pop(k) for k in MATERIAL_KEYS if k in facts}
    mat["what_this_is"] = "material model: 5 segmentation features (S head) + DINOv2 texture head, fused"
    facts.pop("phase_fractions_pct", None)
    head = {"summary_sentences": summary_sentences({**facts, "map_text": mat.get("map_text", {})}),
            "field_guide": FIELD_GUIDE, "sample_id": facts.pop("sample_id")}
    if "true_batch" in facts:
        head["true_batch"] = facts.pop("true_batch")
    order = ("decision", "phases", "known_location_check", "material_range_rule", "fingerprint_model", "texture_model")
    out = {**head, **{k: facts.pop(k) for k in order if k in facts}, "material_model": mat, **facts}
    # copies of three v1 material-model fields that the team's backend (backend/app/routers/lucas.py) reads
    compat = {k: mat[k] for k in ("loo_prediction", "confidence", "validation_reference") if k in mat}
    if compat:
        out["frontend_compat_note"] = ("loo_prediction / confidence / validation_reference below are copies of "
                                       "material_model fields for the team frontend; they are NOT the answer - read decision")
        out.update(compat)
    return out


def flags_from(facts):
    f = []
    if facts.get("evidence", {}).get("shortcut_alarm"):
        f.append("evidence sits on excluded or junk regions")
    nov = facts.get("novelty", {})
    if nov.get("novel_acquisition"):
        f.append("imaging conditions unlike any training image")
    if nov.get("unusual_features_abs_robust_z_ge_3"):
        f.append("unusual feature values: " + ", ".join(nov["unusual_features_abs_robust_z_ge_3"]))
    return f


def cmd_samples(args):
    model = pickle.load(open(C.MODELS / "bid_model.pkl", "rb"))
    scores = json.load(open(C.OUT_MET / "bid_scores.json"))
    null = json.load(open(C.OUT_MET / "bid_null.json")) if (C.OUT_MET / "bid_null.json").exists() else {}
    feats = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_features.csv")}
    probes = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_session_probes.csv")}
    from .decision import decide
    from .fingerprint_model import MODEL as FP_MODEL, calibration_for as fp_calibration, predict as fp_predict
    dec_eval = json.load(open(C.OUT_MET / "decision_eval.json"))["LOO"]
    from . import v3 as V3
    from .texture_model import load_tiles
    v3_loo = json.load(open(V3.EVAL))["LOO"]["per_sample"]
    v3_model = pickle.load(open(V3.MODEL, "rb"))
    tx_tiles = load_tiles()
    fp_model = pickle.load(open(FP_MODEL, "rb"))
    fpv = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "fingerprint_values.csv")}
    fp_feats = {s: {v: {k: float(r[k]) for k in r if k.startswith(v + "_")} for v in ("bse", "inlens", "se2")} for s, r in fpv.items()}
    ref = {"LOO_F": scores["LOO_F"]["balanced_accuracy"], "LOO_S": scores["LOO_S"]["balanced_accuracy"],
           "LOGO_F": scores["LOGO_F"]["balanced_accuracy"], "nuisance_only_LOO": scores["nuisance_only_LOO"]["balanced_accuracy"],
           "permutation": null}
    OUT.mkdir(parents=True, exist_ok=True)
    for batch, sid in C.sample_ids():
        lab = np.load(C.proc_dir(sid) / "final_labels.npy")
        tokens = np.concatenate([np.load(C.proc_dir(sid) / f"dino_{d}_50nm.npy").astype(np.float32) for d in ("BSE", "Inlens")], -1)
        f = {k: float(feats[sid][k]) for k in S_FEATURES}
        loo = scores["per_sample_LOO"][sid]
        facts, ev = explain(model, sid, lab, tokens, f, {
            "true_batch": batch, "note": "training sample: the probabilities above come from a model that saw it; "
                                         "the honest prediction is loo_prediction",
            "loo_prediction": loo, "validation_reference": ref,
            "features_half_split_sd": {k: float(feats[sid][f"{k}_half_sd"]) for k in S_FEATURES}})
        meta = json.load(open(C.proc_dir(sid) / "meta.json"))
        excl = json.load(open(C.ROOT / "data" / "exclusions.json"))
        qc_and_confidence(facts, lab, probes[sid], meta["cu_foil_px"] > 0, sid in excl, self_id=sid)
        # v2: phase block, fingerprint model and decision - from HONEST leave-one-location-out probabilities
        facts["phases"] = phase_block(lab)
        ps = dec_eval["per_sample"][sid]
        fp_views = ps["fingerprint_views"]
        facts["fingerprint_model"] = {
            "probabilities_leave_one_out": dict(zip(BATCHES, ps["fingerprint_p"])),
            "per_view_leave_one_out": {k: dict(zip(BATCHES, v)) for k, v in fp_views.items()},
            "predicted_batch_leave_one_out": BATCHES[int(np.argmax(ps["fingerprint_p"]))],
            "top_measurements": fp_predict(fp_feats[sid], fp_model)["top_measurements"],
            "calibration": fp_calibration(max(ps["fingerprint_p"])),
            "note": "probabilities are leave-one-location-out (honest); top_measurements come from the model trained on all 31"}
        views_agree = len({int(np.argmax(v)) for v in fp_views.values()}) == 1
        facts["decision_v2"] = decide(ps["fingerprint_p"], ps["material_p"], views_agree, flags_from(facts))
        # v3: v2 + the teammate's texture model, calibration, GET4 range rule and matcher (nested leave-one-out)
        p3 = v3_loo[sid]
        va3 = len({int(np.argmax(v)) for v in p3["raw"]["fingerprint_views"].values()}) == 1
        dec = V3.decide(p3["calibrated"]["imaging_side"], p3["calibrated"]["material_side"], va3, flags_from(facts),
                        p3["fold"]["high_threshold"], p3["range_rule"])
        dec["track_record"] = V3.track_record(dec)
        dec["basis"] = "nested leave-one-location-out: models, calibration and thresholds chosen without this location"
        facts["decision"] = dec
        facts["known_location_check"] = {"status": "training location",
                                         "note": "this is one of the 31 reference locations; to demo it as a new image "
                                                 "use `predict --holdout`"}
        facts["material_range_rule"] = {**p3["range_rule"], "material_error_bars": "GET4 (teammate), 95 % = 1.96 x SE"}
        ct = np.array(p3["calibrated"]["texture"])
        facts["texture_model"] = V3.texture_block(v3_model["texture"], {v: tx_tiles[v][sid] for v in V3.TX_VIEWS}, ct,
                                                  {v: np.array(x) for v, x in p3["raw"]["texture_views"].items()},
                                                  BATCHES[int(np.argsort(ct)[-1])], BATCHES[int(np.argsort(ct)[-2])])
        facts["texture_model"]["note"] = ("probabilities are nested leave-one-out (honest); top_features come from the "
                                          "model trained on all 31")
        facts["forced_mode"] = p3["forced"]
        facts = v2_layout(facts)
        C.write_json(OUT / f"{sid}.json", facts)
        evidence_png(C.load_channels(sid, ["BSE"])["BSE"], ev, OUT / f"{sid}_evidence.jpg")
    print(f"facts JSON + evidence maps for {len(C.sample_ids())} samples -> {OUT}")


def cmd_predict(args):
    import torch
    from .dino import load_model as load_dino, token_map
    from .physics import constrain_siox
    from .predict import prepare
    from .student import CKPT, build_model, segment, torch_runner
    from .teacher import clean
    from skimage.filters import threshold_multiotsu
    t0 = time.time()
    timings = {}
    x, exclude = prepare(args.bse, args.inlens)
    timings["read_preprocess_s"] = time.time() - t0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir) if args.models_dir else C.MODELS
    seg = build_model(pretrained=False).to(dev)
    seg.load_state_dict(torch.load(models_dir / "student.pt", map_location=dev))
    seg.eval()
    t = time.time()
    probs = constrain_siox(segment(x, torch_runner(seg, dev)), x[0], exclude, class_axis=0)
    lab = clean(probs.argmax(0), exclude)
    timings["segmentation_s"] = time.time() - t
    t = time.time()
    t_low = float(threshold_multiotsu(x[0][~exclude], classes=3)[0])
    f = label_features(lab, x[0], t_low)
    timings["features_s"] = time.time() - t
    t = time.time()
    dmodel = load_dino(dev)
    img = [np.where(exclude, np.median(x[i][~exclude]), x[i]) for i in range(2)]
    tokens = np.concatenate([token_map(dmodel, a, dev) for a in img], -1)
    timings["dino_s"] = time.time() - t
    from .fingerprint_model import fit_model as fp_fit, measure_view, predict as fp_predict
    from .bid_model import fit_model as mat_fit
    if args.holdout:
        # demo mode: both batch models retrained WITHOUT this location, so the demo is an honest new-image test
        model, fp_model = mat_fit(exclude=[args.holdout]), fp_fit(exclude=[args.holdout])
    else:
        model = pickle.load(open(models_dir / "bid_model.pkl", "rb"))
        fp_model = pickle.load(open(models_dir / "fingerprint_model.pkl", "rb"))
    t = time.time()
    fp_feats = {"bse": measure_view(args.bse, "bse"), "inlens": measure_view(args.inlens, "inlens")}
    if args.etd:
        fp_feats["se2"] = measure_view(args.etd, "se2")
    fp = fp_predict(fp_feats, fp_model)
    timings["fingerprint_s"] = time.time() - t
    t = time.time()
    scores = json.load(open(C.OUT_MET / "bid_scores.json"))
    stem = Path(args.bse).stem
    sample_id = stem[:-4] if stem.upper().endswith("_BSE") else stem           # img_X_BSE -> img_X
    facts, ev = explain(model, sample_id, lab, tokens, f, {
        "inputs": {"BSE": str(args.bse), "Inlens": str(args.inlens), "detectors_used": ["BSE", "Inlens"]},
        "validation_reference": {"LOO_F": scores["LOO_F"]["balanced_accuracy"], "LOGO_F": scores["LOGO_F"]["balanced_accuracy"]}})
    from .bid_features import session_probes
    from .preprocess import cu_foil, read_half
    probes = session_probes(sample_id, {"BSE": args.bse, "Inlens": args.inlens}, lab)
    qc_and_confidence(facts, lab, probes, cu_foil(read_half(args.bse)).any(), False)
    from .decision import decide
    facts["phases"] = phase_block(lab)
    facts["fingerprint_model"] = fp
    mat_p = facts["probabilities"]["fused_F"]
    facts["decision_v2"] = decide(fp["probabilities"], mat_p, fp["views_agree"], flags_from(facts))
    facts["inputs"]["detectors_used_by_fingerprint"] = fp["views_used"]
    timings["classify_explain_s"] = time.time() - t
    # ---- v3: guard + known-location matcher, texture model, calibration, GET4 range rule
    from . import known_spot as KS, texture_model as TXM, v3 as V3
    from .fingerprint_model import view_proba as fp_view_proba
    t = time.time()
    raws = {"bse": KS.read_raw(args.bse), "inlens": KS.read_raw(args.inlens)}
    if args.etd:
        raws["se2"] = KS.read_raw(args.etd)
    names = {"bse": "BSE", "inlens": "Inlens", "se2": "ETD/SE"}
    check = KS.Index(exclude_sids=[args.holdout] if args.holdout else ()).check({names[v]: a for v, a in raws.items()})
    timings["known_location_s"] = time.time() - t
    t = time.time()
    if args.holdout:
        D = V3.load_data()
        parts = V3.fit_parts(D, [i for i, s_ in enumerate(D["sids"]) if s_ != args.holdout])
        fold = json.load(open(V3.EVAL))["LOO"]["per_sample"][args.holdout]["fold"]
        T, t_high = fold["temperatures"], fold["high_threshold"]
    else:
        parts = pickle.load(open(models_dir / "v3_model.pkl", "rb"))
        T, t_high = parts["temperatures"], parts["high_threshold"]
    tiles = {v: TXM.tile_features(a, exclude) for v, a in raws.items()}
    tx_raw = {v: TXM.view_proba(parts["texture"][v], tiles[v]) for v in tiles}
    fp_x = {v: np.array([fp_feats[v][k] for k in fp_model["views"][v]["features"]]) for v in fp_feats}
    fp_raw = {v: fp_view_proba(fp_model["views"][v], fp_x[v])[0] for v in fp_feats}
    lv = {"fp": V3.geomean([p_[None] for p_ in fp_raw.values()]), "tx": V3.geomean([p_[None] for p_ in tx_raw.values()]),
          "mat": np.array([[mat_p[b] for b in BATCHES]])}
    p_img, p_mat = V3.sides(lv, T)
    timings["texture_s"] = time.time() - t
    t = time.time()
    rng = V3.range_rule(V3.material_se(lab), parts["ranges"])
    timings["get4_range_rule_s"] = time.time() - t
    dec = V3.decide(p_img[0], p_mat[0], fp["views_agree"], flags_from(facts), t_high, rng)
    dec["track_record"] = V3.track_record(dec)
    fp_cal = V3.apply_T(lv["fp"], T["fp"])[0]
    p12 = np.mean([V3.b12_proba(parts["b12"][v], fp_x[v]) for v in fp_x], 0)
    f_ans, f_route = V3.forced(fp_cal, rng, p12)
    facts["forced_mode"] = {"answer": f_ans, "route": f_route}
    if check["status"] == "training_image":
        dec = {"answer": "refused (training image)", "answer_type": "refused", "confidence": "n/a",
               "reasons": [check["note"]], "model_answer_for_reference": dec}
    elif check["status"] == "known_location":
        dec = {"answer": check["batch"], "answer_type": "known_location", "confidence": "Certain (known location)",
               "reasons": [check["note"]], "model_answer_for_reference": dec}
    elif args.forced:
        dec = {**dec, "answer": f_ans, "answer_type": "forced", "abstaining_answer": dec["answer"], "forced_route": f_route}
    dec["basis"] = ("all models retrained, calibrated and thresholded without this location (hold-out demo)"
                    if args.holdout else "models trained on all 31 known locations")
    facts["decision"] = dec
    facts["known_location_check"] = check
    facts["material_range_rule"] = {**rng, "material_error_bars": "GET4 (teammate), 95 % = 1.96 x SE"}
    ct = V3.apply_T(lv["tx"], T["tx"])[0]
    facts["texture_model"] = V3.texture_block(parts["texture"], tiles, ct, tx_raw, BATCHES[int(np.argsort(ct)[-1])],
                                              BATCHES[int(np.argsort(ct)[-2])])
    facts = v2_layout(facts)
    timings["total_s"] = time.time() - t0
    facts["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "facts.json", facts)
    Image.fromarray(lab).save(out / "labels.png")
    Image.fromarray(C.overlay_rgb(x[0], lab, alpha=0.45)).save(out / "overlay.jpg", quality=88)
    evidence_png(x[0], ev, out / "evidence.jpg")
    print(json.dumps({"decision": facts["decision"], "timings_s": facts["timings_s"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["samples", "predict"])
    ap.add_argument("--bse")
    ap.add_argument("--inlens")
    ap.add_argument("--out")
    ap.add_argument("--models-dir", help="folder with student.pt, bid_model.pkl, fingerprint_model.pkl (default: models/)")
    ap.add_argument("--etd", help="optional ETD or SE image of the same field (used by the fingerprint model)")
    ap.add_argument("--holdout", help="demo mode: retrain all batch models without this known location id")
    ap.add_argument("--forced", action="store_true", help="always answer (the teammate's forced mode)")
    args = ap.parse_args()
    {"samples": cmd_samples, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
