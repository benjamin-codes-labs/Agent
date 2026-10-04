"""Text descriptions of the label map and the evidence map, so the LLM agent receives text only.

describe(lab, ev, weights) returns a dict with:
  evidence_by_phase   share of positive / negative evidence on each phase, and enrichment vs the phase's area share
  spatial             3 x 3 grid phase fractions with enriched / depleted regions named in words, and a
                      top-to-bottom profile (image vertical = through-thickness direction of the electrode)
  objects             largest pores and SiOx particles (size, shape, location), SiOx clustering (Clark-Evans)
  evidence_regions    strongest positive and negative evidence regions, located in words, with local phase mix
  sentences           plain-English statements built only from the numbers above
lab: label map (50 nm/px, 0 pore, 1 graphite, 2 SiOx, 3 CBD, 255 excluded); ev: per-token evidence
(winner minus runner-up, one value per 14 x 14 px token); weights: per-token phase fractions (hp, wp, 4).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from . import config as C

PX = C.NM_PER_PX / 1000
PATCH = 14
NAMES = {"pore": "pores", "graphite": "graphite", "SiOx": "SiOx", "CBD": "binder"}
ROWS, COLS = ("top", "middle", "bottom"), ("left", "centre", "right")


def where(y, x, h, w):
    """Location of pixel (y, x) in words, e.g. 'top-left' or 'middle-centre'."""
    return f"{ROWS[min(int(3 * y / h), 2)]}-{COLS[min(int(3 * x / w), 2)]}"


def evidence_by_phase(ev, weights):
    valid = weights.sum(-1) > 0.5
    area = weights[valid].sum(0) / weights[valid].sum()
    out = {}
    for sign, name in ((1, "positive"), (-1, "negative")):
        e = np.clip(sign * ev, 0, None) * valid
        tot = e.sum()
        share = (weights * e[..., None]).sum((0, 1)) / tot if tot > 0 else np.zeros(4)
        out[name] = {c: {"share_pct": round(100 * float(share[k]), 1),
                         "enrichment_vs_area": round(float(share[k] / area[k]), 2) if area[k] > 0 else None}
                     for k, c in enumerate(C.CLASSES)}
    out["area_pct"] = {c: round(100 * float(area[k]), 1) for k, c in enumerate(C.CLASSES)}
    pos, neg = np.clip(ev, 0, None)[valid].sum(), np.clip(-ev, 0, None)[valid].sum()
    out["positive_share_of_total_evidence_pct"] = round(100 * float(pos / max(pos + neg, 1e-12)), 1)
    return out


def spatial(lab):
    h, w = lab.shape
    valid = lab != C.EXCLUDED
    mean = {c: float((lab[valid] == k).mean()) for k, c in enumerate(C.CLASSES)}
    grid, notable = {}, []
    for i, r in enumerate(ROWS):
        for j, cname in enumerate(COLS):
            cell = lab[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            v = cell[cell != C.EXCLUDED]
            if not len(v):
                continue
            fr = {c: float((v == k).mean()) for k, c in enumerate(C.CLASSES)}
            grid[f"{r}-{cname}"] = {c: round(100 * f, 1) for c, f in fr.items()}
            for c in C.CLASSES:
                diff = fr[c] - mean[c]
                if mean[c] > 0 and abs(diff) >= 0.02 and (fr[c] / mean[c] >= 1.3 or fr[c] / mean[c] <= 0.7):
                    notable.append({"region": f"{r}-{cname}", "phase": c, "region_pct": round(100 * fr[c], 1),
                                    "image_pct": round(100 * mean[c], 1), "direction": "enriched" if diff > 0 else "depleted"})
    profile = {}
    for i, r in enumerate(ROWS):
        band = lab[i * h // 3:(i + 1) * h // 3]
        v = band[band != C.EXCLUDED]
        profile[r] = {c: round(100 * float((v == k).mean()), 1) for k, c in enumerate(C.CLASSES)}
    notable.sort(key=lambda d: -abs(d["region_pct"] - d["image_pct"]))
    return {"grid_3x3_pct": grid, "notable_regions": notable[:6], "top_to_bottom_pct": profile,
            "image_mean_pct": {c: round(100 * f, 1) for c, f in mean.items()}}


def objects(lab):
    h, w = lab.shape
    out = {}
    pore_lab, _ = ndi.label(lab == C.PORE)
    pores = sorted(regionprops(pore_lab), key=lambda p: -p.area)[:3]
    out["largest_pores"] = [{"area_um2": round(p.area * PX ** 2, 2), "length_um": round(p.major_axis_length * PX, 2),
                             "width_um": round(p.minor_axis_length * PX, 2),
                             "orientation": "horizontal" if abs(abs(p.orientation) - np.pi / 2) < np.pi / 6 else
                                            ("vertical" if abs(p.orientation) < np.pi / 6 else "diagonal"),
                             "location": where(*p.centroid, h, w),
                             "centre_um": [round(p.centroid[1] * PX, 1), round(p.centroid[0] * PX, 1)]} for p in pores]
    si_lab, n = ndi.label(ndi.binary_fill_holes(lab == C.SIOX))
    props = [p for p in regionprops(si_lab) if p.area * PX ** 2 >= 0.1]
    big = sorted(props, key=lambda p: -p.area)[:3]
    out["largest_SiOx_particles"] = [{"area_um2": round(p.area * PX ** 2, 2),
                                      "diameter_um": round(2 * np.sqrt(p.area * PX ** 2 / np.pi), 2),
                                      "location": where(*p.centroid, h, w),
                                      "centre_um": [round(p.centroid[1] * PX, 1), round(p.centroid[0] * PX, 1)]} for p in big]
    out["SiOx_particle_count"] = len(props)
    if len(props) > 3:
        cents = np.array([p.centroid for p in props]) * PX
        area_um2 = (lab != C.EXCLUDED).sum() * PX ** 2
        d, _ = cKDTree(cents).query(cents, k=2)
        r = float(d[:, 1].mean() / (0.5 / np.sqrt(len(cents) / area_um2)))
        out["SiOx_clark_evans"] = round(r, 2)
        out["SiOx_arrangement"] = "clustered" if r < 0.85 else ("evenly spread" if r > 1.15 else "roughly random")
        counts = {}
        for p in props:
            counts[where(*p.centroid, h, w)] = counts.get(where(*p.centroid, h, w), 0) + 1
        out["SiOx_count_by_region"] = counts
    return out


def evidence_by_region(ev, weights):
    """Share of positive / negative evidence in each 3 x 3 region vs that region's share of the valid image area."""
    hp, wp = ev.shape
    valid = weights.sum(-1) > 0.5
    out = {}
    for sign, name in ((1, "positive"), (-1, "negative")):
        e = np.clip(sign * ev, 0, None) * valid
        tot, area_tot = e.sum(), valid.sum()
        cells = {}
        for i, r in enumerate(ROWS):
            for j, c in enumerate(COLS):
                sl = (slice(i * hp // 3, (i + 1) * hp // 3), slice(j * wp // 3, (j + 1) * wp // 3))
                share = float(e[sl].sum() / tot) if tot > 0 else 0.0
                area = float(valid[sl].sum() / area_tot)
                cells[f"{r}-{c}"] = {"share_pct": round(100 * share, 1), "area_pct": round(100 * area, 1),
                                     "enrichment": round(share / area, 2) if area > 0 else None}
        out[name] = cells
    return out


def evidence_regions(ev, weights, lab, k=3):
    h, w = lab.shape
    valid = weights.sum(-1) > 0.5
    out = {}
    for sign, name in ((1, "positive"), (-1, "negative")):
        e = np.where(valid, sign * ev, 0)
        if (e > 0).sum() == 0:
            out[name] = []
            continue
        thr = np.percentile(e[e > 0], 90)
        cc, n = ndi.label(e >= thr)
        regs = []
        tot = np.clip(e, 0, None).sum()
        for i in range(1, n + 1):
            ys, xs = np.nonzero(cc == i)
            regs.append((e[cc == i].sum() / tot, ys, xs))
        regs.sort(key=lambda t: -t[0])
        items = []
        for share, ys, xs in regs[:k]:
            y0, y1, x0, x1 = ys.min() * PATCH, (ys.max() + 1) * PATCH, xs.min() * PATCH, (xs.max() + 1) * PATCH
            sub = lab[y0:y1, x0:x1]
            v = sub[sub != C.EXCLUDED]
            mix = {c: round(100 * float((v == kk).mean()), 1) for kk, c in enumerate(C.CLASSES)} if len(v) else {}
            items.append({"location": where((y0 + y1) / 2, (x0 + x1) / 2, h, w), "share_of_evidence_pct": round(100 * float(share), 1),
                          "bbox_um": [round(v_ * PX, 1) for v_ in (x0, y0, x1, y1)], "local_phase_pct": mix,
                          "dominant_phase": max(mix, key=mix.get) if mix else None})
        out[name] = items
    return out


def cap(t):
    return t[:1].upper() + t[1:]


def sentences(ebp, sp, ob, er, ebr, winner, runner):
    s = []
    pos, neg = ebp["positive"], ebp["negative"]
    top_pos = max(pos, key=lambda c: pos[c]["share_pct"])
    s.append(f"DINOv2 evidence map: {pos[top_pos]['share_pct']:.0f} % of the evidence for {winner} (over {runner}) lies on {NAMES[top_pos]}, "
             f"which covers {ebp['area_pct'][top_pos]:.0f} % of the image.")
    enr = [c for c in C.CLASSES if pos[c]["enrichment_vs_area"] and pos[c]["enrichment_vs_area"] >= 1.3 and ebp["area_pct"][c] >= 2]
    if enr:
        s.append("DINOv2 evidence for the prediction is concentrated on " + " and ".join(
            f"{NAMES[c]} ({pos[c]['enrichment_vs_area']:.1f}x its share of the image)" for c in enr) + ".")
    top_neg = max(neg, key=lambda c: neg[c]["share_pct"])
    s.append(f"DINOv2 evidence map: {neg[top_neg]['share_pct']:.0f} % of the evidence against {winner} (for {runner}) lies on {NAMES[top_neg]}.")
    s.append(f"In the DINOv2 evidence map alone, {ebp['positive_share_of_total_evidence_pct']:.0f} % of all evidence points toward {winner} "
             f"(the DINOv2 head is only one part of the material side).")
    for d in sp["notable_regions"][:3]:
        s.append(f"{cap(NAMES[d['phase']])} {'are' if d['phase'] == 'pore' else 'is'} {d['direction']} in the {d['region']} of the image "
                 f"({d['region_pct']:.0f} % vs {d['image_pct']:.0f} % overall).")
    prof = sp["top_to_bottom_pct"]
    for c in ("pore", "SiOx", "CBD"):
        a, b = prof["top"][c], prof["bottom"][c]
        if abs(a - b) >= 2 and max(a, b) / max(min(a, b), 0.1) >= 1.3:
            s.append(f"{cap(NAMES[c])} {'change' if c == 'pore' else 'changes'} through the thickness: {a:.0f} % in the top third vs {b:.0f} % in the bottom third.")
    if ob.get("largest_pores"):
        p = ob["largest_pores"][0]
        s.append(f"The largest pore ({p['area_um2']:.1f} um^2, {p['length_um']:.1f} x {p['width_um']:.1f} um, {p['orientation']}) is in the {p['location']}.")
    if ob.get("largest_SiOx_particles"):
        p = ob["largest_SiOx_particles"][0]
        s.append(f"The largest SiOx particle ({p['diameter_um']:.1f} um across) is in the {p['location']}.")
    if "SiOx_arrangement" in ob:
        s.append(f"The {ob['SiOx_particle_count']} SiOx particles are {ob['SiOx_arrangement']} (Clark-Evans index {ob['SiOx_clark_evans']}).")
    for name, label in (("positive", f"for {winner}"), ("negative", f"against {winner}")):
        cells = ebr[name]
        best = max(cells, key=lambda c: cells[c]["enrichment"] or 0)
        b = cells[best]
        if b["enrichment"] and b["enrichment"] >= 1.3:
            s.append(f"DINOv2 evidence {label} is concentrated in the {best} of the image ({b['share_pct']:.0f} % of it, "
                     f"in {b['area_pct']:.0f} % of the area).")
    return s


def describe(lab, ev, weights, winner, runner):
    ebp = evidence_by_phase(ev, weights)
    sp = spatial(lab)
    ob = objects(lab)
    er = evidence_regions(ev, weights, lab)
    ebr = evidence_by_region(ev, weights)
    return {"evidence_by_phase": ebp, "evidence_by_region": ebr, "spatial": sp, "objects": ob, "evidence_regions": er,
            "sentences": sentences(ebp, sp, ob, er, ebr, winner, runner),
            "notes": "Image vertical = through-thickness direction of the electrode; locations use a 3 x 3 grid "
                     "(top/middle/bottom x left/centre/right). Evidence = the batch model's DINO-head contribution "
                     "map (winner minus runner-up)."}
