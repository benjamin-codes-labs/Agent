"""Stage 2b: AI-annotated superpixels for the ambiguous regions (no human labelling).

Rule-based seeds (src.seeds) cover the easy pixels. Grey-floored open pores and carbon-binder vs particle rims
have no reliable intensity rule, so a vision model (Claude Opus 5.5) classifies small superpixels shown as
BSE | Inlens | ETD/SE crops with the superpixel outlined. A few already-seeded superpixels are mixed in, blind,
to measure how well the annotator agrees with the rules.

Steps (each a subcommand):
  prepare   SLIC superpixels per sample, choose items, render crops -> data/processed/ai_labels/
  pilot     one synchronous request on a few items, printed for inspection
  submit    send every item through the Message Batches API (50 % price, no long-lived connection)
  collect   download batch results -> outputs/metrics/ai_superpixel_labels.csv
  apply     write data/processed/<sid>/ai_labels.npy (train split) and ai_labels_val.npy (held-out split)

Round 2 (active learning, v2): `prepare-active` picks NEW superpixels where the v1 student is least confident,
focused on pore / CBD (the weakest classes); use `--round 2` with submit / collect. `apply` merges all rounds.

Usage: python -m src.ai_labels prepare|pilot|submit|collect|apply
       python -m src.ai_labels prepare-active; python -m src.ai_labels submit --round 2; python -m src.ai_labels collect --round 2 --wait
"""
import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries, slic

from . import config as C

AI_DIR = C.PROC / "ai_labels"
CROPS = AI_DIR / "crops"
ITEMS_CSV = AI_DIR / "items.csv"
BATCH_FILE = AI_DIR / "batch_id.txt"
LABELS_CSV = C.OUT_MET / "ai_superpixel_labels.csv"


def round_paths(rnd):
    """(items csv, batch-id file, labels csv) for labelling round `rnd` (1 = original, 2 = active learning)."""
    if rnd == 1:
        return ITEMS_CSV, BATCH_FILE, LABELS_CSV
    return AI_DIR / f"items_r{rnd}.csv", AI_DIR / f"batch_id_r{rnd}.txt", C.OUT_MET / f"ai_superpixel_labels_r{rnd}.csv"

MODEL = "claude-opus-5-5"
EFFORT = "max"
ITEMS_PER_REQUEST = 10
N_AMBIGUOUS, N_SEEDED = 32, 6        # per sample
CROP = 160                           # half-res px (8 um)
SCALE = 2.5
VAL_FRACTION = 0.2
LABELS = ["pore", "graphite", "SiOx", "CBD", "mixed", "uncertain"]

SYSTEM = """You are an expert in SEM microstructure analysis of lithium-ion battery electrodes.

The images are polished cross-sections of a calendered graphite + SiOx blended anode, imaged with three \
co-registered detectors at 50 nm/px. Each item shows the same 8 x 8 um area three times, left to right: \
BSE (atomic-number contrast; SiOx is brighter than carbon), Inlens (fine surface detail and edges; strong \
charging/channelling contrast, so its brightness varies between particles and samples), and ETD or SE \
(topography and shading). A thin yellow outline marks one small region (a superpixel, ~1 um) at the centre.

Classify the material that makes up the MAJORITY of the outlined region:
- pore: empty space. Includes deep pores (black in all detectors) AND open, unfilled pores whose floor shows \
material BELOW the polished plane: out-of-focus-looking or 3D-shaded surfaces, shadows in ETD/SE, no flat \
polished face, often ringed by bright Inlens edges. The pores were NOT resin-filled, so a pore floor can look \
grey in BSE.
- graphite: a polished face of a graphite particle: flat, uniform BSE grey, lamellar texture, basal-plane \
cracks, curtaining streaks; large flake-shaped particles.
- SiOx: polished face of a brighter (in BSE), angular, non-lamellar particle, including mottled or porous ones.
- CBD: carbon-binder domain in the polished plane: lacy, spongy, nanoporous material at particle necks, in \
pockets between particles, or as films on particle surfaces; same BSE grey as graphite but finely textured.
- mixed: the outline contains two or more classes in roughly equal amounts.
- uncertain: you cannot tell.

Judge each item independently. Use all three detectors. Be honest about confidence."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "label", "confidence", "reason"],
        "properties": {
            "id": {"type": "integer"},
            "label": {"type": "string", "enum": LABELS},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string", "description": "at most 15 words"},
        }}}},
}


# ---------------------------------------------------------------- prepare

def superpixels(ch, exclude):
    stack = np.stack([ch["BSE"], ch["Inlens"], ch["SE2"]], -1)
    n = int(stack.shape[0] * stack.shape[1] / 300)
    sp = slic(stack, n_segments=n, compactness=3, channel_axis=-1, start_label=1, mask=~exclude)
    return sp


def render(ch, sp, label_id, cy, cx, se2_name):
    h, w = sp.shape
    y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))
    region = sp[y0:y0 + CROP, x0:x0 + CROP] == label_id
    outline = find_boundaries(region, mode="inner")
    size = int(CROP * SCALE)
    panels = []
    for det in ("BSE", "Inlens", "SE2"):
        g = (ch[det][y0:y0 + CROP, x0:x0 + CROP] * 255).astype(np.uint8)
        rgb = np.repeat(g[..., None], 3, -1)
        rgb[outline] = [255, 230, 0]
        im = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
        ImageDraw.Draw(im).text((6, 4), det if det != "SE2" else se2_name, fill=(255, 60, 60))
        panels.append(np.asarray(im))
    gap = np.full((size, 8, 3), 255, np.uint8)
    return Image.fromarray(np.hstack([panels[0], gap, panels[1], gap, panels[2]]))


def prepare_sample(key):
    batch, sid = key
    rng = np.random.default_rng(int(hashlib.md5(sid.encode()).hexdigest()[:8], 16))
    ch = C.load_channels(sid)
    exclude = C.load_exclude(sid)
    seeds = np.load(C.proc_dir(sid) / "seeds.npy")
    meta = json.load(open(C.proc_dir(sid) / "meta.json"))
    sp = superpixels(ch, exclude)
    np.save(C.proc_dir(sid) / "superpixels.npy", sp.astype(np.int32))

    ids = np.arange(1, sp.max() + 1)
    area = ndi.sum(np.ones_like(sp), sp, ids)
    labelled = ndi.sum(seeds != C.EXCLUDED, sp, ids) / np.maximum(area, 1)
    cy, cx = [np.round(v).astype(int) for v in zip(*ndi.center_of_mass(np.ones_like(sp), sp, ids))]
    h, w = sp.shape
    inner = (cy > CROP // 2) & (cy < h - CROP // 2) & (area > 100)
    # majority seed class for seeded superpixels
    seed_major = np.full(len(ids), -1)
    for k in range(4):
        frac = ndi.sum(seeds == k, sp, ids) / np.maximum(area, 1)
        seed_major[frac >= 0.8] = k

    ambiguous = np.flatnonzero(inner & (labelled < 0.2))
    # stratify ambiguous superpixels by mean BSE x Inlens texture so the sample covers the variety
    bse_m = ndi.mean(ch["BSE"], sp, ids)
    tex = ndi.standard_deviation(ch["Inlens"], sp, ids)
    strata = (np.digitize(bse_m, np.quantile(bse_m[ambiguous], [0.25, 0.5, 0.75])) * 4
              + np.digitize(tex, np.quantile(tex[ambiguous], [0.25, 0.5, 0.75])))
    chosen = []
    for s in np.unique(strata[ambiguous]):
        pool = ambiguous[strata[ambiguous] == s]
        chosen += list(rng.choice(pool, min(len(pool), N_AMBIGUOUS // 16 + 1), replace=False))
    chosen = list(rng.permutation(chosen)[:N_AMBIGUOUS])
    seeded_pool = np.flatnonzero(inner & (seed_major >= 0))
    seeded = []
    for k in range(4):   # roughly balanced across seed classes
        pool = seeded_pool[seed_major[seeded_pool] == k]
        seeded += list(rng.choice(pool, min(len(pool), (N_SEEDED + 3) // 4), replace=False))
    seeded = list(rng.permutation(seeded)[:N_SEEDED])

    rows = []
    for idx in chosen + seeded:
        lab = int(ids[idx])
        name = f"{sid}_{lab}"
        render(ch, sp, lab, cy[idx], cx[idx], meta["se2_detector"]).save(CROPS / f"{name}.png")
        rows.append({"item": name, "batch": batch, "sample_id": sid, "superpixel": lab,
                     "cy": int(cy[idx]), "cx": int(cx[idx]), "area_px": int(area[idx]),
                     "kind": "seeded" if idx in seeded else "ambiguous",
                     "seed_class": C.CLASSES[seed_major[idx]] if seed_major[idx] >= 0 else ""})
    return rows


def prepare_active_sample(args):
    (batch, sid), used, n_per = args
    rng = np.random.default_rng(int(hashlib.md5((sid + "r2").encode()).hexdigest()[:8], 16))
    ch = C.load_channels(sid)
    meta = json.load(open(C.proc_dir(sid) / "meta.json"))
    sp = np.load(C.proc_dir(sid) / "superpixels.npy")
    lab = np.load(C.proc_dir(sid) / "student_labels.npy")
    maxp = np.load(C.proc_dir(sid) / "student_maxp.npy").astype(np.float32) / 255
    ids = np.arange(1, sp.max() + 1)
    area = ndi.sum(np.ones_like(sp), sp, ids)
    conf = ndi.mean(maxp, sp, ids)
    frac = np.stack([ndi.sum(lab == k, sp, ids) / np.maximum(area, 1) for k in range(4)], 1)
    major = frac.argmax(1)
    cy, cx = [np.round(v).astype(int) for v in zip(*ndi.center_of_mass(np.ones_like(sp), sp, ids))]
    h, w = sp.shape
    ok = (cy > CROP // 2) & (cy < h - CROP // 2) & (area > 100) & ~np.isin(ids, list(used))
    pc = (np.isin(major, [C.PORE, C.CBD]) | (frac[:, C.PORE] + frac[:, C.CBD] >= 0.3)) & ok
    other = ok & ~pc
    a = np.flatnonzero(pc)[np.argsort(conf[pc])][:n_per // 2]                      # least confident pore/CBD
    b = np.flatnonzero(other)[np.argsort(conf[other])][:n_per // 4]                # least confident elsewhere
    rest = np.setdiff1d(np.flatnonzero(pc & np.isin(major, [C.PORE, C.CBD])), a)
    n_c = max(0, n_per - len(a) - len(b))
    c = rng.choice(rest, min(len(rest), n_c), replace=False) if len(rest) else np.array([], int)
    rows = []
    for kind, group in (("active_uncertain_pore_cbd", a), ("active_uncertain_other", b), ("active_random_pore_cbd", c)):
        for idx in group:
            labid = int(ids[idx])
            name = f"{sid}_{labid}"
            render(ch, sp, labid, cy[idx], cx[idx], meta["se2_detector"]).save(CROPS / f"{name}.png")
            rows.append({"item": name, "batch": batch, "sample_id": sid, "superpixel": labid, "cy": int(cy[idx]),
                         "cx": int(cx[idx]), "area_px": int(area[idx]), "kind": kind, "seed_class": "",
                         "student_class": C.CLASSES[major[idx]], "student_conf": round(float(conf[idx]), 3)})
    return rows


def cmd_prepare_active(args):
    items, _, _ = round_paths(2)
    used = {}
    for r in C.read_csv(ITEMS_CSV):
        used.setdefault(r["sample_id"], set()).add(int(r["superpixel"]))
    jobs = [((b, s), used.get(s, set()), args.per_sample) for b, s in C.sample_ids()]
    with ProcessPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(prepare_active_sample, jobs) for r in rs]
    rng = np.random.default_rng(1)
    for n, i in enumerate(rng.permutation(len(rows))):
        rows[i]["num"] = n
        rows[i]["request"] = f"r2req{n // ITEMS_PER_REQUEST:04d}"
        rows[i]["split"] = "val" if rng.random() < VAL_FRACTION else "train"
    rows.sort(key=lambda r: r["num"])
    C.write_csv(items, rows)
    kinds = {k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})}
    print(f"{len(rows)} round-2 items {kinds}, {len({r['request'] for r in rows})} requests -> {items}")


def cmd_prepare(_):
    CROPS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(C.WORKERS) as ex:
        rows = [r for rs in ex.map(prepare_sample, C.sample_ids()) for r in rs]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))                     # mix samples within each request
    for n, i in enumerate(order):
        rows[i]["num"] = n
        rows[i]["request"] = f"req{n // ITEMS_PER_REQUEST:04d}"
        rows[i]["split"] = "val" if rng.random() < VAL_FRACTION else "train"
    rows.sort(key=lambda r: r["num"])
    C.write_csv(ITEMS_CSV, rows)
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("ambiguous", "seeded")}
    print(f"{len(rows)} items {kinds}, {len({r['request'] for r in rows})} requests -> {ITEMS_CSV}")


# ---------------------------------------------------------------- requests

def request_params(items):
    content = [{"type": "text", "text": f"{len(items)} items follow. Classify each outlined region."}]
    for r in items:
        b64 = base64.standard_b64encode((CROPS / f"{r['item']}.png").read_bytes()).decode()
        content += [{"type": "text", "text": f"Item id {r['num']}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]
    return {"model": MODEL, "max_tokens": 64000, "system": SYSTEM,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}}}


def client():
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    return anthropic.Anthropic(max_retries=4)


def grouped(rnd=1):
    rows = C.read_csv(round_paths(rnd)[0])
    reqs = {}
    for r in rows:
        r["num"] = int(r["num"])
        reqs.setdefault(r["request"], []).append(r)
    return reqs


def cmd_pilot(_):
    reqs = grouped()
    items = reqs[sorted(reqs)[0]]
    with client().messages.stream(**request_params(items)) as s:
        msg = s.get_final_message()
    out = json.loads(next(b.text for b in msg.content if b.type == "text"))["items"]
    by = {r["num"]: r for r in items}
    for o in out:
        r = by[o["id"]]
        print(f"{r['item']:22s} {r['kind']:9s} seed={r['seed_class'] or '-':8s} -> {o['label']:9s} ({o['confidence']}) {o['reason']}")
    cost = msg.usage.input_tokens * 4 / 1e6 + msg.usage.output_tokens * 20 / 1e6
    print(f"usage in={msg.usage.input_tokens} out={msg.usage.output_tokens} cost ${cost:.3f} (full price)")


def cmd_submit(args):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs = grouped(args.round)
    skip = set(args.skip or [])
    batch = client().messages.batches.create(requests=[
        Request(custom_id=rid, params=MessageCreateParamsNonStreaming(**request_params(items)))
        for rid, items in sorted(reqs.items()) if rid not in skip])
    round_paths(args.round)[1].write_text(batch.id)
    print(f"submitted batch {batch.id} with {len(reqs) - len(skip)} requests")


def cmd_collect(args):
    cl = client()
    _, batch_file, labels_csv = round_paths(args.round)
    bid = args.batch or batch_file.read_text().strip()
    while True:
        b = cl.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"{time.strftime('%H:%M')} batch {b.processing_status}: {b.request_counts}", flush=True)
        if not args.wait:
            return
        time.sleep(60)
    reqs = grouped(args.round)
    by_num = {r["num"]: r for rs in reqs.values() for r in rs}
    out_rows, tok_in, tok_out, failed = [], 0, 0, []
    for res in cl.messages.batches.results(bid):
        if res.result.type != "succeeded":
            failed.append(res.custom_id)
            continue
        msg = res.result.message
        tok_in += msg.usage.input_tokens
        tok_out += msg.usage.output_tokens
        if msg.stop_reason != "end_turn":
            failed.append(res.custom_id)
            continue
        for o in json.loads(next(b.text for b in msg.content if b.type == "text"))["items"]:
            if o["id"] in by_num and by_num[o["id"]]["request"] == res.custom_id:
                r = by_num[o["id"]]
                out_rows.append({**{k: r.get(k, "") for k in ("item", "batch", "sample_id", "superpixel", "cy", "cx",
                                                              "kind", "seed_class", "split", "request",
                                                              "student_class", "student_conf")},
                                 "ai_label": o["label"], "confidence": o["confidence"], "reason": o["reason"]})
    if labels_csv.exists() and args.append:
        old = C.read_csv(labels_csv)
        done = {r["item"] for r in out_rows}
        out_rows = [r for r in old if r["item"] not in done] + out_rows
    C.write_csv(labels_csv, out_rows)
    cost = (tok_in * 4 + tok_out * 20) / 1e6 / 2
    print(f"{len(out_rows)} labels, failed requests {failed}; tokens in={tok_in} out={tok_out}; batch cost ${cost:.2f}")
    seeded = [r for r in out_rows if r["kind"] == "seeded"]
    if seeded:
        agree = np.mean([r["ai_label"] == r["seed_class"] for r in seeded])
        print(f"annotator vs rule seeds on {len(seeded)} seeded items: {100 * agree:.0f}% agreement")
    for lab in LABELS:
        print(f"  {lab:9s} {sum(r['ai_label'] == lab for r in out_rows)}")


def cmd_apply(_):
    rows = []
    for rnd in (1, 2, 3):
        path = round_paths(rnd)[2]
        if path.exists():
            rows += C.read_csv(path)
    usable = [r for r in rows if r["ai_label"] in C.CLASSES and r["confidence"] in ("medium", "high")]
    per = {}
    for r in usable:
        per.setdefault(r["sample_id"], []).append(r)
    for _, sid in C.sample_ids():
        sp = np.load(C.proc_dir(sid) / "superpixels.npy")
        for split in ("train", "val"):
            lab = np.full(sp.shape, C.EXCLUDED, np.uint8)
            for r in per.get(sid, []):
                if r["split"] == split:
                    lab[sp == int(r["superpixel"])] = C.CLASSES.index(r["ai_label"])
            np.save(C.proc_dir(sid) / ("ai_labels.npy" if split == "train" else "ai_labels_val.npy"), lab)
    print(f"applied {len(usable)} usable labels of {len(rows)} "
          f"(train {sum(r['split'] == 'train' for r in usable)}, val {sum(r['split'] == 'val' for r in usable)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "prepare-active", "pilot", "submit", "collect", "apply"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--per-sample", type=int, default=32)
    ap.add_argument("--batch")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--skip", nargs="*")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "prepare-active": cmd_prepare_active, "pilot": cmd_pilot, "submit": cmd_submit,
     "collect": cmd_collect, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
