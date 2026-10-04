"""Independent accuracy check: a vision model (Claude Opus 5.5) classifies single marked points.

Two point sets, rendered as BSE | Inlens | ETD/SE crops (6.4 um) with a small ring around the point:
  uniform     240 points drawn uniformly over all valid pixels of all samples (excluding superpixels used
              for training). Gives an unbiased overall accuracy and an independent estimate of phase
              fractions, without looking at any model output.
  stratified  30 points per class of the delivered label maps -> per-class precision.

Subcommands: prepare-uniform | prepare-stratified | submit SET | collect SET [--wait]
Results: outputs/metrics/ai_points_<set>.csv

Usage: python -m src.ai_validate prepare-uniform
"""
import argparse
import base64
import json
import time

import numpy as np
from PIL import Image, ImageDraw

from . import config as C
from .ai_labels import EFFORT, MODEL, client

PT_DIR = C.PROC / "ai_points"
CROP, SCALE = 128, 3
ITEMS_PER_REQUEST = 10
LABELS = ["pore", "graphite", "SiOx", "CBD", "uncertain"]

SYSTEM = """You are an expert in SEM microstructure analysis of lithium-ion battery electrodes.

The images are polished cross-sections of a calendered graphite + SiOx blended anode, imaged with three \
co-registered detectors at 50 nm/px. Each item shows the same 6.4 x 6.4 um area three times, left to right: \
BSE (atomic-number contrast; SiOx is brighter than carbon), Inlens (fine surface detail and edges; strong \
charging/channelling contrast, so its brightness varies between particles and samples), and ETD or SE \
(topography and shading). A small yellow ring marks ONE point, at the same position in all three panels (usually the centre; off-centre near image edges).

Classify the material at the centre of the ring:
- pore: empty space. Includes deep pores (black in all detectors) AND open, unfilled pores whose floor shows \
material BELOW the polished plane: 3D-shaded surfaces, shadows in ETD/SE, no flat polished face, often ringed \
by bright Inlens edges. The pores were NOT resin-filled, so a pore floor can look grey in BSE.
- graphite: a polished face of a graphite particle: flat, uniform BSE grey, lamellar texture, cracks, curtaining.
- SiOx: polished face of a brighter (in BSE), angular, non-lamellar particle, including mottled or porous ones.
- CBD: carbon-binder domain in the polished plane: lacy, spongy, nanoporous material at particle necks, in \
pockets, or as films on particle surfaces; same BSE grey as graphite but finely textured.
- uncertain: you cannot tell (e.g. the point sits exactly on a boundary).

Judge each item independently and use all three detectors."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "label", "confidence"],
        "properties": {"id": {"type": "integer"},
                       "label": {"type": "string", "enum": LABELS},
                       "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}}},
}


def render(ch, y, x, se2_name):
    h, w = ch["BSE"].shape
    y0, x0 = int(np.clip(y - CROP // 2, 0, h - CROP)), int(np.clip(x - CROP // 2, 0, w - CROP))
    cy, cx = (y - y0) * SCALE + SCALE // 2, (x - x0) * SCALE + SCALE // 2
    size = CROP * SCALE
    panels = []
    for det in ("BSE", "Inlens", "SE2"):
        g = (ch[det][y0:y0 + CROP, x0:x0 + CROP] * 255).astype(np.uint8)
        im = Image.fromarray(g).convert("RGB").resize((size, size), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 230, 0), width=2)
        d.text((6, 4), det if det != "SE2" else se2_name, fill=(255, 60, 60))
        panels.append(np.asarray(im))
    gap = np.full((size, 8, 3), 255, np.uint8)
    return Image.fromarray(np.hstack([panels[0], gap, panels[1], gap, panels[2]]))


def write_points(name, points):
    (PT_DIR / name).mkdir(parents=True, exist_ok=True)
    cache = {}
    rows = []
    for n, (sid, y, x, extra) in enumerate(points):
        if sid not in cache:
            cache = {sid: (C.load_channels(sid), json.load(open(C.proc_dir(sid) / "meta.json")))}
        ch, meta = cache[sid]
        render(ch, y, x, meta["se2_detector"]).save(PT_DIR / name / f"p{n:04d}.png")
        rows.append({"num": n, "sample_id": sid, "y": y, "x": x, "request": f"{name}{n // ITEMS_PER_REQUEST:03d}", **extra})
    C.write_csv(PT_DIR / f"{name}.csv", rows)
    print(f"{len(rows)} points -> {PT_DIR / name}")


def cmd_prepare_uniform(args):
    rng = np.random.default_rng(7)
    sids = [s for _, s in C.sample_ids()]
    valid = {}
    for sid in sids:
        ex = C.load_exclude(sid)
        sp = np.load(C.proc_dir(sid) / "superpixels.npy")
        used = np.isin(sp, [int(r["superpixel"]) for r in C.read_csv(C.PROC / "ai_labels" / "items.csv") if r["sample_id"] == sid])
        valid[sid] = ~ex & ~used
    counts = np.array([v.sum() for v in valid.values()], float)
    picks = rng.choice(len(sids), args.n, p=counts / counts.sum())
    points = []
    for i in sorted(picks):
        sid = sids[i]
        flat = np.flatnonzero(valid[sid])
        y, x = np.unravel_index(rng.choice(flat), valid[sid].shape)
        points.append((sid, int(y), int(x), {}))
    write_points("uniform", points)


def cmd_prepare_stratified(args):
    rng = np.random.default_rng(11)
    sids = [s for _, s in C.sample_ids()]
    maps = {sid: np.load(C.proc_dir(sid) / f"{args.labels}_labels.npy") for sid in sids}
    points = []
    for k, cls in enumerate(C.CLASSES):
        pool = [(sid, i) for sid in sids for i in rng.choice(np.flatnonzero(maps[sid].ravel() == k), 20)]
        for j in rng.choice(len(pool), args.per_class, replace=False):
            sid, i = pool[j]
            y, x = np.unravel_index(i, maps[sid].shape)
            points.append((sid, int(y), int(x), {"model_class": cls}))
    order = rng.permutation(len(points))
    write_points("stratified", [points[i] for i in order])


def params(name, items):
    content = [{"type": "text", "text": f"{len(items)} items follow. Classify the material at each ring centre."}]
    for r in items:
        b64 = base64.standard_b64encode((PT_DIR / name / f"p{int(r['num']):04d}.png").read_bytes()).decode()
        content += [{"type": "text", "text": f"Item id {r['num']}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]
    return {"model": MODEL, "max_tokens": 64000, "system": SYSTEM, "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}}}


def cmd_submit(args):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    rows = C.read_csv(PT_DIR / f"{args.set}.csv")
    reqs = {}
    for r in rows:
        reqs.setdefault(r["request"], []).append(r)
    b = client().messages.batches.create(requests=[
        Request(custom_id=rid, params=MessageCreateParamsNonStreaming(**params(args.set, items)))
        for rid, items in sorted(reqs.items())])
    (PT_DIR / f"{args.set}_batch_id.txt").write_text(b.id)
    print(f"submitted {args.set} batch {b.id} ({len(reqs)} requests)")


def cmd_collect(args):
    cl = client()
    bid = (PT_DIR / f"{args.set}_batch_id.txt").read_text().strip()
    while True:
        b = cl.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"{time.strftime('%H:%M')} {args.set} batch {b.processing_status}: {b.request_counts}", flush=True)
        if not args.wait:
            return
        time.sleep(60)
    rows = {int(r["num"]): r for r in C.read_csv(PT_DIR / f"{args.set}.csv")}
    out, tin, tout, failed = [], 0, 0, []
    for res in cl.messages.batches.results(bid):
        if res.result.type != "succeeded" or res.result.message.stop_reason != "end_turn":
            failed.append(res.custom_id)
            continue
        msg = res.result.message
        tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
        for o in json.loads(next(x.text for x in msg.content if x.type == "text"))["items"]:
            if o["id"] in rows and rows[o["id"]]["request"] == res.custom_id:
                out.append({**rows[o["id"]], "ai_label": o["label"], "confidence": o["confidence"]})
    out.sort(key=lambda r: int(r["num"]))
    C.write_csv(C.OUT_MET / f"ai_points_{args.set}.csv", out)
    print(f"{len(out)} points labelled, failed {failed}; batch cost ${(tin * 4 + tout * 20) / 1e6 / 2:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare-uniform", "prepare-stratified", "submit", "collect"])
    ap.add_argument("set", nargs="?", default="uniform")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--labels", default="final")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    {"prepare-uniform": cmd_prepare_uniform, "prepare-stratified": cmd_prepare_stratified,
     "submit": cmd_submit, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
