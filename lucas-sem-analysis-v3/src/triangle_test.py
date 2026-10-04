"""Blinded triangle (odd-one-out) test: can a vision model tell batches apart at all?

Each trial shows three SEM cross-sections (BSE above Inlens, per-image normalised, so absolute brightness is
removed), two from one batch and one from another, in random order and without any batch information. The model
must pick the odd one out. If two batches are indistinguishable, the success rate is 1/3 by construction, so no
separate control is needed. 30 trials per batch pair (B1-B2, B1-B3, B2-B3), balanced so each batch is the
"double" in half of its trials.

Subcommands: prepare | submit | collect [--wait]
Results: outputs/metrics/triangle_test.csv, triangle_test.json

Usage: python -m src.triangle_test prepare
"""
import argparse
import base64
import io
import json
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import stats

from . import config as C
from .ai_labels import EFFORT, MODEL, client

DIR = C.PROC / "triangle"
PAIRS = [("Batch_1", "Batch_2"), ("Batch_1", "Batch_3"), ("Batch_2", "Batch_3")]
N_PER_PAIR = 30
WIDTH = 1400

SYSTEM = """You are an expert in SEM microstructure analysis of lithium-ion battery electrodes.

You will see three polished cross-sections (A, B, C) of graphite + SiOx blended anodes. Each panel shows the BSE
image (top; SiOx particles are brighter) and the Inlens image (bottom) of the same ~175 x 50 um field. Each image
was contrast-normalised separately, so absolute brightness carries no information.

Exactly two of the three samples come from the same production batch; the third comes from a different batch.
Identify the odd one out by comparing microstructure: particle sizes and shapes, SiOx amount and dispersion,
porosity and pore shapes, binder distribution, packing and flake alignment. If you see no real difference, still
choose your best guess, and say so in your confidence."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["odd_one_out", "confidence", "main_difference"],
          "properties": {"odd_one_out": {"type": "string", "enum": ["A", "B", "C"]},
                         "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                         "main_difference": {"type": "string", "description": "at most 25 words"}}}


def panel(sid):
    ch = C.load_channels(sid, ["BSE", "Inlens"])
    ims = []
    for d in ("BSE", "Inlens"):
        im = Image.fromarray((ch[d] * 255).astype(np.uint8))
        ims.append(im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS))
    out = Image.new("L", (WIDTH, ims[0].height + ims[1].height + 6), 255)
    out.paste(ims[0], (0, 0)); out.paste(ims[1], (0, ims[0].height + 6))
    return out


def b64png(im):
    """JPEG q90 (keeps 90 trials under the Batch API size limit; artefacts are negligible at this quality)."""
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def cmd_prepare(_):
    DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    by = {}
    for b, s in C.sample_ids():
        by.setdefault(b, []).append(s)
    rows = []
    for x, y in PAIRS:
        for t in range(N_PER_PAIR):
            dbl, odd = (x, y) if t % 2 == 0 else (y, x)
            two = list(rng.choice(by[dbl], 2, replace=False))
            one = str(rng.choice(by[odd]))
            order = rng.permutation(3)
            members = [two[0], two[1], one]
            slots = {"ABC"[order[i]]: members[i] for i in range(3)}
            answer = "ABC"[order[2]]
            rows.append({"trial": f"{x[-1]}{y[-1]}_{t:02d}", "pair": f"{x}-{y}", "double_batch": dbl, "odd_batch": odd,
                         "A": slots["A"], "B": slots["B"], "C": slots["C"], "answer": answer})
    C.write_csv(DIR / "trials.csv", rows)
    for s in {r[k] for r in rows for k in "ABC"}:
        panel(s).save(DIR / f"{s}.png")
    print(f"{len(rows)} trials, {len({r[k] for r in rows for k in 'ABC'})} sample panels -> {DIR}")


def params(r):
    content = [{"type": "text", "text": "Three samples follow. Which one is from a different batch?"}]
    for k in "ABC":
        content += [{"type": "text", "text": f"Sample {k}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": b64png(Image.open(DIR / f"{r[k]}.png"))}}]
    return {"model": MODEL, "max_tokens": 64000, "system": SYSTEM, "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}}}


def cmd_submit(_):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    rows = C.read_csv(DIR / "trials.csv")
    b = client().messages.batches.create(requests=[
        Request(custom_id=r["trial"], params=MessageCreateParamsNonStreaming(**params(r))) for r in rows])
    (DIR / "batch_id.txt").write_text(b.id)
    print(f"submitted triangle batch {b.id} ({len(rows)} trials)")


def binom_summary(k, n):
    p = stats.binomtest(k, n, 1 / 3, alternative="greater").pvalue
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
    return {"correct": f"{k}/{n}", "rate": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "p_vs_chance_1_3": round(float(p), 4)}


def cmd_collect(args):
    cl = client()
    bid = (DIR / "batch_id.txt").read_text().strip()
    while True:
        b = cl.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"{time.strftime('%H:%M')} triangle batch {b.processing_status}: {b.request_counts}", flush=True)
        if not args.wait:
            return
        time.sleep(60)
    trials = {r["trial"]: r for r in C.read_csv(DIR / "trials.csv")}
    out, tin, tout = [], 0, 0
    for res in cl.messages.batches.results(bid):
        if res.result.type != "succeeded" or res.result.message.stop_reason != "end_turn":
            continue
        m = res.result.message
        tin += m.usage.input_tokens; tout += m.usage.output_tokens
        o = json.loads(next(x.text for x in m.content if x.type == "text"))
        r = trials[res.custom_id]
        out.append({**r, "chosen": o["odd_one_out"], "correct": o["odd_one_out"] == r["answer"],
                    "confidence": o["confidence"], "main_difference": o["main_difference"]})
    out.sort(key=lambda r: r["trial"])
    C.write_csv(C.OUT_MET / "triangle_test.csv", out)
    summary = {"cost_usd_batch": round((tin * 4 + tout * 20) / 1e6 / 2, 2)}
    for pair in sorted({r["pair"] for r in out}):
        sub = [r for r in out if r["pair"] == pair]
        summary[pair] = binom_summary(sum(r["correct"] for r in sub), len(sub))
        conf = [r for r in sub if r["confidence"] in ("medium", "high")]
        if conf:
            summary[pair]["medium_high_confidence_only"] = binom_summary(sum(r["correct"] for r in conf), len(conf))
    C.write_json(C.OUT_MET / "triangle_test.json", summary)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "submit", "collect"])
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
