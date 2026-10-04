"""Analyse the SEM cross-section dataset with Claude Opus 5.5 (max effort).

For each sample (one image per detector: BSE, Inlens, ETD or SE):
  1. compute rough phase fractions locally from the BSE image (multi-Otsu threshold),
  2. send overview + native-resolution crops to Claude for a structured analysis,
  3. cache the result in outputs/samples/ so re-runs skip finished samples.
Then one text-only call synthesises a dataset-level report (outputs/report.md).

Usage:
  python analyze_sem.py --metrics-only          # local metrics + previews, no API calls
  python analyze_sem.py --only 4ih2ggld         # one sample through the API
  python analyze_sem.py                         # every sample, then the dataset report
"""
import argparse
import base64
import csv
import glob
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import httpx2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.morphology import disk, remove_small_objects

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "prior_analysis", "llm_run")
SAMPLES_DIR = os.path.join(OUT, "samples")
PREVIEW_DIR = os.path.join(OUT, "previews")

MODEL = "claude-opus-5-5"
EFFORT = "max"
NM_PER_PX = 25.0
OVERVIEW_WIDTH = 2576          # model's max long edge
CROP_W, CROP_H = 2560, 1440    # native 25 nm/px crop, just under the 3.75 MP cap
PRICE_IN, PRICE_OUT = 4.00, 20.00  # USD per million tokens, Opus 5.5
DETECTOR_ORDER = ["BSE", "Inlens", "ETD", "SE"]


# ---------------------------------------------------------------- dataset

def discover_samples():
    samples = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(RAW, "Batch_*", "img_*_*.tif"))):
        batch = os.path.basename(os.path.dirname(path))
        m = re.match(r"img_([a-z0-9]+)_([A-Za-z]+)\.tif$", os.path.basename(path))
        if m:
            samples[(batch, m.group(1))][m.group(2)] = path
    return samples


def load_grey(path):
    return np.asarray(Image.open(path).convert("L"))


# ---------------------------------------------------------------- local metrics

def bse_metrics(bse):
    """Rough 3-class segmentation of the BSE image: pore / matrix / bright phase."""
    smooth = ndi.median_filter(bse, size=5)
    t_low, t_high = threshold_multiotsu(smooth, classes=3)
    labels = np.digitize(smooth, [t_low, t_high])

    um2_per_px = (NM_PER_PX / 1000) ** 2
    pores = remove_small_objects(labels == 0, max_size=80)            # < 0.05 um^2 = noise
    bright_raw = labels == 2
    # Thin bright features (carbon/binder web, particle rims) are removed by an opening;
    # what survives and is > 2 um^2 is counted as a bright-phase particle.
    bright_particles = remove_small_objects(ndi.binary_opening(bright_raw, structure=disk(6)), max_size=3200)

    lab, n = ndi.label(bright_particles)
    areas_px = np.bincount(lab.ravel())[1:]
    eq_diam_um = 2 * np.sqrt(areas_px * um2_per_px / np.pi) if n else np.array([])

    h, w = bse.shape
    metrics = {
        "field_of_view_um": [round(w * NM_PER_PX / 1000, 1), round(h * NM_PER_PX / 1000, 1)],
        "bse_thresholds": [int(t_low), int(t_high)],
        "pore_area_pct": round(100 * pores.mean(), 1),
        "bright_raw_area_pct": round(100 * bright_raw.mean(), 1),
        "bright_particle_area_pct": round(100 * bright_particles.mean(), 1),
        "bright_particle_count": int(n),
        "bright_particle_eq_diam_um_median": round(float(np.median(eq_diam_um)), 2) if n else None,
        "bright_particle_eq_diam_um_p90": round(float(np.percentile(eq_diam_um, 90)), 2) if n else None,
        "bright_particles_per_1000um2": round(n / (h * w * um2_per_px) * 1000, 2),
    }
    return metrics, pores, bright_particles


def save_overlay(bse, pores, bright_particles, path):
    rgb = np.stack([bse] * 3, -1).astype(np.float32)
    rgb[pores] = rgb[pores] * 0.3 + np.array([0, 90, 255]) * 0.7
    rgb[bright_particles] = rgb[bright_particles] * 0.3 + np.array([255, 140, 0]) * 0.7
    im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    im.thumbnail((1800, 1800))
    im.save(path, quality=90)


# ---------------------------------------------------------------- image prep

def to_jpeg_b64(arr, max_width=None):
    im = Image.fromarray(arr)
    if max_width and im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode(), im.size


def centre_crop(arr):
    h, w = arr.shape
    ch = min(CROP_H, h)
    x0, y0 = (w - CROP_W) // 2, (h - ch) // 2
    return arr[y0:y0 + ch, x0:x0 + CROP_W], (x0, y0, CROP_W, ch)


def image_block(b64):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


# ---------------------------------------------------------------- prompts / schema

SYSTEM = """You are an expert in scanning electron microscopy and the microstructural characterisation of \
battery electrodes (active-material particles, carbon-binder domain, porosity, calendering effects, \
cross-section preparation artefacts).

You are analysing one sample from a dataset of polished SEM cross-sections. Each sample was imaged at the \
same location with several detectors, which are co-registered:
- BSE (backscattered electrons): contrast dominated by mean atomic number; higher-Z phases appear brighter.
- Inlens: high-resolution secondary-electron/topography signal, very sensitive to edges and fine features.
- ETD or SE (Everhart-Thornley / secondary electrons): topographic contrast.

Pixel size is 25 nm. Base your conclusions on what is visible; separate observation from interpretation, \
say how confident you are, and say so plainly when something cannot be determined from these images. \
Quantities you estimate visually are approximate - give your best number anyway, rather than refusing.

You are also given rough measurements computed automatically from a 3-class threshold of the BSE image. \
They can be wrong (e.g. resin or shadowing counted as pore, rims or binder counted as bright phase): check \
them against the images and say whether you trust them."""

nullable_number = {"anyOf": [{"type": "number"}, {"type": "null"}]}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "material_system", "phases", "porosity", "binder_carbon_network",
        "particle_packing_and_alignment", "heterogeneity", "defects_and_damage",
        "imaging_artifacts", "detector_insights", "threshold_metrics_assessment",
        "notable_features", "summary",
    ],
    "properties": {
        "material_system": {"type": "string", "description": "Best identification of what this sample is (electrode type, likely active materials) and the key evidence."},
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "identification_evidence", "appearance_by_detector",
                             "estimated_area_percent", "typical_size_um", "morphology", "confidence"],
                "properties": {
                    "name": {"type": "string"},
                    "identification_evidence": {"type": "string"},
                    "appearance_by_detector": {"type": "string"},
                    "estimated_area_percent": {**nullable_number, "description": "Visual estimate of area fraction in the cross-section, 0-100."},
                    "typical_size_um": {**nullable_number, "description": "Typical particle or feature size in micrometres."},
                    "morphology": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        },
        "porosity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["estimated_percent", "character", "threshold_estimate_agreement", "comment"],
            "properties": {
                "estimated_percent": nullable_number,
                "character": {"type": "string", "description": "Pore size, shape, distribution and apparent connectivity."},
                "threshold_estimate_agreement": {"type": "string", "enum": ["agrees", "roughly", "disagrees", "cannot_tell"]},
                "comment": {"type": "string"},
            },
        },
        "binder_carbon_network": {"type": "string", "description": "Distribution and coverage of the carbon-binder domain."},
        "particle_packing_and_alignment": {"type": "string", "description": "Packing density, contacts, orientation/alignment of particles, which image direction is likely through-thickness."},
        "heterogeneity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["score_1_to_5", "description"],
            "properties": {
                "score_1_to_5": {"type": "integer", "description": "1 = very uniform across the field of view, 5 = very heterogeneous."},
                "description": {"type": "string"},
            },
        },
        "defects_and_damage": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "description", "location", "severity"],
                "properties": {
                    "type": {"type": "string", "description": "e.g. particle cracking, delamination, agglomerate, large void, debonding."},
                    "description": {"type": "string"},
                    "location": {"type": "string", "description": "Where in the field of view, e.g. 'left third, upper half'."},
                    "severity": {"type": "string", "enum": ["minor", "moderate", "major"]},
                },
            },
        },
        "imaging_artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "description", "impact_on_analysis"],
                "properties": {
                    "type": {"type": "string", "description": "e.g. charging, curtaining, smearing, polishing scratches, drift, stitching seam, edge effects."},
                    "description": {"type": "string"},
                    "impact_on_analysis": {"type": "string", "enum": ["negligible", "minor", "significant"]},
                },
            },
        },
        "detector_insights": {
            "type": "object",
            "additionalProperties": False,
            "required": ["BSE", "Inlens", "ETD_or_SE"],
            "properties": {
                "BSE": {"type": "string"},
                "Inlens": {"type": "string"},
                "ETD_or_SE": {"type": "string"},
            },
        },
        "threshold_metrics_assessment": {"type": "string", "description": "Which automatic measurements look trustworthy and which are biased, and in which direction."},
        "notable_features": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "description": "3-5 sentence summary of this sample."},
    },
}


# ---------------------------------------------------------------- API calls

def make_client():
    load_dotenv(os.path.join(ROOT, ".env"))
    return anthropic.Anthropic(max_retries=4)


def call_claude(client, system, content, output_config, max_tokens=128000, attempts=3, extra_betas=()):
    """Streamed request (max effort can think for a long time) with server-side refusal fallback.

    The SDK retries failed connections, but not a stream that drops mid-response, so retry those here.
    """
    for attempt in range(1, attempts + 1):
        try:
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config=output_config,
                betas=["server-side-fallback-2026-07-01", *extra_betas],
                fallbacks="default",
            ) as stream:
                msg = stream.get_final_message()
            break
        except (anthropic.APIConnectionError, httpx2.TransportError) as e:
            if attempt == attempts:
                raise
            print(f"  connection dropped ({e!r}), retrying {attempt}/{attempts - 1}", flush=True)
            time.sleep(30 * attempt)
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"request declined: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens before finishing")
    text = "".join(b.text for b in msg.content if b.type == "text")
    usage = {
        "model": msg.model,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
    usage["cost_usd"] = round(usage["input_tokens"] * PRICE_IN / 1e6 + usage["output_tokens"] * PRICE_OUT / 1e6, 4)
    return text, usage


def prepare_sample(batch, sid, paths):
    """Local metrics + preview files; returns (metrics, images for the prompt)."""
    imgs = {d: load_grey(paths[d]) for d in DETECTOR_ORDER if d in paths}
    metrics, pores, bright = bse_metrics(imgs["BSE"])
    save_overlay(imgs["BSE"], pores, bright, os.path.join(PREVIEW_DIR, f"{batch}_{sid}_BSE_segmentation.jpg"))

    prompt_images = []
    for det, arr in imgs.items():
        b64, size = to_jpeg_b64(arr, OVERVIEW_WIDTH)
        with open(os.path.join(PREVIEW_DIR, f"{batch}_{sid}_{det}.jpg"), "wb") as f:
            f.write(base64.b64decode(b64))
        prompt_images.append((f"{det} - full field of view, downsampled to {size[0]}x{size[1]} px "
                              f"({NM_PER_PX * arr.shape[1] / size[0]:.0f} nm/px).", b64))
    for det in ("BSE", "Inlens"):
        if det in imgs:
            crop, (x0, y0, w, h) = centre_crop(imgs[det])
            b64, _ = to_jpeg_b64(crop)
            prompt_images.append((f"{det} - native-resolution crop (25 nm/px) from the centre: "
                                  f"x {x0 * NM_PER_PX / 1000:.1f}-{(x0 + w) * NM_PER_PX / 1000:.1f} um, "
                                  f"y {y0 * NM_PER_PX / 1000:.1f}-{(y0 + h) * NM_PER_PX / 1000:.1f} um.", b64))
    return metrics, prompt_images


def analyse_sample(client, batch, sid, paths, metrics_only=False):
    out_path = os.path.join(SAMPLES_DIR, f"{batch}_{sid}.json")
    if not metrics_only and os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f), True

    metrics, prompt_images = prepare_sample(batch, sid, paths)
    record = {"batch": batch, "sample_id": sid, "detectors": sorted(paths), "metrics": metrics}
    if metrics_only:
        return record, False

    content = [{"type": "text", "text": (
        f"Sample {sid} ({batch}). Detectors: {', '.join(sorted(paths))}. "
        f"Field of view {metrics['field_of_view_um'][0]} x {metrics['field_of_view_um'][1]} um.\n\n"
        f"Automatic BSE threshold measurements (rough):\n{json.dumps(metrics, indent=2)}\n"
        "pore = darkest class; bright_raw = brightest class; bright_particle = brightest class after removing "
        "thin features (<0.3 um wide) and objects < 2 um^2.")}]
    for label, b64 in prompt_images:
        content += [{"type": "text", "text": label}, image_block(b64)]
    content.append({"type": "text", "text": "Analyse this sample's microstructure."})

    text, usage = call_claude(client, SYSTEM, content,
                              {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}})
    record["analysis"] = json.loads(text)
    record["usage"] = usage
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return record, False


# ---------------------------------------------------------------- dataset report

REPORT_SYSTEM = """You are an expert in battery-electrode microstructure. You are given per-sample SEM \
cross-section analyses (written by a vision model, so they can contain mistakes) together with automatic \
threshold measurements, for a dataset split into batches. Write a dataset-level report in Markdown for a \
hackathon team. Be concrete and quantitative where the data allows, cite sample IDs, flag where the \
per-sample analyses disagree with each other or with the measurements, and do not invent information \
that is not in the data."""

REPORT_TASK = """Write the report with these sections:
1. Dataset overview - what the samples are, imaging conditions, what was analysed.
2. Material system and phases - consensus identification and how confident it is.
3. Batch comparison - a table of per-batch means/ranges for the key measurements, and the meaningful differences between batches.
4. Sample ranking and outliers - which samples stand out and why (porosity, bright-phase content, damage, heterogeneity).
5. Defects and imaging artefacts - recurring issues and which samples they affect.
6. Reliability of the automatic measurements - where they can and cannot be trusted.
7. Recommended next steps - concrete analyses or experiments for the team."""


SI_PHASE = re.compile(r"silicon|\bsio|\bsi\b", re.IGNORECASE)


def key_numbers(r):
    """The headline numbers for one sample, from the local metrics and Claude's analysis."""
    m, a = r["metrics"], r["analysis"]
    si = next((p for p in a["phases"] if SI_PHASE.search(p["name"])), None)
    return {
        "pore_thr": m["pore_area_pct"],
        "pore_claude": a["porosity"]["estimated_percent"],
        "si_thr": m["bright_particle_area_pct"],
        "si_claude": si["estimated_area_percent"] if si else None,
        "si_size_claude": si["typical_size_um"] if si else None,
        "heterogeneity": a["heterogeneity"]["score_1_to_5"],
        "defects": len(a["defects_and_damage"]),
        "moderate_or_major_defects": sum(d["severity"] != "minor" for d in a["defects_and_damage"]),
        "significant_artifacts": sum(x["impact_on_analysis"] == "significant" for x in a["imaging_artifacts"]),
    }


def stats_tables(records):
    """Per-sample and per-batch tables, computed here so the model does not do the arithmetic."""
    cols = ["pore_thr", "pore_claude", "si_thr", "si_claude", "si_size_claude", "heterogeneity",
            "defects", "moderate_or_major_defects", "significant_artifacts"]
    lines = ["Per-sample key numbers (pore/si in area %, size in um):",
             "| batch | sample | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    by_batch = defaultdict(list)
    for r in records:
        k = key_numbers(r)
        by_batch[r["batch"]].append(k)
        lines.append(f"| {r['batch']} | {r['sample_id']} | " + " | ".join(str(k[c]) for c in cols) + " |")
    lines += ["", "Per-batch mean (min-max):", "| batch | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for batch, ks in sorted(by_batch.items()):
        cells = []
        for c in cols:
            v = [k[c] for k in ks if k[c] is not None]
            cells.append(f"{np.mean(v):.1f} ({min(v):g}-{max(v):g})" if v else "-")
        lines.append(f"| {batch} | {len(ks)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_dataset_report(client, records):
    compact = [{"batch": r["batch"], "sample_id": r["sample_id"], "detectors": r["detectors"],
                "metrics": r["metrics"],
                "analysis": {k: v for k, v in r["analysis"].items() if k != "detector_insights"}}
               for r in records]
    content = [{"type": "text", "text": (
        f"{len(compact)} samples.\n\n{stats_tables(records)}\n\n"
        "Use the numbers in these tables as given; do not recompute them.\n\n"
        f"Per-sample analyses:\n{json.dumps(compact, separators=(',', ':'))}\n\n{REPORT_TASK}")}]
    # The task budget tells the model how much it can spend, so a max-effort synthesis
    # plans its thinking instead of running into max_tokens.
    text, usage = call_claude(client, REPORT_SYSTEM, content,
                              {"effort": EFFORT, "task_budget": {"type": "tokens", "total": 64000}},
                              extra_betas=["task-budgets-2026-03-13"])
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return usage


def write_csv(records):
    cols = ["batch", "sample_id", "detectors", "pore_area_pct", "bright_raw_area_pct", "bright_particle_area_pct",
            "bright_particle_count", "bright_particle_eq_diam_um_median", "bright_particle_eq_diam_um_p90",
            "claude_porosity_pct", "claude_porosity_vs_threshold", "heterogeneity_1_to_5",
            "n_defects", "n_major_defects", "n_significant_artifacts", "summary"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            m, a = r["metrics"], r.get("analysis", {})
            defects = a.get("defects_and_damage", [])
            row = {"batch": r["batch"], "sample_id": r["sample_id"], "detectors": "+".join(r["detectors"])}
            row.update({k: m.get(k) for k in cols if k in m})
            if a:
                row.update({
                    "claude_porosity_pct": a["porosity"]["estimated_percent"],
                    "claude_porosity_vs_threshold": a["porosity"]["threshold_estimate_agreement"],
                    "heterogeneity_1_to_5": a["heterogeneity"]["score_1_to_5"],
                    "n_defects": len(defects),
                    "n_major_defects": sum(d["severity"] == "major" for d in defects),
                    "n_significant_artifacts": sum(x["impact_on_analysis"] == "significant" for x in a["imaging_artifacts"]),
                    "summary": a["summary"],
                })
            w.writerow(row)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-only", action="store_true", help="local metrics and previews only, no API calls")
    ap.add_argument("--only", nargs="*", help="sample IDs to process")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-report", action="store_true", help="skip the dataset-level report")
    args = ap.parse_args()

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    samples = discover_samples()
    if args.only:
        samples = {k: v for k, v in samples.items() if k[1] in args.only}
    client = None if args.metrics_only else make_client()

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyse_sample, client, b, s, p, args.metrics_only): (b, s)
                   for (b, s), p in sorted(samples.items())}
        for fut in as_completed(futures):
            b, s = futures[fut]
            try:
                rec, cached = fut.result()
            except Exception as e:  # keep going; the failed sample is retried on the next run
                failures.append((b, s, repr(e)))
                print(f"FAIL {b}/{s}: {e!r}", flush=True)
                continue
            records.append(rec)
            u = rec.get("usage")
            tag = "cached" if cached else (f"${u['cost_usd']:.3f} in={u['input_tokens']} out={u['output_tokens']}" if u else "metrics")
            m = rec["metrics"]
            print(f"done {b}/{s}: pores {m['pore_area_pct']}%, bright particles {m['bright_particle_area_pct']}% [{tag}]", flush=True)

    records.sort(key=lambda r: (r["batch"], r["sample_id"]))
    write_csv(records)
    total = sum(r.get("usage", {}).get("cost_usd", 0) for r in records)
    print(f"\n{len(records)} samples ok, {len(failures)} failed; per-sample API cost so far ${total:.2f}")

    if not args.metrics_only and not args.no_report and not failures and not args.only:
        usage = write_dataset_report(client, records)
        print(f"dataset report written (${usage['cost_usd']:.3f}); total ${total + usage['cost_usd']:.2f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
