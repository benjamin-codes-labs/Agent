"""Stage 6c: report figures.

  reports/figures/contact_sheet.jpg     all 31 overlays with their phase fractions
  reports/figures/hard_<sid>.jpg        BSE | Inlens | overlay panels for the hard cases
  reports/figures/ai_label_examples.jpg AI-annotated superpixels per class (what the annotator saw)

Usage: python -m src.figures
"""
import numpy as np
from PIL import Image, ImageDraw

from . import config as C

FIG = C.REPORTS / "figures"
HARD = {   # (y0, y1, x0, x1) at half resolution
    "kbdh4tri": (0, 420, 1500, 2300),
    "epqdaau9": (674, 1074, 0, 800),
    "r17byphk": (300, 700, 1200, 2000),
    "5n1q8atc": (0, 400, 0, 800),
    "i9jiqjwl": (0, 400, 2600, 3400),
}


def text(im, xy, s, fill=(255, 255, 255)):
    d = ImageDraw.Draw(im)
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        d.text((x + dx, y + dy), s, fill=(0, 0, 0))
    d.text(xy, s, fill=fill)


def contact_sheet():
    fr = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "phase_fractions.csv")}
    tiles = []
    for batch, sid in C.sample_ids():
        im = Image.open(C.OUT_SEG / batch / sid / "overlay.jpg")
        im = im.resize((1100, round(im.height * 1100 / im.width)))
        f = fr[sid]
        text(im, (8, 6), f"{batch} {sid}   pore {float(f['pore_pct']):.1f}%  graphite {float(f['graphite_pct']):.1f}%  "
                         f"SiOx {float(f['SiOx_pct']):.1f}%  CBD {float(f['CBD_pct']):.1f}%")
        tiles.append(np.asarray(im.convert("RGB")))
    h = 340
    tiles = [np.asarray(Image.fromarray(t).resize((1100, h))) for t in tiles]
    rows = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2] + ([np.full_like(tiles[0], 255)] if i + 1 >= len(tiles) else [])
        rows.append(np.hstack([pair[0], np.full((h, 10, 3), 255, np.uint8), pair[1]]))
        rows.append(np.full((10, rows[-1].shape[1], 3), 255, np.uint8))
    Image.fromarray(np.vstack(rows)).save(FIG / "contact_sheet.jpg", quality=85)


def hard_cases():
    for batch, sid in C.sample_ids():
        if sid not in HARD:
            continue
        y0, y1, x0, x1 = HARD[sid]
        ch = C.load_channels(sid, ["BSE", "Inlens"])
        lab = np.load(C.proc_dir(sid) / "final_labels.npy")
        panels = [np.repeat((ch[d][y0:y1, x0:x1] * 255).astype(np.uint8)[..., None], 3, -1) for d in ("BSE", "Inlens")]
        panels.append(C.overlay_rgb(ch["BSE"][y0:y1, x0:x1], lab[y0:y1, x0:x1], alpha=0.5))
        gap = np.full((y1 - y0, 8, 3), 255, np.uint8)
        im = Image.fromarray(np.hstack([panels[0], gap, panels[1], gap, panels[2]]))
        for i, name in enumerate(("BSE", "Inlens", "labels")):
            text(im, (8 + i * (x1 - x0 + 8), 6), f"{sid} {name}")
        im.save(FIG / f"hard_{sid}.jpg", quality=88)


def ai_examples(per_class=5):
    path = C.OUT_MET / "ai_superpixel_labels.csv"
    if not path.exists():
        return
    rows = [r for r in C.read_csv(path) if r["kind"] == "ambiguous" and r["confidence"] == "high"]
    rng = np.random.default_rng(3)
    out = []
    for c in C.CLASSES:
        sub = [r for r in rows if r["ai_label"] == c]
        for r in (rng.choice(sub, min(per_class, len(sub)), replace=False) if sub else []):
            im = Image.open(C.PROC / "ai_labels" / "crops" / f"{r['item']}.png").convert("RGB").resize((600, 200))
            text(im, (8, 180), f"AI label: {c}  ({r['reason'][:70]})", fill=(255, 255, 0))
            out.append(np.asarray(im))
    if out:
        Image.fromarray(np.vstack(out)).save(FIG / "ai_label_examples.jpg", quality=85)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    contact_sheet()
    hard_cases()
    ai_examples()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
