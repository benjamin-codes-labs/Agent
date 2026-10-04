"""Stage 2: automatic seed labels (confident pixels only; no hand labelling).

Rules (half resolution, all intensities normalised to [0, 1]):
  SiOx      BSE above the upper multi-Otsu threshold, opened (r=3), objects >= 2 um^2, eroded 2 px.
  pore      (Inlens < its p5 AND SE2 < its p10 AND BSE below the SiOx threshold)
            OR (BSE in the dark Otsu class AND SE2 < its p15), eroded 2 px.
            Grey-floored open pores are NOT reliably dark in Inlens (polished graphite is the darkest Inlens
            phase in some samples, e.g. kbdh4tri), so they are left to the AI-annotated superpixels (src.ai_labels).
  graphite  BSE mid-grey AND low Inlens texture (local std at sigma 4 < p40 of mid-grey pixels)
            AND > 6 px from any strong edge (interiors of large smooth regions), eroded 1 px.
  CBD       BSE mid-grey AND high Inlens fine-texture energy (smoothed |DoG(1)-DoG(3)| > p80 of mid-grey)
            AND outside (dilated) graphite interiors AND >= 4 px from any SiOx or pore seed,
            opened with r=5 so only areal lacy patches remain (thin particle-edge lines are dropped).
Pixels claimed by more than one rule are left unlabelled.

Writes data/processed/<sid>/seeds.npy (uint8: 0-3 classes, 255 unlabelled) and prints per-class coverage.

Usage: python -m src.seeds [--overlay SID ...]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.morphology import disk, remove_small_objects

from . import config as C


def local_std(a, sigma):
    m = ndi.gaussian_filter(a, sigma)
    return np.sqrt(np.maximum(ndi.gaussian_filter(a * a, sigma) - m * m, 0))


def make_seeds(ch, exclude):
    bse, inl, se2 = ch["BSE"], ch["Inlens"], ch["SE2"]
    valid = ~exclude
    t_low, t_high = threshold_multiotsu(bse[valid], classes=3)

    siox = ndi.binary_opening(bse > t_high, structure=disk(3))
    siox = remove_small_objects(siox & valid, max_size=int(2 / C.PX_UM2) - 1)
    siox_seed = ndi.binary_erosion(siox, structure=disk(2))

    se2_dark = se2 < np.percentile(se2[valid], 10)
    pore = (inl < np.percentile(inl[valid], 5)) & se2_dark & (bse < t_high)
    pore |= (bse < t_low) & (se2 < np.percentile(se2[valid], 15))     # deep voids: black in BSE and SE2
    pore_seed = ndi.binary_erosion(pore & valid, structure=disk(2))

    mid = (bse >= t_low) & (bse <= t_high) & valid & ~ndi.binary_dilation(siox, structure=disk(3))
    tex = local_std(inl, 4)
    grad = ndi.gaussian_gradient_magnitude(bse, 1.5) + ndi.gaussian_gradient_magnitude(inl, 1.5)
    edges = (grad > np.percentile(grad[valid], 85)) | siox | pore
    far_from_edge = ndi.distance_transform_edt(~edges) > 6
    graphite = mid & (tex < np.percentile(tex[mid], 40)) & far_from_edge
    graphite_seed = ndi.binary_erosion(graphite, structure=disk(1))

    dog = np.abs(ndi.gaussian_filter(inl, 1) - ndi.gaussian_filter(inl, 3))
    energy = ndi.gaussian_filter(dog, 2)
    cbd = mid & (energy > np.percentile(energy[mid], 80))
    cbd &= ~ndi.binary_dilation(graphite, structure=disk(4))
    cbd &= ndi.distance_transform_edt(~(siox_seed | pore_seed)) >= 4
    # keep areal lacy patches only; thin bright lines along particle edges are edge effects, not binder
    cbd_seed = remove_small_objects(ndi.binary_opening(cbd, structure=disk(5)), max_size=60)

    seeds = np.full(bse.shape, C.EXCLUDED, np.uint8)
    masks = [pore_seed, graphite_seed, siox_seed, cbd_seed]
    claimed = np.sum(masks, axis=0)
    for k, m in enumerate(masks):
        seeds[m & (claimed == 1)] = k
    seeds[exclude] = C.EXCLUDED
    return seeds, {"t_low": float(t_low), "t_high": float(t_high)}


def run(key):
    batch, sid = key
    ch = C.load_channels(sid)
    seeds, thr = make_seeds(ch, C.load_exclude(sid))
    np.save(C.proc_dir(sid) / "seeds.npy", seeds)
    frac = {c: round(100 * float((seeds == k).mean()), 2) for k, c in enumerate(C.CLASSES)}
    return batch, sid, frac, thr


def save_overlay(sid, path):
    ch = C.load_channels(sid, ["BSE", "Inlens"])
    seeds = np.load(C.proc_dir(sid) / "seeds.npy")
    shown = np.where(seeds == C.EXCLUDED, 254, seeds)          # unlabelled: no tint
    top = C.overlay_rgb(ch["BSE"], shown, alpha=0.65)
    bottom = np.repeat((ch["Inlens"] * 255).astype(np.uint8)[..., None], 3, -1)
    Image.fromarray(np.vstack([top, bottom])).save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", nargs="*", default=["kbdh4tri", "r17byphk", "epqdaau9"])
    args = ap.parse_args()
    with ProcessPoolExecutor(C.WORKERS) as ex:
        results = list(ex.map(run, C.sample_ids()))
    rows = []
    for batch, sid, frac, thr in results:
        rows.append({"batch": batch, "sample_id": sid, **{f"seed_{c}_pct": v for c, v in frac.items()},
                     "bse_t_low": round(thr["t_low"], 3), "bse_t_high": round(thr["t_high"], 3)})
        print(f"{batch}/{sid}: " + "  ".join(f"{c} {v:5.2f}%" for c, v in frac.items()))
    C.write_csv(C.OUT_MET / "seed_coverage.csv", rows)
    for sid in args.overlay:
        save_overlay(sid, C.REPORTS / "figures" / f"seeds_{sid}.jpg")
    print("seed overlays:", [f"reports/figures/seeds_{s}.jpg" for s in args.overlay])


if __name__ == "__main__":
    (C.REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    main()
