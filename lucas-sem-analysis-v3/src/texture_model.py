"""Tile texture batch model (the teammate's v3 design): one logistic regression per detector view.

Each image is cut into 12.8 um tiles (512 px at 25 nm/px). Every tile is described by
  - local binary patterns (uniform, P = 8, radius 1 and 3 px): 2 x 10-bin histograms,
  - grey-level co-occurrence texture (32 levels, distance 1 and 4 px, 0 and 90 deg): contrast, homogeneity,
    energy, correlation,
  - the power spectrum: share of power in 8 log-spaced frequency bands, and x/y anisotropy at high frequency,
then classified per tile with a standardised logistic regression; tile probabilities are averaged per image.
Tiles more than 20 % excluded (Cu foil, unpolished strips, border) are skipped.

  features   compute and cache tile features for every view of the 31 locations -> data/processed/texture_tiles.npz

Usage: python -m src.texture_model features
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

Image.MAX_IMAGE_PIXELS = None
TILE = 512
VIEWS = {"bse": "BSE", "inlens": "Inlens", "se2": "ETD/SE"}
CACHE = C.PROC / "texture_tiles.npz"
PROPS = ("contrast", "homogeneity", "energy", "correlation")
NAMES = ([f"lbp_r1_{k}" for k in range(10)] + [f"lbp_r3_{k}" for k in range(10)] +
         [f"glcm_{p}_d{d}_{a}" for d in (1, 4) for a in ("x", "y") for p in PROPS] +
         [f"spectrum_band{k}" for k in range(8)] + ["spectrum_hf_x_over_y"])
MEANING = {"lbp": "fine local texture pattern frequency", "glcm_contrast": "grey-level contrast between neighbours",
           "glcm_homogeneity": "smoothness between neighbouring pixels", "glcm_energy": "texture uniformity",
           "glcm_correlation": "how predictable a pixel is from its neighbour", "spectrum_band": "power at one spatial scale",
           "spectrum_hf": "fine-scale texture direction (along vs across scan lines)"}

_YY, _XX = np.indices((TILE, TILE))
_R = np.hypot(np.minimum(_YY, TILE - _YY), np.minimum(_XX, TILE - _XX))
_EDGES = np.geomspace(2, TILE / 2, 9)
_BAND = np.digitize(_R, _EDGES) - 1        # 0..7 inside the bands, -1 / 8 outside


def meaning(name):
    for k, v in MEANING.items():
        if name.startswith(k):
            return v
    return name


def read_raw(path):
    return np.asarray(Image.open(path))[..., 1]


def tile_features(a, exclude_half=None):
    """a: raw uint8 image (25 nm/px). exclude_half: optional bool mask at half resolution. -> (n_tiles, 45)"""
    a = a.astype(np.uint8)
    lbp1 = local_binary_pattern(a, 8, 1, "uniform")
    lbp3 = local_binary_pattern(a, 8, 3, "uniform")
    q = a // 8
    rows = []
    for y in range(0, a.shape[0] - TILE + 1, TILE):
        for x in range(0, a.shape[1] - TILE + 1, TILE):
            if exclude_half is not None and exclude_half[y // 2:(y + TILE) // 2, x // 2:(x + TILE) // 2].mean() > 0.2:
                continue
            t = a[y:y + TILE, x:x + TILE].astype(np.float32)
            f = [np.bincount(l[y:y + TILE, x:x + TILE].astype(int).ravel(), minlength=10)[:10] / TILE ** 2
                 for l in (lbp1, lbp3)]
            g = graycomatrix(q[y:y + TILE, x:x + TILE], [1, 4], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
            f.append(np.array([graycoprops(g, p)[d, k] for d in range(2) for k in range(2) for p in PROPS]))
            ps = np.abs(np.fft.fft2(t - t.mean())) ** 2
            band = np.array([ps[_BAND == k].sum() for k in range(8)])
            f.append(np.log10(band / band.sum() + 1e-12))
            hf = _R > TILE / 8
            f.append([np.log10((ps[hf & (_YY == 0)].sum() + 1e-9) / (ps[hf & (_XX == 0)].sum() + 1e-9))])
            rows.append(np.concatenate([np.ravel(v) for v in f]))
    return np.array(rows, np.float32)


def _one(args):
    batch, sid = args
    paths = C.discover()[(batch, sid)]
    exc = np.load(C.proc_dir(sid) / "exclude.npy")
    det = {"bse": "BSE", "inlens": "Inlens", "se2": "SE2"}
    return sid, {v: tile_features(read_raw(paths[d]), exc) for v, d in det.items()}


def cmd_features(_):
    out = {}
    with ProcessPoolExecutor(max(1, min(8, (os.cpu_count() or 2) - 2))) as ex:
        for sid, feats in ex.map(_one, C.sample_ids()):
            for v, f in feats.items():
                out[f"{sid}__{v}"] = f
            print(sid, {v: f.shape[0] for v, f in feats.items()}, flush=True)
    np.savez_compressed(CACHE, **out)
    print(f"tile features for {len(out)} images -> {CACHE}")


def load_tiles():
    z = np.load(CACHE)
    tiles = {}
    for k in z.files:
        sid, v = k.split("__")
        tiles.setdefault(v, {})[sid] = z[k]
    return tiles


def fit_view(tiles_by_sid, labels_by_sid):
    X = np.concatenate([tiles_by_sid[s] for s in labels_by_sid])
    y = np.concatenate([np.full(len(tiles_by_sid[s]), labels_by_sid[s]) for s in labels_by_sid])
    w = np.concatenate([np.full(len(tiles_by_sid[s]), 1.0 / len(tiles_by_sid[s])) for s in labels_by_sid])
    cls_w = {c: 1.0 / sum(1 for s in labels_by_sid if labels_by_sid[s] == c) for c in set(labels_by_sid.values())}
    w = w * np.array([cls_w[c] for c in y])          # each location, and each batch, weighs the same
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    m.fit(X, y, logisticregression__sample_weight=w * len(w) / w.sum())
    return m


def view_proba(m, tiles):
    p = np.zeros(3)
    p[m.classes_] = m.predict_proba(tiles).mean(0)
    return p


def top_features(m, tiles, win, run, top=3):
    """Features that pushed the image toward `win` over `run` (mean standardised value x coefficient difference)."""
    sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
    z = sc.transform(tiles).mean(0)
    ci = {c: i for i, c in enumerate(lr.classes_)}
    if win not in ci or run not in ci:
        return []
    push = (lr.coef_[ci[win]] - lr.coef_[ci[run]]) * z
    return [{"feature": NAMES[j], "meaning": meaning(NAMES[j]), "mean_z": round(float(z[j]), 2),
             "push_toward_winner": round(float(push[j]), 3)} for j in np.argsort(-np.abs(push))[:top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features"])
    args = ap.parse_args()
    {"features": cmd_features}[args.cmd](args)


if __name__ == "__main__":
    main()
