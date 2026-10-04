"""Training-data guard and known-location matcher (the teammate's v3 design; edge maps as in his field_matching.py).

guard    an input whose pixels are identical to one of the 93 reference images is a training image: refused.
matcher  the input's particle/pore edges (gradient magnitude at ~200 nm/px, Hann-windowed) are phase-correlated
         with every reference image. The peak is scored in standard deviations of the correlation surface;
         a peak >= MATCH_SD means the input shows a known location (any detector, or a crop of it).

  index      build data/processed/known_spot_index.npz (edge maps + pixel hashes of the 93 reference images)
  evaluate   cross-detector matches, random crops and different-location negatives
             -> outputs/metrics/known_spot_eval.json

Usage: python -m src.known_spot index ; python -m src.known_spot evaluate
"""
import argparse
import hashlib

import numpy as np
from PIL import Image
from scipy import fft, ndimage

from . import config as C

Image.MAX_IMAGE_PIXELS = None
INDEX = C.PROC / "known_spot_index.npz"
EVAL = C.OUT_MET / "known_spot_eval.json"
BIN = 8                 # ~200 nm/px from 25 nm/px
SHAPE = (200, 860)      # common FFT shape (rows, cols) at ~200 nm/px
MATCH_SD = 15.0         # the teammate's threshold on the peak height
MARGIN = 2.0            # and the best location must score >= MARGIN x the best other location (crops raise the noise floor)
DETS = {"BSE": "BSE", "Inlens": "Inlens", "SE2": "ETD/SE"}


def read_raw(path):
    a = np.asarray(Image.open(path))
    return a[..., 1] if a.ndim == 3 else a


def pixel_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes() + str(a.shape).encode()).hexdigest()


def edge_map(a):
    """Windowed, standardised gradient magnitude at ~200 nm/px, zero-padded to SHAPE."""
    a = a[:, :-2].astype(np.float32) if a.shape[1] > 16 else a.astype(np.float32)
    h, w = a.shape[0] // BIN * BIN, a.shape[1] // BIN * BIN
    b = a[:h, :w].reshape(h // BIN, BIN, w // BIN, BIN).mean(axis=(1, 3))[:SHAPE[0], :SHAPE[1]]
    b = ndimage.gaussian_filter(b, 1.0)
    g = np.hypot(ndimage.sobel(b, 0), ndimage.sobel(b, 1))
    g = (g - g.mean()) / (g.std() + 1e-9)
    g = g * np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    out = np.zeros(SHAPE, np.float32)
    out[:g.shape[0], :g.shape[1]] = g
    return out


def match_scores(e, refs_fft):
    """Peak of the normalised cross-power spectrum against every reference, in SDs of the correlation surface."""
    f = fft.rfft2(e)
    cross = refs_fft * np.conj(f)[None]
    corr = fft.irfft2(cross / (np.abs(cross) + 1e-9), s=SHAPE)
    flat = corr.reshape(len(corr), -1)
    return (flat.max(1) - flat.mean(1)) / (flat.std(1) + 1e-12)


def cmd_index(_):
    rows, edges, hashes = [], [], []
    for (batch, sid), dets in C.discover().items():
        for d in DETS:
            a = read_raw(dets[d])
            rows.append((batch, sid, DETS[d]))
            edges.append(edge_map(a).astype(np.float16))
            hashes.append(pixel_hash(a))
    np.savez_compressed(INDEX, batch=[r[0] for r in rows], sid=[r[1] for r in rows], det=[r[2] for r in rows],
                        edges=np.stack(edges), hashes=hashes)
    print(f"known-spot index: {len(rows)} reference images -> {INDEX}")


class Index:
    def __init__(self, exclude_sids=()):
        z = np.load(INDEX)
        keep = np.array([s not in set(exclude_sids) for s in z["sid"]])
        self.batch, self.sid, self.det = z["batch"][keep], z["sid"][keep], z["det"][keep]
        self.hashes = set(z["hashes"][keep])
        self.fft = fft.rfft2(z["edges"][keep].astype(np.float32))

    def check(self, arrays):
        """arrays: {view name: raw image}. Returns the guard and matcher verdict, with the evidence."""
        hit = [v for v, a in arrays.items() if pixel_hash(a) in self.hashes]
        if hit:
            return {"status": "training_image", "views": hit,
                    "note": "pixel-identical to a training image; batch prediction refused (it would not be a test)"}
        best = []
        for v, a in arrays.items():
            s = match_scores(edge_map(a), self.fft)
            k = int(s.argmax())
            second = float(np.sort(s)[-2]) if len(s) > 1 else 0.0
            best.append({"view": v, "best_score_sd": round(float(s[k]), 1), "location": str(self.sid[k]),
                         "batch": str(self.batch[k]), "reference_detector": str(self.det[k]),
                         "next_best_other_score_sd": round(max([float(x) for x, sid in zip(s, self.sid) if sid != self.sid[k]] or [0]), 1)})
        strong = [b for b in best if b["best_score_sd"] >= MATCH_SD and b["best_score_sd"] >= MARGIN * b["next_best_other_score_sd"]]
        if strong and len({b["location"] for b in strong}) == 1:
            b = max(strong, key=lambda r: r["best_score_sd"])
            return {"status": "known_location", "location": b["location"], "batch": b["batch"], "matches": best,
                    "note": f"edge map matches known location {b['location']} at {b['best_score_sd']:.0f} SD "
                            f"(threshold {MATCH_SD:.0f} SD and {MARGIN:.0f}x the next location); its batch is known"}
        return {"status": "new", "matches": best,
                "note": f"no reference image matches at >= {MATCH_SD:.0f} SD with a {MARGIN:.0f}x margin; treated as a new location"}


def cmd_evaluate(_):
    z = np.load(INDEX)
    E = z["edges"].astype(np.float32)
    F = fft.rfft2(E)
    sids, dets = z["sid"], z["det"]
    paths = {(s, DETS[d]): p for (b, s), ds in C.discover().items() for d, p in ds.items() if d in DETS}
    rng = np.random.default_rng(0)
    res = {"full_cross_detector": [], "crop_cross_detector": [], "crop_same_image": [], "negative_full": [], "negative_crop": []}
    accept = {"crop_cross_detector": [], "crop_same_image": [], "negative_full": [], "negative_crop": []}

    def loc_best(s, mask):
        """best score per location among `mask` references, sorted high to low"""
        return sorted((float(s[mask & (sids == l)].max()) for l in set(sids[mask])), reverse=True)
    for i in range(len(sids)):
        a = read_raw(paths[(sids[i], dets[i])])
        h, w = a.shape
        ch, cw = h // 2, w // 2
        y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
        crop_e = edge_map(a[y0:y0 + ch, x0:x0 + cw])
        s_full = match_scores(E[i], F)
        s_crop = match_scores(crop_e, F)
        own = sids == sids[i]
        others_same = own & (np.arange(len(sids)) != i)
        res["full_cross_detector"].append(float(s_full[others_same].max()))
        res["crop_cross_detector"].append(float(s_crop[others_same].max()))
        res["crop_same_image"].append(float(s_crop[i]))
        res["negative_full"].append(float(s_full[~own].max()))
        res["negative_crop"].append(float(s_crop[~own].max()))
        for key, s, mask in (("crop_cross_detector", s_crop, np.arange(len(sids)) != i), ("crop_same_image", s_crop, np.ones(len(sids), bool)),
                             ("negative_full", s_full, ~own), ("negative_crop", s_crop, ~own)):
            lb = loc_best(s, mask)
            accept[key].append(lb[0] >= MATCH_SD and lb[0] >= MARGIN * lb[1])
    out = {"threshold_sd": MATCH_SD, "margin": MARGIN, "n_images": len(sids), "crop": "random 50 % x 50 % crop of each image",
           "with_margin_rule": {k: f"{int(np.sum(v))}/{len(v)}" + (" false matches" if k.startswith("negative") else " found")
                                for k, v in accept.items()}}
    print(out["with_margin_rule"])
    for k, v in res.items():
        v = np.array(v)
        out[k] = {"min": round(float(v.min()), 1), "median": round(float(np.median(v)), 1), "max": round(float(v.max()), 1),
                  ("found_at_threshold" if not k.startswith("negative") else "false_matches_at_threshold"):
                  f"{int((v >= MATCH_SD).sum())}/{len(v)}"}
        print(k, out[k])
    C.write_json(EVAL, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "evaluate"])
    args = ap.parse_args()
    {"index": cmd_index, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
