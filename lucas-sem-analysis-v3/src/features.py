"""Stage 3a: per-pixel features for the teacher model.

Per detector (BSE, Inlens, SE2), at sigma = 1, 2, 4, 8 px: Gaussian, gradient magnitude, Laplacian of Gaussian,
both Hessian eigenvalues, local standard deviation (24 features) + the denoised value + fine-texture energy
(smoothed |G1 - G3|) + context (value minus its local mean: sigma 16 px for BSE as a depth/shadow cue,
sigma 8 px for Inlens and SE2 to compensate charging/channelling) -> 27 features per detector.

Writes data/processed/<sid>/feat_<detector>.npy, float16, shape (H, W, 27).

Usage: python -m src.features
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi

from . import config as C

SIGMAS = (1, 2, 4, 8)
CONTEXT_SIGMA = {"BSE": 16, "Inlens": 8, "SE2": 8}


def feature_names(det):
    names = [f"{det}_value", f"{det}_energy", f"{det}_ctx"]
    for s in SIGMAS:
        names += [f"{det}_{n}_s{s}" for n in ("gauss", "gradmag", "log", "hess1", "hess2", "std")]
    return names


def detector_features(a):
    a = a.astype(np.float32)
    out = [a,
           ndi.gaussian_filter(np.abs(ndi.gaussian_filter(a, 1) - ndi.gaussian_filter(a, 3)), 2),
           None]  # context filled by caller
    for s in SIGMAS:
        g = ndi.gaussian_filter(a, s)
        hxx = ndi.gaussian_filter(a, s, order=(0, 2)) * s * s
        hyy = ndi.gaussian_filter(a, s, order=(2, 0)) * s * s
        hxy = ndi.gaussian_filter(a, s, order=(1, 1)) * s * s
        root = np.sqrt(((hxx - hyy) / 2) ** 2 + hxy ** 2)
        mean_h = (hxx + hyy) / 2
        std = np.sqrt(np.maximum(ndi.gaussian_filter(a * a, s) - g * g, 0))
        out += [g, ndi.gaussian_gradient_magnitude(a, s) * s, hxx + hyy, mean_h + root, mean_h - root, std]
    return out


def run(key):
    batch, sid = key
    d = C.proc_dir(sid)
    ch = C.load_channels(sid)
    for det, a in ch.items():
        feats = detector_features(a)
        feats[2] = a - ndi.gaussian_filter(a, CONTEXT_SIGMA[det])
        np.save(d / f"feat_{det}.npy", np.stack(feats, -1).astype(np.float16))
    return sid


def main():
    with ProcessPoolExecutor(C.WORKERS) as ex:
        for sid in ex.map(run, C.sample_ids()):
            print("features done:", sid, flush=True)


if __name__ == "__main__":
    main()
