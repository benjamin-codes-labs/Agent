"""Physical constraint shared by the teacher, the student and the predictor.

SiOx is defined by Z-contrast: it is brighter than carbon in BSE. A pixel whose (lightly smoothed) BSE value is
below the image's upper multi-Otsu threshold (the SiOx / carbon split) cannot be SiOx, whatever its texture;
such pixels (particle rims, halos, bright sub-surface walls) get their next most likely class. Small dark
specks inside SiOx particles (mottled / porous particles) are filled back in.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu

from . import config as C

MAX_HOLE_UM2 = 1.0


def bse_t_high(bse, exclude):
    return float(threshold_multiotsu(bse[~exclude], classes=3)[1])


def constrain_siox(probs, bse, exclude, class_axis=-1, stride=1):
    """probs: class probabilities (class axis first or last). Returns a renormalised copy."""
    t = bse_t_high(bse, exclude)
    dark = (ndi.gaussian_filter(bse, 1) < t)[::stride, ::stride]
    p = np.moveaxis(probs, class_axis, -1).copy()
    p[..., C.SIOX][dark] = 0
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-9)
    return np.moveaxis(p, -1, class_axis)


def fill_siox_holes(labels):
    siox = labels == C.SIOX
    holes = ndi.binary_fill_holes(siox) & ~siox
    lab, n = ndi.label(holes)
    if n:
        small = np.flatnonzero(np.bincount(lab.ravel()) <= MAX_HOLE_UM2 / C.PX_UM2)
        fill = np.isin(lab, small[small > 0])
        labels = labels.copy()
        labels[fill & (labels != C.EXCLUDED)] = C.SIOX
    return labels
