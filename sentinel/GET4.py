"""
Uncertainty of phase-fraction KPIs from SEM images of one batch.

Question answered: "Given the images we have, how well do we actually know this
batch's phase fractions, what is limiting that, and what should we image next?"

The error bar is built from three factors, each acting at a different length scale:

1. AREA USED (scale of one feature)
   A phase fraction is a mean over pixels, but pixels are correlated: a pixel in a
   particle predicts its neighbours. With C(r) the normalised two-point covariance
   (C(0) = 1), the variance of the mean over an image of area A is

       Var_area = phi(1-phi)/A * sum_r C(r) w_A(r)  ~=  phi(1-phi) * a / A,   a = sum_r C(r)

   so the image is worth N_eff = A / a independent samples, where `a` (the
   "integral range") is roughly the area of one typical feature.
   Information view: Fisher information about phi is N_eff / (phi(1-phi)); bits gained
   over a flat prior are -1/2 log2(2 pi e SE^2). Every 4x area buys exactly one bit.

2. REGULARITY (scale of the image)
   The C(r) model only knows about feature-scale correlation. If the image also has
   larger-scale structure (a through-thickness gradient, a cluster, a crack band),
   L x L tiles will vary MORE than C(r) predicts. The ratio

       D = observed tile variance / predicted tile variance     (largest L with >= 16 tiles)

   is an overdispersion factor: D ~ 1 means the image is statistically regular,
   D > 1 means it is not, and Var_image = Var_area * max(D, 1). We never shrink the
   error bar when D < 1. C(r) is cut at 5 correlation lengths so that large-scale
   structure cannot hide inside the "feature size" and must show up here instead.
   D only probes scales up to ~1/4 of the image. A gradient across the whole image
   height is reported separately as a top-to-bottom trend: if the image spans the
   electrode thickness it is a material property (e.g. binder migration), not noise.

3. DIFFERENCE BETWEEN IMAGES (scale of the electrode)
   No single image can show how much the electrode varies from place to place.
   Fields of view are combined with a random-effects model (as in meta-analysis):

       phi_i = mu + b_i + e_i,   b_i ~ N(0, tau^2),   e_i ~ N(0, Var_image_i)

   tau^2 (true field-to-field variation) is estimated by DerSimonian-Laird; the batch
   CI uses the Hartung-Knapp correction with a t distribution, which is honest when
   there are only a few images. I^2 = share of the observed spread that is real
   field-to-field variation rather than sampling noise.

The batch variance splits exactly into the three factors (the "uncertainty budget"),
which also tells you what to do next: if AREA dominates, image bigger fields; if
BETWEEN dominates, image more locations; if REGULARITY dominates, look at the image.

Segmentation sensitivity (threshold nudged by +-2% of the grey range) is systematic,
does not shrink with more images, and is reported separately.

Imaging conditions (so an imaging change is not mistaken for a material change)
--------------------------------------------------------------------------------
SCALE: every image is resampled to one physical pixel size before anything else
   (the baseline's, if given, else the coarsest image x --bin), so smoothing, tiles
   and thresholds act on the same physical scale. Missing / mixed pixel sizes are
   refused or flagged; a phase whose features are < 5 px is flagged resolution-limited.
   --tilt-deg corrects the y foreshortening of FIB cross-sections imaged under tilt.
IMAGING QUALITY: sharpness, directional blur (astigmatism), noise, brightness,
   contrast, clipping, phase separability and stripe strength are measured per image
   and compared with the baseline images (or the rest of the batch). An image outside
   the reference range is reported as an IMAGING difference to investigate first.
SURFACE ROUGHNESS / SHADING: smooth shading is removed by fitting a low-order surface
   to the majority phase only (so real composition gradients survive); curtaining and
   scan lines are notch-filtered in Fourier space when detected. Remaining local
   shading is measured by re-fitting thresholds per tile: the phi difference between
   local and global thresholds is the roughness part of the segmentation error.
   Not handled: "pore-back" (material deeper inside pores showing through).
   Irregular / organic particle SHAPES need no special handling: C(r) assumes none.

Note: images passed together must be different LOCATIONS imaged with the SAME
detector. The same field seen by BSE / SE / InLens is not three independent samples.
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile
from scipy import fft, ndimage, stats
from skimage import filters, io, transform

PHASES = {0: "pore", 1: "graphite", 2: "bright phase"}
Z95 = 1.959964
IMAGE_SUFFIXES = {".tif", ".tiff", ".png"}
FULL_FFT_MAX_PX = 40_000_000  # above this, accumulate the correlation over tiles
FFT_TILE = 4096


# ---------------------------------------------------------------- loading

UNIT_TO_NM = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "µm": 1e3, "mm": 1e6, "m": 1e9}


def _pixel_size_nm(tif):
    """Best-effort pixel size from FEI/Thermo or Zeiss SEM metadata."""
    fei = tif.fei_metadata
    if fei:
        try:
            return float(fei["Scan"]["PixelWidth"]) * 1e9
        except (KeyError, TypeError, ValueError):
            pass
    sem = tif.sem_metadata
    if sem:
        for key, value in sem.items():
            if "pixel_size" not in key.lower():
                continue
            text = " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
            m = re.search(r"([\d.]+(?:e-?\d+)?)\s*(pm|nm|um|µm|mm|m)\b", text)
            if m:
                return float(m.group(1)) * UNIT_TO_NM[m.group(2)]
    ij = tif.imagej_metadata
    if ij and "unit" in ij:  # FIJI / ImageJ calibration: XResolution = pixels per unit
        unit = {"micron": "um", "\\u00B5m": "um"}.get(ij["unit"], ij["unit"])
        tag = tif.pages[0].tags.get("XResolution")
        if tag and unit in UNIT_TO_NM:
            num, den = tag.value
            if num:
                return den / num * UNIT_TO_NM[unit]
    # standard TIFF resolution tags (pixels per inch / cm); ignore placeholder values like 72 dpi
    page = tif.pages[0]
    xres, unit = page.tags.get("XResolution"), page.tags.get("ResolutionUnit")
    per_unit_nm = {2: 25.4e6, 3: 1e7}.get(int(unit.value) if unit else 2)
    if xres and per_unit_nm and xres.value[0]:
        px = per_unit_nm * xres.value[1] / xres.value[0]
        if px < 10_000:  # < 10 um/px: a real calibration, not a screen-dpi default
            return px
    return None


def _databar_rows(tif, height):
    """FEI/Thermo images store the true image height; anything below it is the info bar."""
    fei = tif.fei_metadata
    try:
        return max(0, height - int(fei["Image"]["ResolutionY"]))
    except (KeyError, TypeError, ValueError):
        return 0


def _bin_chunked(arr, b, chunk_rows=2048):
    """Grey-convert and bin in row chunks so a memory-mapped TIF never loads whole."""
    h, w = arr.shape[0] // b * b, arr.shape[1] // b * b
    out = np.empty((h // b, w // b), np.float32)
    step = max(b, chunk_rows // b * b)
    for y in range(0, h, step):
        blk = np.asarray(arr[y: min(y + step, h), :w], dtype=np.float32)
        if blk.ndim == 3:
            blk = blk[..., :3].mean(axis=-1)
        out[y // b: (y + blk.shape[0]) // b] = blk.reshape(blk.shape[0] // b, b, w // b, b).mean(axis=(1, 3))
    return out


def read_meta(path, page=0):
    """Cheap metadata pass (no pixel data), so the batch scale can be chosen first."""
    meta = {"pixel_size_nm": None, "databar_rows": 0, "n_pages": 1}
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            meta["pixel_size_nm"] = _pixel_size_nm(tif)
            meta["databar_rows"] = _databar_rows(tif, tif.pages[page].shape[0])
            meta["n_pages"] = len(tif.pages)
    return meta


def _open_array(path, page):
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        try:
            return tifffile.memmap(path, page=page, mode="r")
        except Exception:  # compressed or non-contiguous: read normally
            return tifffile.imread(path, key=page)
    return io.imread(path)


def load_image(path, meta, page=0, crop_bottom=None, px_nm=None, target_nm=None, bin_factor=2, tilt_deg=None):
    """Crop the databar, scale grey levels to [0, 1] of the detector range and resample to
    square pixels of target_nm (integer block-binning, then an anti-aliased resize for the
    remainder). Without pixel sizes this is plain binning by bin_factor."""
    arr = _open_array(path, page)
    rows = meta["databar_rows"] if crop_bottom is None else crop_bottom
    if rows:
        arr = arr[: arr.shape[0] - rows]
    if np.issubdtype(arr.dtype, np.integer):
        full_scale = float(np.iinfo(arr.dtype).max)
    else:
        full_scale = float(np.nanmax(np.asarray(arr[::16, ::16]))) or 1.0

    factor = target_nm / px_nm if (px_nm and target_nm) else float(bin_factor)
    b = max(1, int(np.floor(factor + 1e-9)))
    img = _bin_chunked(arr, b) / full_scale
    rx = factor / b
    ry = rx * np.sin(np.radians(tilt_deg)) if tilt_deg else rx  # tilt: each row spans more real length
    shape = (max(1, int(round(img.shape[0] / ry))), max(1, int(round(img.shape[1] / rx))))
    if shape != img.shape:
        img = transform.resize(img, shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    info = {"databar_rows": rows, "shape_px": [int(arr.shape[0]), int(arr.shape[1])],
            "full_scale": full_scale, "scale_factor": float(factor),
            "shape_analysed": [int(img.shape[0]), int(img.shape[1])], "upsampled": bool(factor < 1)}
    return img, info


# ---------------------------------------------------------------- imaging quality + preprocessing

STRIPE_LIMIT = 3.0   # stripe index above which we destripe / flag
F0 = 1 / 64          # cycles/px: structure coarser than 64 px is never touched by destriping


def _central_crop(img, size=2048):
    h, w = img.shape
    ch, cw = min(h, size), min(w, size)
    return img[(h - ch) // 2: (h - ch) // 2 + ch, (w - cw) // 2: (w - cw) // 2 + cw]


def separability(values, thresholds):
    """Otsu's eta: between-class / total grey-level variance (1 = phases perfectly separated)."""
    v = values.ravel()
    lab = np.digitize(v, thresholds)
    mu = v.mean()
    between = sum((lab == k).mean() * (v[lab == k].mean() - mu) ** 2 for k in np.unique(lab))
    return float(between / max(v.var(), 1e-12))


def stripe_index(img, step=0.5):
    """Straight-streak artefacts: FIB curtaining (any angle) and raster scan lines (horizontal).

    Straight streaks put their power in a narrow wedge of the Fourier spectrum, at right
    angles to the streaks. Index = power in a +-1 deg wedge / power 6-12 deg either side,
    over wavelengths 4-32 px; ~1 for a clean image. Particle shape anisotropy gives broad
    bumps, not narrow wedges, so it barely registers. Angle 0 = vertical streaks."""
    c = _central_crop(img)
    if min(c.shape) < 128:
        return {"curtaining": 1.0, "curtaining_angle": 0.0, "scan_lines": 1.0}
    c = (c - c.mean()) * np.outer(np.hanning(c.shape[0]), np.hanning(c.shape[1]))
    p = np.abs(fft.fftshift(fft.fft2(c))) ** 2
    fy = (np.arange(c.shape[0]) - c.shape[0] // 2) / c.shape[0]
    fx = (np.arange(c.shape[1]) - c.shape[1] // 2) / c.shape[1]
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    band = (np.hypot(fyy, fxx) > 1 / 32) & (np.hypot(fyy, fxx) < 1 / 4)
    ang = np.degrees(np.arctan2(fyy[band], fxx[band])) % 180
    bins = (ang / step).astype(int)
    n_bins = int(180 / step)
    order = np.argsort(bins)
    edges = np.searchsorted(bins[order], np.arange(n_bins + 1))
    pv = p[band][order]
    med = np.array([np.median(pv[edges[i]: edges[i + 1]]) if edges[i + 1] > edges[i] else np.nan
                    for i in range(n_bins)])

    def wedge(a, lo, hi):
        d = np.abs((np.arange(n_bins) * step - a + 90) % 180 - 90)
        return np.nanmean(med[(d >= lo) & (d <= hi)])

    ratios = np.array([wedge(a, 0, 1) / wedge(a, 6, 12) for a in np.arange(n_bins) * step])
    angles = np.arange(n_bins) * step
    curtain = np.abs((angles - 90 + 90) % 180 - 90) > 3  # scan lines live at exactly 90 deg
    i = int(np.nanargmax(np.where(curtain, ratios, -np.inf)))
    signed = (angles[i] + 90) % 180 - 90
    return {"curtaining": float(ratios[i]), "curtaining_angle": float(signed),
            "scan_lines": float(ratios[int(90 / step)])}


def noise_sigma(img):
    """Immerkaer (1996) fast noise estimate: a Laplacian-difference kernel cancels smooth
    structure, leaving mostly pixel noise."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = ndimage.convolve(img, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def image_quality(img):
    """Imaging metrics on the resampled image (so they compare across magnifications)."""
    sub = img[::2, ::2]
    p1, p50, p99 = np.percentile(sub, [1, 50, 99])
    contrast = float(max(p99 - p1, 1e-6))
    sm = ndimage.gaussian_filter(img, 1.0)
    gx, gy = ndimage.sobel(sm, axis=1), ndimage.sobel(sm, axis=0)
    thresholds = filters.threshold_multiotsu(sm[::4, ::4], classes=3)
    q = {
        "median": float(p50),
        "contrast": contrast,
        "clipped_black": float(np.mean(sub <= 0.002)),
        "clipped_white": float(np.mean(sub >= 0.998)),
        "sharpness": float(np.percentile(np.hypot(gx, gy)[::2, ::2], 99) / contrast),
        "anisotropy": float(np.sqrt(np.mean(gx ** 2) / max(np.mean(gy ** 2), 1e-12))),
        "noise": noise_sigma(_central_crop(img, 1024)) / contrast,
        "separability": separability(sm[::4, ::4], thresholds),
    }
    q.update({f"stripes_{k}": v for k, v in stripe_index(img).items()})
    return q


def _poly_terms(y, x, degree):
    return [y ** i * x ** j for i in range(degree + 1) for j in range(degree + 1 - i)]


def flatten_shading(img, degree=2, rows=256, iterations=2):
    """Remove smooth shading (charging, detector geometry, uneven tilt) as a gain AND an
    offset map. A low-order surface is fitted to the grey level of each of the two most
    abundant phases; since each phase should look the same everywhere, the two surfaces
    fix gain g(x) and offset o(x) with I = g * I_true + o. Fitting phase grey LEVELS (not
    the overall image) means real composition gradients are not flattened away.
    Returns (corrected image, shading amplitude in grey levels)."""
    h, w = img.shape
    s = max(1, h // rows)
    small = ndimage.uniform_filter(img, s)[::s, ::s] if s > 1 else img.copy()
    y = (np.arange(h, dtype=np.float32) / h)[:, None]
    x = (np.arange(w, dtype=np.float32) / w)[None, :]
    gain, offset = np.ones((h, w), np.float32), np.zeros((h, w), np.float32)
    corrected = small
    for _ in range(iterations):
        lab = np.digitize(corrected, filters.threshold_multiotsu(corrected, classes=3))
        counts = np.bincount(lab.ravel(), minlength=3)
        a_cls, b_cls = np.argsort(counts)[::-1][:2]
        surfaces, levels = [], []
        for cls in (a_cls, b_cls):
            yy, xx = np.nonzero(lab == cls)
            if len(yy) < 100:
                return img, 0.0
            design = np.stack(_poly_terms(yy * s / h, xx * s / w, degree), axis=1)
            coef, *_ = np.linalg.lstsq(design, small[yy, xx], rcond=None)
            surfaces.append(coef)
            levels.append(float(np.median(small[yy, xx])))
        if abs(levels[0] - levels[1]) < 1e-3:
            return img, 0.0
        sa = sum(np.float32(c) * t for c, t in zip(surfaces[0], _poly_terms(y, x, degree)))
        sb = sum(np.float32(c) * t for c, t in zip(surfaces[1], _poly_terms(y, x, degree)))
        gain = np.broadcast_to(np.clip((sa - sb) / (levels[0] - levels[1]), 0.5, 2.0), (h, w))
        offset = np.broadcast_to(sa - gain * levels[0], (h, w))
        # re-label on the corrected small image for the next iteration
        g_small, o_small = gain[::s, ::s][: small.shape[0], : small.shape[1]], offset[::s, ::s][: small.shape[0], : small.shape[1]]
        corrected = (small - o_small) / g_small
    out = ((img - offset) / gain).astype(np.float32)
    # grey-level drift of a phase across the image (5-95% range: polynomial corners overshoot)
    amp = float(max(np.subtract(*np.percentile(np.broadcast_to(srf, (h, w))[::s, ::s], [95, 5])) for srf in (sa, sb)))
    return out, amp


def destripe(img, q, width=1.5):
    """Notch the detected stripe lines out of the Fourier spectrum, keeping |f| < F0 intact."""
    applied = []
    lines = []
    if q["stripes_curtaining"] > STRIPE_LIMIT:
        lines.append((q["stripes_curtaining_angle"], f"curtaining ({q['stripes_curtaining_angle']:+.0f} deg)"))
    if q["stripes_scan_lines"] > STRIPE_LIMIT:
        lines.append((90.0, "horizontal scan lines"))
    if not lines or img.size > FULL_FFT_MAX_PX:
        return img, applied
    h, w = img.shape
    f = fft.rfft2(img, workers=-1)
    fy = fft.fftfreq(h).astype(np.float32)[:, None]
    fx = fft.rfftfreq(w).astype(np.float32)[None, :]
    # the notch is ~1.5 frequency bins wide, so it can go almost down to zero frequency
    # without touching real structure; only the lowest few bins (image-scale shading) are kept
    low = np.hypot(fy, fx) < 4 / min(h, w)
    mask = np.ones(f.shape, np.float32)
    for angle, label in lines:
        a = np.radians(angle)  # the streaks' power lies on the line through 0 at this angle
        d = np.abs(fy * np.cos(a) - fx * np.sin(a))
        mask *= np.where(low, 1.0, 1 - np.exp(-0.5 * (d * min(h, w) / width) ** 2))
        applied.append(label)
    return fft.irfft2(f * mask, s=img.shape, workers=-1).astype(np.float32), applied


# ---------------------------------------------------------------- segmentation

def segment_bse(img, smooth=1.0):
    """Placeholder 3-phase segmentation of a BSE image: pore / graphite / bright phase."""
    sm = ndimage.gaussian_filter(img, smooth)
    thresholds = filters.threshold_multiotsu(sm[::4, ::4], classes=3)
    return np.digitize(sm, thresholds), thresholds, sm


def segmentation_sensitivity(sm, thresholds, frac=0.02):
    """Phase-fraction range when each threshold moves by +-frac of the grey-level range."""
    lo, hi = np.percentile(sm[::4, ::4], [1, 99])
    d = frac * (hi - lo)
    fractions = []
    for s0 in (-d, d):
        for s1 in (-d, d):
            labels = np.digitize(sm, thresholds + np.array([s0, s1]))
            fractions.append(np.bincount(labels.ravel(), minlength=len(PHASES)) / labels.size)
    fractions = np.array(fractions)
    return fractions.min(axis=0), fractions.max(axis=0), fractions


def local_segmentation_fractions(sm, global_t, labels, tile=None, max_shift=0.15, min_class=0.02):
    """Phase fractions with thresholds that follow the local grey level of each phase.

    On a rough or unevenly lit surface a phase's grey level drifts across the image. Per
    tile we measure how far each phase's mean grey level sits from its global mean and
    move each threshold by the average drift of the two phases it separates (a
    proportion-independent measure, unlike re-running Otsu per tile). The phi difference
    between local and global thresholds is the roughness/shading part of the
    segmentation error. A phase only counts in a tile if it covers >= 2% of it, and a
    shift is only applied if it stays within 15% of the grey range."""
    h, w = sm.shape
    tile = tile or int(np.clip(min(h, w) // 4, 128, 512))
    lo, hi = np.percentile(sm[::4, ::4], [1, 99])
    ny, nx = max(1, round(h / tile)), max(1, round(w / tile))
    ys, xs = np.linspace(0, h, ny + 1).astype(int), np.linspace(0, w, nx + 1).astype(int)
    global_t = np.asarray(global_t, np.float32)
    lab_s, sm_s = labels[::2, ::2], sm[::2, ::2]
    global_means = np.array([sm_s[lab_s == k].mean() if (lab_s == k).any() else np.nan for k in range(len(PHASES))])
    grid = np.tile(global_t, (ny, nx, 1))
    refit = 0
    for i in range(ny):
        for j in range(nx):
            block = (slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
            lb, vb = labels[block][::2, ::2].ravel(), sm[block][::2, ::2].ravel()
            frac = np.bincount(lb, minlength=len(PHASES)) / lb.size
            drift = np.array([vb[lb == k].mean() - global_means[k] if frac[k] >= min_class else np.nan
                              for k in range(len(PHASES))])
            moved = False
            for t in range(len(global_t)):
                shift = np.nanmean(drift[t: t + 2]) if not np.all(np.isnan(drift[t: t + 2])) else np.nan
                if np.isfinite(shift) and abs(shift) <= max_shift * (hi - lo):
                    grid[i, j, t] = global_t[t] + shift
                    moved = True
            refit += moved
    local = np.zeros(sm.shape, np.uint8)
    for k in range(grid.shape[-1]):
        tmap = transform.resize(grid[..., k], (h, w), order=1, preserve_range=True, anti_aliasing=False)
        local += sm >= tmap.astype(np.float32)
    return np.bincount(local.ravel(), minlength=len(PHASES)) / local.size, refit, ny * nx


# ---------------------------------------------------------------- factor 1: area (two-point statistics)

def _raw_autocorr(m, max_lag):
    """Zero-padded FFT autocorrelation for |dy|, |dx| <= max_lag, plus pair counts."""
    h, w = m.shape
    shape = (h + max_lag, w + max_lag)
    f = fft.rfft2(m, s=shape, workers=-1)
    corr = fft.irfft2(f * np.conj(f), s=shape, workers=-1)
    corr = np.roll(corr, (max_lag, max_lag), axis=(0, 1))[: 2 * max_lag + 1, : 2 * max_lag + 1]
    lags = np.abs(np.arange(-max_lag, max_lag + 1))
    overlap = np.outer(np.clip(h - lags, 0, None), np.clip(w - lags, 0, None))
    return corr.astype(np.float64), overlap.astype(np.float64)


def normalised_covariance(mask, max_lag, tile=None):
    """C(dy, dx). For huge images the pair sums are accumulated over tiles (bounded memory)."""
    m = mask.astype(np.float32)
    phi = float(m.mean())
    h, w = m.shape
    blocks = [m] if tile is None else [m[y: y + tile, x: x + tile]
                                       for y in range(0, h, tile) for x in range(0, w, tile)]
    acc = np.zeros((2 * max_lag + 1,) * 2)
    cnt = np.zeros_like(acc)
    for b in blocks:
        if min(b.shape) > max_lag:
            c, o = _raw_autocorr(b, max_lag)
            acc += c
            cnt += o
    if not 0 < phi < 1:
        return np.zeros_like(acc), phi
    s2 = acc / np.maximum(cnt, 1)
    return (s2 - phi ** 2) / (phi * (1 - phi)), phi


FEATURE_CUTOFF = 5  # in correlation lengths: keeps ~96% of an exponential C(r)


def integral_range(cov, max_lag, corr_len=None):
    """Sum C(r) out to the first zero of its radial average, or FEATURE_CUTOFF correlation
    lengths if sooner. The cap stops large-scale structure (gradients) from posing as
    "bigger features": anything beyond it must show up in the tile test as irregularity.
    Returns (a, r_cut, truncated, C_trunc)."""
    yy, xx = np.indices(cov.shape) - max_lag
    r = np.rint(np.hypot(yy, xx)).astype(int)
    profile = (np.bincount(r.ravel(), cov.ravel()) / np.bincount(r.ravel()))[: max_lag + 1]
    zeros = np.nonzero(profile <= 0)[0]
    r_cut = max_lag if len(zeros) == 0 else int(zeros[0])
    if corr_len:
        r_cut = min(r_cut, int(np.ceil(FEATURE_CUTOFF * corr_len)))
    truncated = r_cut >= max_lag
    cov_trunc = np.where(r < r_cut, cov, 0.0)
    return float(cov_trunc.sum()), r_cut, truncated, cov_trunc


def correlation_length(cov, max_lag, axis):
    """Lag at which C drops below 1/e along x or y: a feature-size proxy."""
    line = cov[max_lag, max_lag:] if axis == "x" else cov[max_lag:, max_lag]
    below = np.nonzero(line < 1 / np.e)[0]
    return int(below[0]) if len(below) else None


def predicted_se(cov_trunc, max_lag, phi, height, width):
    """Exact finite-window SE of phi_hat for a height x width window."""
    ky, kx = min(height - 1, max_lag), min(width - 1, max_lag)
    sub = cov_trunc[max_lag - ky: max_lag + ky + 1, max_lag - kx: max_lag + kx + 1]
    wy = 1 - np.abs(np.arange(-ky, ky + 1)) / height
    wx = 1 - np.abs(np.arange(-kx, kx + 1)) / width
    var = phi * (1 - phi) * (wy[:, None] * wx[None, :] * sub).sum() / (height * width)
    return float(np.sqrt(max(var, 0.0)))


def info_bits(se):
    """Bits gained about phi relative to a flat prior on [0, 1]."""
    return float(-0.5 * np.log2(2 * np.pi * np.e * se ** 2)) if se > 0 else float("inf")


# ---------------------------------------------------------------- factor 2: regularity

def tile_scaling(mask, sizes, min_tiles=8):
    """Observed std of phi across non-overlapping L x L tiles."""
    m = mask.astype(np.float32)
    rows = []
    for s in sizes:
        ny, nx = m.shape[0] // s, m.shape[1] // s
        n = ny * nx
        if n < min_tiles:
            continue
        tiles = m[: ny * s, : nx * s].reshape(ny, s, nx, s).mean(axis=(1, 3))
        sd = float(tiles.std(ddof=1))
        rows.append({"side": int(s), "n_tiles": int(n), "observed_se": sd,
                     "observed_se_err": sd / np.sqrt(2 * (n - 1))})
    return rows


def dispersion(tiles, min_tiles=16):
    """Overdispersion D at the largest tile size with enough tiles, with a 90% interval."""
    ok = [t for t in tiles if t["n_tiles"] >= min_tiles and t["predicted_se"] > 0]
    if not ok:
        return None
    t = ok[-1]
    d = (t["observed_se"] / t["predicted_se"]) ** 2
    dof = t["n_tiles"] - 1
    return {"D": d, "D_90": [d * dof / stats.chi2.ppf(0.95, dof), d * dof / stats.chi2.ppf(0.05, dof)],
            "tile_side": t["side"], "n_tiles": t["n_tiles"]}


def depth_trend(mask, cov_trunc, max_lag, phi, n_bands=8):
    """Change in phi from top to bottom of the image (assumes y = through-thickness)."""
    h, w = mask.shape
    bh = h // n_bands
    if bh < 2:
        return None
    y = (np.arange(n_bands) + 0.5) / n_bands
    f = np.array([mask[i * bh: (i + 1) * bh].mean() for i in range(n_bands)])
    slope = np.polyfit(y, f, 1)[0]
    se_band = predicted_se(cov_trunc, max_lag, phi, bh, w)
    slope_se = se_band / np.sqrt(((y - y.mean()) ** 2).sum())
    return {"change_top_to_bottom": float(slope), "se": float(slope_se), "band_phi": f.tolist()}


# ---------------------------------------------------------------- per-image analysis

def analyse_phase(mask, max_lag, tile, sizes, unit_len):
    h, w = mask.shape
    area = h * w
    cov, phi = normalised_covariance(mask, max_lag, tile)
    lx, ly = correlation_length(cov, max_lag, "x"), correlation_length(cov, max_lag, "y")
    corr_len = np.sqrt(lx * ly) if lx and ly else None
    a, r_cut, truncated, cov_trunc = integral_range(cov, max_lag, corr_len)

    var_floor = 0.25 / area  # keeps absent / saturated phases from getting zero variance
    var_area = max(predicted_se(cov_trunc, max_lag, phi, h, w) ** 2, var_floor)

    tiles = tile_scaling(mask, sizes)
    for row in tiles:
        s = row["side"]
        row["predicted_se"] = predicted_se(cov_trunc, max_lag, phi, s, s)
        row["naive_se"] = float(np.sqrt(phi * (1 - phi)) / s)
    disp = dispersion(tiles)
    d_used = max(disp["D"], 1.0) if disp else 1.0
    var_irreg = var_area * (d_used - 1.0)
    var_image = var_area + var_irreg

    start = tiles[-1]["side"] if tiles else 4
    extrapolated = [{"side": float(s), "predicted_se": predicted_se(cov_trunc, max_lag, phi, int(s), int(s))}
                    for s in np.geomspace(start, np.sqrt(area), 8)]

    return {
        "phi": phi,
        "var_area": var_area,
        "var_irregularity": var_irreg,
        "var_image": var_image,
        "se_image": float(np.sqrt(var_image)),
        "se_naive_pixels": float(np.sqrt(phi * (1 - phi) / area)),
        "dispersion": disp,
        "integral_range": a * unit_len ** 2,
        "integral_range_truncated": truncated,
        "corr_length_x": None if lx is None else lx * unit_len,
        "corr_length_y": None if ly is None else ly * unit_len,
        "resolution_limited": bool(lx and ly and min(lx, ly) < 5),
        "n_eff": float(area / a) if a > 0 else float(area),
        "bits": info_bits(np.sqrt(var_image)),
        "depth_trend": depth_trend(mask, cov_trunc, max_lag, phi),
        "tiles": tiles,
        "extrapolated": extrapolated,
    }


def analyse_image(path, meta, target_nm, args):
    px = args.px_size or meta["pixel_size_nm"]
    plain_bin = args.bin or 2
    img, info = load_image(path, meta, args.page, args.crop_bottom, px, target_nm, plain_bin, args.tilt_deg)
    unit, unit_len = ("nm", target_nm) if target_nm else ("px", float(plain_bin))

    quality = image_quality(img)
    preprocessing = []
    if not args.no_flatten:
        img, amp = flatten_shading(img)
        quality["shading"] = amp / quality["contrast"]
        preprocessing.append(f"shading flattened (was {100 * quality['shading']:.0f}% of contrast)")
    applied = []
    if not args.no_destripe:
        img, applied = destripe(img, quality)
        preprocessing += [f"removed {a}" for a in applied]
    after = stripe_index(img) if applied else {}
    quality["stripes_curtaining_after"] = after.get("curtaining", quality["stripes_curtaining"])
    quality["stripes_scan_lines_after"] = after.get("scan_lines", quality["stripes_scan_lines"])

    labels, thresholds, sm = segment_bse(img)
    seg_lo, seg_hi, seg_variants = segmentation_sensitivity(sm, thresholds)
    phi_local, n_refit, n_tiles = local_segmentation_fractions(sm, thresholds, labels)

    h, w = labels.shape
    tile = None if h * w <= FULL_FFT_MAX_PX else FFT_TILE
    max_lag = min(h, w) // 2 if tile is None else FFT_TILE // 2
    if args.max_lag:
        max_lag = min(max_lag, args.max_lag)
    sizes = np.unique(np.geomspace(4, min(h, w) // 2, 16).astype(int))

    result = {"image": Path(path).name, **meta, **info, "pixel_size_nm_used": px, "target_pixel_nm": target_nm,
              "unit": unit, "thresholds": thresholds.tolist(), "quality": quality, "preprocessing": preprocessing,
              "local_threshold_tiles": [n_refit, n_tiles], "phases": {}}
    for k, name in PHASES.items():
        r = analyse_phase(labels == k, max_lag, tile, sizes, unit_len)
        r["threshold_range"] = [float(seg_lo[k]), float(seg_hi[k])]
        r["phi_local_threshold"] = float(phi_local[k])
        r["shading_shift"] = float(phi_local[k] - r["phi"])
        r["systematic_range"] = [float(min(seg_lo[k], phi_local[k])), float(max(seg_hi[k], phi_local[k]))]
        # the same segmentation variants in every image/lot, so lot comparisons can cancel common-mode bias:
        # 4 threshold nudges (-/-, -/+, +/-, +/+) and local thresholds
        r["variant_shifts"] = [float(v - r["phi"]) for v in list(seg_variants[:, k]) + [phi_local[k]]]
        result["phases"][name] = r

    print_image_summary(result)
    if not args.no_plots:
        plot_image_report(labels, result, unit_len, Path(args.out) / f"{Path(path).stem}_uncertainty.png")
    return result


# ---------------------------------------------------------------- imaging comparison

QUALITY_TOLERANCE = {  # metric: (relative or absolute, tolerance) vs the reference images
    "sharpness": ("rel", 0.25), "noise": ("rel", 0.30), "contrast": ("rel", 0.25),
    "anisotropy": ("rel", 0.15), "median": ("abs", 0.08), "separability": ("abs", 0.05),
    "stripes_curtaining": ("rel", 0.25), "stripes_scan_lines": ("rel", 0.25),
}
QUALITY_LIMITS = {  # absolute sanity limits: metric: (min, max)
    # black clipping mostly hits pores (darkest phase anyway); white clipping erases bright-phase detail
    "clipped_black": (None, 0.05), "clipped_white": (None, 0.01), "separability": (0.70, None), "shading": (None, 0.15),
    # stripes strong enough to need removal are flagged even when removal worked: it leaves residual error
    "stripes_curtaining": (None, STRIPE_LIMIT), "stripes_scan_lines": (None, STRIPE_LIMIT),
    "stripes_curtaining_after": (None, STRIPE_LIMIT), "stripes_scan_lines_after": (None, STRIPE_LIMIT),
}


def quality_flags(q, ref):
    """Imaging metrics outside absolute limits or outside the reference images' range.
    With >= 3 references the band is +-3 sd (never tighter than the tolerance), else the tolerance."""
    flags = []
    for key, (lo, hi) in QUALITY_LIMITS.items():
        v = q.get(key)
        if v is not None and lo is not None and v < lo:
            flags.append(f"{key} {v:.3f} below limit {lo}")
        if v is not None and hi is not None and v > hi:
            flags.append(f"{key} {v:.3f} above limit {hi}")
    for key, (kind, tol) in QUALITY_TOLERANCE.items():
        vals = np.array([r[key] for r in ref if key in r])
        if not len(vals) or key not in q:
            continue
        center = float(np.median(vals))
        dev = q[key] / center - 1 if kind == "rel" else q[key] - center
        band = tol
        if len(vals) >= 3:
            band = max(tol, 3 * vals.std(ddof=1) / (abs(center) if kind == "rel" else 1.0))
        if abs(dev) > band:
            shown = f"{100 * dev:+.0f}%" if kind == "rel" else f"{dev:+.3f}"
            flags.append(f"{key} {q[key]:.3f} vs reference {center:.3f} ({shown})")
    return flags


def resolve_target(metas, args, baseline):
    """One physical pixel size for the whole comparison. Returns (target_nm, note)."""
    pxs = [args.px_size or m["pixel_size_nm"] for m in metas]
    known = [p for p in pxs if p]
    if known and len(known) < len(pxs):
        raise ValueError("some images have no pixel size in their metadata: pass --px-size")
    if args.target_px:
        target, note = args.target_px, "set by --target-px"
    elif baseline and baseline.get("target_pixel_nm"):
        target, note = baseline["target_pixel_nm"], "taken from baseline"
    elif known:
        b = args.bin or 1
        target, note = max(known) * b, "coarsest image" + (f" x bin {b}" if b > 1 else "")
    else:
        return None, f"NO PIXEL SIZES: plain {args.bin or 2}x binning, scale consistency cannot be checked"
    if not known:
        raise ValueError("a target pixel size needs image pixel sizes: pass --px-size")
    if max(known) / min(known) > 1.05:
        note += f"; mixed magnifications ({min(known):.2f}-{max(known):.2f} nm/px) resampled to match"
    return target, note


# ---------------------------------------------------------------- factor 3: between images

def random_effects(phis, var_area, var_irreg, tau2_prior=None):
    """DerSimonian-Laird random-effects pooling with Hartung-Knapp CI and an exact variance budget."""
    phis, var_area, var_irreg = map(np.asarray, (phis, var_area, var_irreg))
    v = var_area + var_irreg
    k = len(phis)
    w = 1 / v
    mu_fe = (w * phis).sum() / w.sum()
    q = float((w * (phis - mu_fe) ** 2).sum())
    if k > 1:
        c = w.sum() - (w ** 2).sum() / w.sum()
        tau2, tau2_source = max(0.0, (q - (k - 1)) / c), "estimated"
        if tau2_prior is not None and k < 3 and tau2_prior >= tau2:
            # too few images to trust the estimate; borrow the baseline's unless these images disagree more
            tau2, tau2_source = tau2_prior, "from baseline (too few images to estimate)"
    elif tau2_prior is not None:
        tau2, tau2_source = tau2_prior, "from baseline (single image)"
    else:
        tau2, tau2_source = 0.0, "NOT ESTIMABLE from one image"

    ws = 1 / (v + tau2)
    mu = float((ws * phis).sum() / ws.sum())
    se = float(np.sqrt(1 / ws.sum()))
    if k > 1:
        hk = (ws * (phis - mu) ** 2).sum() / (k - 1) / ws.sum()
        se_ci, tcrit = max(se, float(np.sqrt(hk))), float(stats.t.ppf(0.975, k - 1))
    else:
        se_ci, tcrit = se, Z95
    pred = None
    if k >= 3:
        half = float(stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_ci ** 2))
        pred = [max(0.0, mu - half), min(1.0, mu + half)]

    norm = ws.sum() ** 2
    budget = {"area": float((ws ** 2 * var_area).sum() / norm),
              "regularity": float((ws ** 2 * var_irreg).sum() / norm),
              "between images": float(tau2 * (ws ** 2).sum() / norm)}
    return {
        "phi": mu, "se": se, "se_ci": se_ci,
        "ci95": [max(0.0, mu - tcrit * se_ci), min(1.0, mu + tcrit * se_ci)],
        "prediction_interval_new_field": pred, "tau": float(np.sqrt(tau2)), "tau2": float(tau2),
        "tau2_source": tau2_source, "I2": float(max(0.0, (q - (k - 1)) / q)) if k > 1 and q > 0 else 0.0,
        "n_images": k, "budget": budget, "bits": info_bits(se_ci),
    }


def next_measurement(var_area, var_irreg, tau2):
    """Same extra imaging area spent two ways: double every field, or add as many new fields.

    Only tau2 separates them: within-image variance falls with area either way, but
    field-to-field variance only averages down with more locations."""
    v = np.asarray(var_area) + np.asarray(var_irreg)

    def se(variances):
        return np.sqrt(1 / (1 / (variances + tau2)).sum())

    now = se(v)
    return {"bits_bigger_fields": float(np.log2(now / se(v / 2))),
            "bits_more_fields": float(np.log2(now / se(np.concatenate([v, v]))))}


def analyse_batch(results, tau2_prior=None):
    batch = {}
    for name in PHASES.values():
        ph = [r["phases"][name] for r in results]
        prior = None if tau2_prior is None else tau2_prior.get(name)
        re_ = random_effects([p["phi"] for p in ph], [p["var_area"] for p in ph],
                             [p["var_irregularity"] for p in ph], prior)
        re_["next"] = next_measurement([p["var_area"] for p in ph], [p["var_irregularity"] for p in ph], re_["tau2"])
        re_["threshold_range"] = [min(p["threshold_range"][0] for p in ph), max(p["threshold_range"][1] for p in ph)]
        re_["shading_shift_mean"] = float(np.mean([p["shading_shift"] for p in ph]))
        re_["systematic_range"] = [min(p["systematic_range"][0] for p in ph), max(p["systematic_range"][1] for p in ph)]
        re_["systematic_halfwidth"] = float(np.mean([(p["systematic_range"][1] - p["systematic_range"][0]) / 2
                                                     for p in ph]))
        re_["variant_shifts"] = np.mean([p["variant_shifts"] for p in ph], axis=0).tolist()
        re_["images"] = [{"image": r["image"], "phi": p["phi"], "se": p["se_image"]} for r, p in zip(results, ph)]
        batch[name] = re_
    return batch


# ---------------------------------------------------------------- lot A vs lot B

def parse_tolerances(text):
    """'pore=0.02,bright phase=0.01' -> {'pore': 0.02, 'bright phase': 0.01} (absolute phase fraction)."""
    tol = {}
    for part in filter(None, (s.strip() for s in (text or "").split(","))):
        name, value = part.split("=")
        tol[name.strip()] = float(value)
    return tol


def _lot_variance(phase, other):
    """Variance of a lot's mean phi. A lot whose field-to-field variation could not be
    estimated (one image) borrows the other lot's, if that one has it."""
    var = phase["se_ci"] ** 2
    unknown = "NOT ESTIMABLE" in phase["tau2_source"]
    if unknown and "NOT ESTIMABLE" not in other["tau2_source"]:
        return var + other["tau2"], False, True
    return var, unknown, False


def _cdf(x, nu):
    return stats.t.cdf(x, nu) if np.isfinite(nu) else stats.norm.cdf(x)


def _prob_range(prob, d, sys_h, n=41):
    """Lowest / highest prob(centre) as the segmentation bias runs over +-sys_h. A bias
    bound is not a distribution, so it gives a probability RANGE, not one number."""
    vals = [prob(d + bias) for bias in np.linspace(-sys_h, sys_h, n)]
    return [float(min(vals)), float(max(vals))]


def how_sure(p):
    """Plain words for a probability range [lo, hi]."""
    lo, hi = p
    if lo >= 0.95:
        return "very likely"
    if lo >= 0.80:
        return "likely"
    if hi <= 0.05:
        return "very unlikely"
    if hi <= 0.20:
        return "unlikely"
    return "uncertain"


def compare_phase(name, a, b, tol, imaging_flagged, scale_mismatch):
    """How sure are we that lot B differs from lot A for one phase, and by how much?

    delta = phi_B - phi_A. Its interval = t * sampling SE (random: area, regularity,
    field-to-field) + systematic half-width (segmentation, added linearly as a bias bound).
    Segmentation bias is mostly common-mode: nudging the thresholds moves both lots the
    same way. So delta is recomputed under every segmentation variant (threshold nudges,
    local thresholds) and only the change in DELTA counts. If imaging differs between
    lots that cancellation can't be trusted, and both lots' full ranges are combined.

    Two probabilities, each a range over the segmentation bias (flat prior on delta, so
    they are the confidence intervals read as probabilities):
      p_direction  the true change has the sign we observed (a real change, of any size)
      p_beyond     the true change is larger than +-tol (a change that matters)
    No decision is made here: the numbers say how sure we are, not what to do."""
    d = b["phi"] - a["phi"]
    va, unknown_a, borrowed_a = _lot_variance(a, b)
    vb, unknown_b, borrowed_b = _lot_variance(b, a)
    se = float(np.sqrt(va + vb))

    def dof(k):
        return k - 1 if k > 1 else np.inf

    denom = va ** 2 / dof(a["n_images"]) + vb ** 2 / dof(b["n_images"])
    nu = (va + vb) ** 2 / denom if denom > 0 else np.inf  # Welch-Satterthwaite
    t95 = float(stats.t.ppf(0.975, nu)) if np.isfinite(nu) else Z95
    t90 = float(stats.t.ppf(0.95, nu)) if np.isfinite(nu) else 1.644854

    if a.get("variant_shifts") and b.get("variant_shifts") and not imaging_flagged:
        sys_h = float(np.max(np.abs(np.subtract(b["variant_shifts"], a["variant_shifts"]))))
        sys_kind = "common-mode cancelled"
    else:
        ha = a.get("systematic_halfwidth", (a["systematic_range"][1] - a["systematic_range"][0]) / 2)
        hb = b.get("systematic_halfwidth", (b["systematic_range"][1] - b["systematic_range"][0]) / 2)
        sys_h, sys_kind = float(np.hypot(ha, hb)), "independent (imaging differs or old JSON)"
    h95, h90 = t95 * se + sys_h, t90 * se + sys_h

    sign = 1.0 if d >= 0 else -1.0
    p_direction = _prob_range(lambda c: _cdf(sign * c / se, nu), d, sys_h)
    p_beyond = _prob_range(lambda c: 1 - (_cdf((tol - c) / se, nu) - _cdf((-tol - c) / se, nu)), d, sys_h)
    # what the width of the delta interval is made of (95% half-widths)
    width = {"lot A sampling": float(t95 * np.sqrt(va)), "lot B sampling": float(t95 * np.sqrt(vb)),
             "segmentation": sys_h}

    caveats = []
    if borrowed_a or borrowed_b:
        caveats.append("one lot has a single field: borrowed the other lot's field-to-field variation")
    if unknown_a or unknown_b:
        caveats.append("field-to-field variation unknown (1 field per lot): probabilities are OVERSTATED, "
                       "a lot difference cannot be told apart from a location difference")
    if imaging_flagged:
        caveats.append("imaging conditions flagged in at least one image: part of delta may be imaging, not material")
    if scale_mismatch:
        caveats.append("lots analysed at different pixel sizes: these probabilities are not meaningful")

    settle = None
    if how_sure(p_beyond) not in ("very likely", "very unlikely") and not scale_mismatch:
        # sampling SE falls as 1/sqrt(imaging); how much imaging would push the interval to one side of tol?
        if abs(d) < tol:
            room, goal, t_used = tol - abs(d) - sys_h, "95% sure the change is within tolerance", t90
        else:
            room, goal, t_used = abs(d) - tol - sys_h, "95% sure the change exceeds tolerance", t95
        if room > 0:
            settle = (f"~{max((t_used * se / room) ** 2, 1.0):.1f}x more imaging per lot (area or locations) "
                      f"would make us {goal}, if the difference stays this size")
            if unknown_a or unknown_b:
                settle = "image >= 3 LOCATIONS per lot (more area alone cannot reveal location variation); " + settle
        else:
            settle = ("more imaging cannot make us sure: difference +- segmentation error straddles the tolerance; "
                      "only better segmentation can")

    return {"phase": name, "phi_a": a["phi"], "phi_b": b["phi"], "delta": float(d),
            "delta_rel": float(d / a["phi"]) if a["phi"] > 0 else None, "tolerance": float(tol),
            "se_sampling": se, "dof": float(nu), "systematic_halfwidth": sys_h, "systematic_kind": sys_kind,
            "interval95": [float(d - h95), float(d + h95)], "interval90": [float(d - h90), float(d + h90)],
            "p_direction": p_direction, "p_beyond_tolerance": p_beyond,
            "how_sure_real_change": how_sure(p_direction), "how_sure_beyond_tolerance": how_sure(p_beyond),
            "interval_width": width, "caveats": caveats, "to_settle": settle,
            "n_images": [a["n_images"], b["n_images"]]}


def compare_batch_summaries(a, b, name_a, name_b, tol_abs, tol_rel):
    detector_a, detector_b = a.get("detector"), b.get("detector")
    if detector_a and detector_b and detector_a.lower() != detector_b.lower():
        raise ValueError("cannot compare batches acquired with different detectors")
    scale_a, scale_b = a.get("target_pixel_nm"), b.get("target_pixel_nm")
    scale_mismatch = bool(scale_a and scale_b and abs(scale_a / scale_b - 1) > 0.01) or (bool(scale_a) != bool(scale_b))
    flagged = []
    for name, lot, other in ((name_a, a, b), (name_b, b, a)):
        reference = [im["quality"] for im in other.get("images", []) if "quality" in im]
        for im in lot.get("images", []):
            if im.get("imaging_flags") or quality_flags(im.get("quality", {}), reference):
                flagged.append(f"{name}/{im['image']}")
    rows = []
    for name, pa in a["phases"].items():
        if name not in b["phases"]:
            continue
        tol = tol_abs.get(name, tol_rel * pa["phi"])
        rows.append(compare_phase(name, pa, b["phases"][name], tol, bool(flagged), scale_mismatch))
    if not rows:
        raise ValueError("the batches have no phase measurements in common")
    strongest = max(rows, key=lambda r: (r["p_beyond_tolerance"][0], r["p_direction"][0]))
    return {"lot_a": str(name_a), "lot_b": str(name_b), "scale_nm": [scale_a, scale_b],
            "scale_mismatch": scale_mismatch, "flagged_images": flagged,
            "strongest_evidence": strongest["phase"], "phases": rows}


def compare_lots(path_a, path_b, tol_abs, tol_rel):
    lots = []
    for path in (path_a, path_b):
        data = json.loads(Path(path).read_text())
        lots.append(data if "phases" in data else {"phases": data})
    return compare_batch_summaries(*lots, path_a, path_b, tol_abs, tol_rel)


def print_comparison(c, name_a, name_b, tol_note):
    print(f"\n=== {name_b} vs {name_a} (baseline) ===")
    sa, sb = c["scale_nm"]
    fmt = lambda s: f"{s:.2f}" if s else "unknown"
    print(f"  scale: {fmt(sa)} vs {fmt(sb)} nm/px" + ("  -> MISMATCH, not comparable" if c["scale_mismatch"] else ""))
    print(f"  tolerances: {tol_note}")
    if c["flagged_images"]:
        print(f"  imaging flagged in: {', '.join(c['flagged_images'])}")
    for r in c["phases"]:
        rel = f" ({100 * r['delta_rel']:+.0f}%)" if r["delta_rel"] is not None else ""
        lo, hi = r["interval95"]
        print(f"\n  {r['phase']:13s} {name_a} {r['phi_a']:.4f} -> {name_b} {r['phi_b']:.4f}:  "
              f"delta {r['delta']:+.4f}{rel}, 95% interval [{lo:+.4f}, {hi:+.4f}], tolerance +-{r['tolerance']:.4f}")
        t95 = float(stats.t.ppf(0.975, r["dof"])) if np.isfinite(r["dof"]) else Z95
        dof_text = f"t, {r['dof']:.1f} dof" if np.isfinite(r["dof"]) else "z"
        print(f"  {'':13s} interval = sampling +-{t95 * r['se_sampling']:.4f} ({dof_text})"
              f" (images {r['n_images'][0]} vs {r['n_images'][1]}) + systematic +-{r['systematic_halfwidth']:.4f}"
              f" ({r['systematic_kind']})")
        w = r["interval_width"]
        print(f"  {'':13s} width from: " + ", ".join(f"{k} +-{v:.4f}" for k, v in w.items())
              + f"  (largest: {max(w, key=w.get)})")
        pd_, pb = r["p_direction"], r["p_beyond_tolerance"]
        print(f"  {'':13s} real change ({'up' if r['delta'] >= 0 else 'down'}):"
              f"  P = {pd_[0]:.2f}-{pd_[1]:.2f}  -> {r['how_sure_real_change']}")
        print(f"  {'':13s} change beyond +-{r['tolerance']:.4f}:  P = {pb[0]:.2f}-{pb[1]:.2f}"
              f"  -> {r['how_sure_beyond_tolerance']}")
        for cav in r["caveats"]:
            print(f"  {'':13s}    note: {cav}")
        if r["to_settle"]:
            print(f"  {'':13s}    to be sure: {r['to_settle']}")
    s = next(r for r in c["phases"] if r["phase"] == c["strongest_evidence"])
    print(f"\n  strongest evidence of a change beyond tolerance: {s['phase']} "
          f"(P = {s['p_beyond_tolerance'][0]:.2f}-{s['p_beyond_tolerance'][1]:.2f})")
    print("  P ranges span the segmentation bias bound; they say how sure we are, not what to decide.")


# ---------------------------------------------------------------- text output

def print_image_summary(result):
    px, target = result["pixel_size_nm_used"], result["target_pixel_nm"]
    h, w = result["shape_px"]
    ah, aw = result["shape_analysed"]
    px_text = f"{px:.2f} nm/px" if px else "pixel size unknown"
    scale_text = f"analysed at {target:.2f} nm/px" if target else "analysed in binned px"
    bar_text = f", {result['databar_rows']} databar rows cropped" if result["databar_rows"] else ""
    print(f"\n=== {result['image']}  ({w} x {h} px, {px_text}{bar_text}; {scale_text} -> {aw} x {ah}) ===")
    q = result["quality"]
    print(f"  imaging: sharpness {q['sharpness']:.3f}, x/y blur ratio {q['anisotropy']:.2f}, noise {q['noise']:.3f}, "
          f"separability {q['separability']:.2f}, "
          f"clipped black/white {100 * q['clipped_black']:.1f}/{100 * q['clipped_white']:.1f}%, "
          f"curtaining {q['stripes_curtaining']:.2f} ({q['stripes_curtaining_angle']:+.0f} deg), "
          f"scan lines {q['stripes_scan_lines']:.2f}")
    if result["preprocessing"]:
        print(f"  preprocessing: {'; '.join(result['preprocessing'])}")
    n_refit, n_tiles = result["local_threshold_tiles"]
    print(f"  local thresholds re-fitted in {n_refit}/{n_tiles} tiles")
    for name, r in result["phases"].items():
        d = r["dispersion"]
        reg = (f"D={d['D']:.2f} [{d['D_90'][0]:.2f}-{d['D_90'][1]:.2f}]" if d else "D=n/a (image too small)")
        trend = r["depth_trend"]
        tr = (f"top->bottom {trend['change_top_to_bottom']:+.3f} +- {Z95 * trend['se']:.3f}" if trend else "")
        print(f"  {name:13s} phi={r['phi']:.4f}  +-{Z95 * r['se_image']:.4f} (95%)  "
              f"N_eff~{r['n_eff']:,.0f}  regularity {reg}  {tr}")
        if d and d["D_90"][0] > 1:
            print(f"  {'':13s} -> image is NOT statistically regular for {name}: error bar inflated x{np.sqrt(d['D']):.2f}")
        if trend and abs(trend["change_top_to_bottom"]) > Z95 * trend["se"]:
            print(f"  {'':13s} -> significant through-thickness gradient (if image y is the electrode depth)")
        lo, hi = r["threshold_range"]
        print(f"  {'':13s} systematic: threshold {lo:.4f}-{hi:.4f}, local shading/roughness shift {r['shading_shift']:+.4f}")
        if r["resolution_limited"]:
            print(f"  {'':13s} -> RESOLUTION-LIMITED: features < 5 px at this scale, phi biased by edge pixels")


def print_quality_report(results, reference):
    print(f"\n=== imaging conditions vs {reference} ===")
    for r in results:
        flags = r["imaging_flags"]
        if flags:
            print(f"  {r['image']}: imaging differs from reference")
            for f in flags:
                print(f"      {f}")
        else:
            print(f"  {r['image']}: consistent")
    if any(r["imaging_flags"] for r in results):
        print("  -> differences in flagged images may come from imaging, not material: check these first.")


def print_batch_summary(batch):
    print("\n=== batch: pooled over fields of view (random effects) ===")
    for name, b in batch.items():
        bud = b["budget"]
        total = sum(bud.values())
        shares = ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in bud.items())
        pi = b["prediction_interval_new_field"]
        print(f"  {name:13s} phi={b['phi']:.4f}  95% CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]"
              f"  ({b['n_images']} images, {b['bits']:.1f} bits)")
        print(f"  {'':13s} field-to-field tau={b['tau']:.4f} ({b['tau2_source']}), I2={100 * b['I2']:.0f}%"
              + (f", a new field should land in [{pi[0]:.4f}, {pi[1]:.4f}]" if pi else ""))
        print(f"  {'':13s} uncertainty budget: {shares}")
        nx = b["next"]
        gain = nx["bits_more_fields"] - nx["bits_bigger_fields"]
        if gain > 0.05:
            best = f"MORE LOCATIONS beat bigger fields (+{nx['bits_more_fields']:.2f} vs +{nx['bits_bigger_fields']:.2f} bits)"
        else:
            best = f"bigger fields as good as more locations (+{nx['bits_bigger_fields']:.2f} bits), and cheaper"
        print(f"  {'':13s} next {b['n_images']}-fields' worth of imaging: {best}")
        lo, hi = b["systematic_range"]
        print(f"  {'':13s} systematic range {lo:.4f}-{hi:.4f} (threshold + shading/roughness, mean shading shift "
              f"{b['shading_shift_mean']:+.4f}; not reduced by more imaging)")
        if b["n_images"] < 3 and "baseline" not in b["tau2_source"]:
            print(f"  {'':13s} WARNING: <3 images, field-to-field variation is unknown or poorly known; CI is optimistic.")


# ---------------------------------------------------------------- figures

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
BUDGET_COLORS = {"area": "#2a78d6", "regularity": "#eb6834", "between images": "#1baf7a"}


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    return plt


def plot_image_report(labels, result, unit_len, out_png):
    plt = _style()
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(14, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    step = max(1, max(labels.shape) // 2000)
    cmap = ListedColormap([PHASE_COLORS[k] for k in sorted(PHASES)])
    ax.imshow(labels[::step, ::step], cmap=cmap, interpolation="nearest", vmin=-0.5, vmax=len(PHASES) - 0.5)
    ax.set_title(f"Segmentation: {result['image']}  (check this before trusting any number below)",
                 loc="left", color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for k, name in PHASES.items():
        ax.plot([], [], "s", color=PHASE_COLORS[k], markersize=10,
                label=f"{name}  phi={result['phases'][name]['phi']:.3f}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)

    full_side = np.sqrt(labels.size) * unit_len
    for i, (k, name) in enumerate(PHASES.items()):
        r = result["phases"][name]
        ax = fig.add_subplot(gs[1, i], facecolor=SURFACE)
        c = PHASE_COLORS[k]
        side = np.array([t["side"] for t in r["tiles"]]) * unit_len
        ax.plot(side, [t["naive_se"] for t in r["tiles"]], "--", color=INK2, lw=1.5, label="naive (pixels independent)")
        ax.plot(side, [t["predicted_se"] for t in r["tiles"]], "-", color=c, lw=2, label="predicted from C(r)")
        ax.plot(np.array([t["side"] for t in r["extrapolated"]]) * unit_len,
                [t["predicted_se"] for t in r["extrapolated"]], ":", color=c, lw=2, label="extrapolated (no tile check)")
        ax.errorbar(side, [t["observed_se"] for t in r["tiles"]], yerr=[t["observed_se_err"] for t in r["tiles"]],
                    fmt="o", color=c, mfc=SURFACE, mew=2, ms=8, elinewidth=1.5, capsize=0,
                    label="observed spread across tiles")
        ax.axvline(full_side, color=INK2, lw=1, ls=":")
        ax.plot([full_side], [r["se_image"]], "o", color=c, ms=8)
        ax.annotate(f"full image\n+-{Z95 * r['se_image']:.4f}", (full_side, r["se_image"]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", color=INK2, fontsize=9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_xlabel(f"window side length ({result['unit']})")
        if i == 0:
            ax.set_ylabel("standard error of phase fraction")
            ax.legend(loc="lower left", frameon=False, fontsize=9)
        d = r["dispersion"]
        ax.set_title(f"{name}:  N_eff ~ {r['n_eff']:,.0f},  D = {d['D']:.2f}" if d else name, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.savefig(out_png, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_batch_report(batch, out_png):
    """Forest plot per phase (each field, pooled CI, next-field interval) + uncertainty budget."""
    plt = _style()
    names = list(batch)
    n_img = batch[names[0]]["n_images"]
    fig = plt.figure(figsize=(14, 4.5 + 0.35 * n_img), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[max(2.5, 0.35 * n_img + 1), 1.4], hspace=0.55, wspace=0.12)

    for i, name in enumerate(names):
        b = batch[name]
        ax = fig.add_subplot(gs[0, i], facecolor=SURFACE)
        ax.set_axisbelow(True)
        ys = np.arange(n_img, 0, -1) + 1
        for y, im in zip(ys, b["images"]):
            ax.errorbar(im["phi"], y, xerr=Z95 * im["se"], fmt="o", color=INK2, ms=7, elinewidth=2, capsize=0)
        lo, hi = b["ci95"]
        ax.fill([lo, b["phi"], hi, b["phi"]], [1, 1.25, 1, 0.75], color=INK, lw=0)
        ticks, ticklabels = list(ys) + [1], [im["image"] for im in b["images"]] + ["batch mean (95% CI)"]
        if b["prediction_interval_new_field"]:
            plo, phi_ = b["prediction_interval_new_field"]
            ax.errorbar((plo + phi_) / 2, 0, xerr=(phi_ - plo) / 2, fmt="none", color=INK2, elinewidth=1.5,
                        capsize=4, ls="none")
            ticks.append(0)
            ticklabels.append("next field (95%)")
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels if i == 0 else [""] * len(ticks))
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, n_img + 1.7)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("phase fraction")
        ax.set_title(f"{name}:  tau = {b['tau']:.4f},  I2 = {100 * b['I2']:.0f}%", loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)

    ax = fig.add_subplot(gs[1, :], facecolor=SURFACE)
    for row, name in enumerate(names):
        bud = batch[name]["budget"]
        total = sum(bud.values())
        left = 0.0
        for comp, val in bud.items():
            share = val / total
            ax.barh(row, share, left=left, color=BUDGET_COLORS[comp], edgecolor=SURFACE, linewidth=2, height=0.6,
                    label=comp if row == 0 else None)
            if share > 0.06:
                ax.text(left + share / 2, row, f"{100 * share:.0f}%", ha="center", va="center", color=SURFACE, fontsize=9)
            left += share
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("What drives the batch error bar (share of variance)", loc="left", color=INK)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, ncol=3, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.savefig(out_png, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- CLI

DETECTOR_TAGS = ("bse", "etd", "inlens", "se", "tld", "cbs")


def _detector_of(path):
    """Detector named in the filename, e.g. img_0grcilhi_BSE.tif -> 'bse' (None if untagged)."""
    parts = re.split(r"[_\-. ]", Path(path).stem.lower())
    return next((t for t in DETECTOR_TAGS if t in parts), None)


def collect_images(inputs, detector="bse"):
    """Images to treat as separate fields. In a folder, files tagged with another detector
    are skipped: BSE / ETD / InLens of one spot are one field, not three."""
    paths, skipped = [], []
    for p in map(Path, inputs):
        if p.is_dir():
            for q in sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES):
                tag = _detector_of(q)
                (skipped if tag and detector != "all" and tag != detector else paths).append(q)
        else:
            paths.append(p)
    if skipped:
        print(f"using {detector.upper()} images only; skipped: {', '.join(q.name for q in skipped)}")
    return paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_DETECTORS = ("bse", "etd", "inlens")


def parse_image_identity(path):
    match = re.fullmatch(r"img_(.+)_(BSE|ETD|InLens)", Path(path).stem, re.IGNORECASE)
    if not match or not match.group(1).strip():
        raise ValueError(f"{Path(path).name}: expected img_batteryname_filter with filter BSE, ETD or Inlens")
    return match.group(1), match.group(2).lower()


def collect_batch_groups(inputs, detector="all", excluded_images=None):
    detector = detector.lower()
    if detector not in (*BATCH_DETECTORS, "all"):
        raise ValueError("detector must be BSE, ETD, Inlens or all")
    folders = {}
    for raw in inputs:
        root = Path(raw).resolve()
        if not root.is_dir():
            raise ValueError(f"{raw}: expected a Batch_* folder or its parent directory")
        candidates = [root] if root.name.lower().startswith("batch_") else sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.lower().startswith("batch_"))
        if not candidates:
            raise ValueError(f"{root}: no Batch_* folders found")
        for folder in candidates:
            key = folder.name.casefold()
            if key in folders and folders[key] != folder:
                raise ValueError(f"different directories have the same batch name: {folders[key]} and {folder}")
            folders[key] = folder
    if len(folders) < 2:
        raise ValueError("all-batch comparison requires at least two Batch_* folders")
    groups = {}
    for folder in sorted(folders.values(), key=lambda path: path.name):
        groups[folder.name] = {}
        seen = set()
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if re.fullmatch(r"img_.+_SE", path.stem, re.IGNORECASE):
                if excluded_images is not None:
                    excluded_images.append({"batch": folder.name, "image": path.name,
                                            "reason": "SE detector excluded from BSE/ETD/Inlens comparisons"})
                continue
            battery, tag = parse_image_identity(path)
            if detector != "all" and tag != detector:
                continue
            identity = (battery.casefold(), tag)
            if identity in seen:
                raise ValueError(f"{folder.name}: duplicate image for battery {battery} and detector {tag}")
            seen.add(identity)
            groups[folder.name].setdefault(tag, []).append(path)
    return groups


def analyse_batch_files(paths, metas, target_nm, target_note, args, baseline=None, batch_name=None, detector=None,
                        write_reports=True):
    out = Path(args.out)
    if write_reports:
        out.mkdir(parents=True, exist_ok=True)
    print(f"scale: {f'{target_nm:.2f} nm/px' if target_nm else 'pixels'} ({target_note})")
    results = [analyse_image(path, meta, target_nm, args) for path, meta in zip(paths, metas)]

    ref_images = (baseline or {}).get("images")
    for i, r in enumerate(results):
        ref = [b["quality"] for b in ref_images] if ref_images else \
              [o["quality"] for j, o in enumerate(results) if j != i]
        flags = quality_flags(r["quality"], ref if (ref_images or len(ref) >= 2) else [])
        if r["upsampled"]:
            flags.append("image is coarser than the analysis scale (upsampled): fine features may be lost")
        flags += [f"{name} resolution-limited (features < 5 px)"
                  for name, ph in r["phases"].items() if ph["resolution_limited"]]
        r["imaging_flags"] = flags
        if write_reports:
            (out / f"{Path(r['image']).stem}_uncertainty.json").write_text(json.dumps(r, indent=2))
    reference = "baseline images" if ref_images else (
        "the rest of this batch" if len(results) >= 3 else "absolute limits only (need >= 3 images or --baseline)")
    print_quality_report(results, reference)

    tau2_prior = None
    if baseline:
        tau2_prior = {name: b["tau2"] for name, b in baseline["phases"].items() if "estimated" in b["tau2_source"]}
    batch = analyse_batch(results, tau2_prior)
    print_batch_summary(batch)
    if write_reports and not args.no_plots:
        plot_batch_report(batch, out / "batch_uncertainty.png")
    summary = {"target_pixel_nm": target_nm, "scale_note": target_note,
               "images": [{"image": r["image"], "quality": r["quality"], "imaging_flags": r["imaging_flags"]}
                          for r in results],
               "phases": batch}
    if batch_name is not None:
        summary.update(batch=batch_name, detector=detector, input_files=[str(path) for path in paths],
                       phase_labels_validated=False)
    if write_reports:
        (out / "batch_uncertainty.json").write_text(json.dumps(summary, indent=2))
    return summary


def compare_all_batches(args):
    if args.baseline:
        raise ValueError("--baseline cannot be combined with --compare-all; scales are matched across all batches")
    excluded_images = []
    groups = collect_batch_groups(args.inputs, args.detector or "all", excluded_images=excluded_images)
    tol_abs = parse_tolerances(args.tol)
    if any(not np.isfinite(value) or value < 0 for value in [args.tol_rel, *tol_abs.values()]):
        raise ValueError("tolerances must be finite and non-negative")
    if set(tol_abs) - set(PHASES.values()):
        raise ValueError("unknown tolerance phase; expected " + ", ".join(PHASES.values()))
    plans = {}
    for detector in BATCH_DETECTORS:
        selected = {batch: images[detector] for batch, images in groups.items() if detector in images}
        if not selected:
            continue
        metadata = {batch: [read_meta(path, args.page) for path in paths] for batch, paths in selected.items()}
        target, note = resolve_target([meta for values in metadata.values() for meta in values], args, None)
        if target is None or not np.isfinite(target) or target <= 0:
            raise ValueError(f"{detector}: comparison requires a physical pixel size; supply metadata or --px-size")
        plans[detector] = (selected, metadata, target, note)
    if not any(len(selected) >= 2 for selected, _, _, _ in plans.values()):
        raise ValueError("no detector has images in at least two batches")

    report = {"batches": list(groups), "detectors": {}, "excluded_images": excluded_images, "caveats": [
        "Batch folders define battery types; differences describe these batches, not independently replicated type effects.",
        "Pairwise intervals and probability ranges are not adjusted for multiple comparisons; they are exploratory, not an omnibus test.",
        "Images within a batch/detector must be independent samples, not duplicate views of the same battery or location.",
        "Phase labels use placeholder intensity segmentation and require validation against the actual material.",
        "Pairs use alphabetical batch order; delta is batch B minus batch A and relative tolerance uses batch A as reference.",
    ]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for caveat in report["caveats"]:
        print(f"NOTE: {caveat}")
    tol_note = (", ".join(f"{key} +-{value}" for key, value in tol_abs.items()) + "; " if tol_abs else "")
    tol_note += f"others +-{100 * args.tol_rel:.0f}% of the reference batch (set real limits with --tol)"
    for detector, (selected, metadata, target, note) in plans.items():
        data = {"target_pixel_nm": target, "missing_batches": [name for name in groups if name not in selected],
                "batches": {}, "comparisons": [], "caveats": []}
        if detector != "bse":
            data["caveats"].append(
                "BSE-oriented segmentation is used on this detector: pore/graphite/bright phase are intensity-class labels, not verified material identities.")
        if data["missing_batches"]:
            data["caveats"].append("No images for " + ", ".join(data["missing_batches"]) + "; their pairs are omitted.")
        if len(selected) < 2:
            data["caveats"].append("Fewer than two batches have this detector; no pairwise comparison is possible.")
        for caveat in data["caveats"]:
            print(f"NOTE ({detector}): {caveat}")
        for batch, paths in selected.items():
            print(f"\n=== {detector.upper()} / {batch} ({len(paths)} images) ===")
            local_args = argparse.Namespace(**{**vars(args), "no_plots": True})
            data["batches"][batch] = analyse_batch_files(
                paths, metadata[batch], target, note, local_args,
                batch_name=batch, detector=detector, write_reports=False)
        for name_a, name_b in combinations(data["batches"], 2):
            comparison = compare_batch_summaries(
                data["batches"][name_a], data["batches"][name_b], name_a, name_b, tol_abs, args.tol_rel)
            data["comparisons"].append(comparison)
            print(f"\nDetector: {detector.upper()}")
            print_comparison(comparison, name_a, name_b, tol_note)
        report["detectors"][detector] = data
    output_path = out / "all_batch_comparisons.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comparison JSON to {output_path.resolve()}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compare batch phase fractions. With no inputs, analyse the project's Batches folder and write all_batch_comparisons.json.")
    p.add_argument("inputs", nargs="*", help="images for one batch, or with --compare-all: Batch_* folders or their parent")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--compare", nargs=2, metavar=("LOT_A_JSON", "LOT_B_JSON"), default=None,
                      help="compare two batch_uncertainty.json files (A = baseline / approved lot)")
    mode.add_argument("--compare-all", action="store_true",
                      help="analyse Batch_* folders and compare every batch pair separately for each detector")
    p.add_argument("--tol", default=None,
                   help="absolute tolerances on phi per phase, e.g. \"pore=0.02,bright phase=0.01\"")
    p.add_argument("--tol-rel", type=float, default=0.10,
                   help="tolerance as a fraction of lot A's phi, for phases without --tol (default 0.10)")
    p.add_argument("--bin", type=int, default=None,
                   help="target pixel = coarsest image pixel x bin (default 1); without pixel sizes, plain binning (default 2)")
    p.add_argument("--target-px", type=float, default=None, help="analyse at this pixel size, nm (overrides baseline)")
    p.add_argument("--px-size", type=float, default=None, help="override pixel size of the ORIGINAL images, nm")
    p.add_argument("--tilt-deg", type=float, default=None,
                   help="stage tilt of a FIB cross-section if the microscope did not tilt-correct (e.g. 52)")
    p.add_argument("--crop-bottom", type=int, default=None, help="override databar rows to crop")
    p.add_argument("--page", type=int, default=0, help="page of a multi-page TIF")
    p.add_argument("--max-lag", type=int, default=None, help="max correlation lag in analysed px")
    p.add_argument("--no-flatten", action="store_true", help="skip shading flattening")
    p.add_argument("--no-destripe", action="store_true", help="skip curtaining / scan-line removal")
    p.add_argument("--baseline", default=None,
                   help="batch JSON from a baseline run: its scale, imaging conditions and tau2 are the reference")
    p.add_argument("--detector", default=None, type=str.lower, choices=(*DETECTOR_TAGS, "all"),
                   help="detector filter: defaults to all with --compare-all, otherwise BSE")
    p.add_argument("--no-plots", action="store_true", help="write numerical reports without PNG plots")
    p.add_argument("--out", default=None,
                   help="output directory; defaults to the project root for batch comparisons, otherwise ./out")
    args = p.parse_args(argv)

    if args.compare_all or (not args.inputs and not args.compare):
        args.inputs = args.inputs or [str(PROJECT_ROOT / "Batches")]
        args.out = args.out or str(PROJECT_ROOT)
        try:
            compare_all_batches(args)
        except ValueError as exc:
            p.error(str(exc))
        return
    args.detector = args.detector or "bse"
    args.out = args.out or "out"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.compare:
        tol_abs = parse_tolerances(args.tol)
        result = compare_lots(*args.compare, tol_abs, args.tol_rel)
        names = [Path(f).parent.name or Path(f).stem for f in args.compare]
        tol_note = ", ".join(f"{k} +-{v}" for k, v in tol_abs.items())
        tol_note = (tol_note + "; " if tol_note else "") + f"others +-{100 * args.tol_rel:.0f}% of lot A " \
                   "(set real limits from the material spec with --tol)"
        print_comparison(result, names[0], names[1], tol_note)
        (out / "comparison.json").write_text(json.dumps(result, indent=2))
        return

    paths = collect_images(args.inputs, args.detector.lower())
    if not paths:
        p.error(f"no images found (in folders, only files tagged {args.detector} or untagged are used)")
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    if baseline and "phases" not in baseline:  # older batch JSON: phases at top level
        baseline = {"phases": baseline}

    metas = [read_meta(path, args.page) for path in paths]
    try:
        target_nm, target_note = resolve_target(metas, args, baseline)
    except ValueError as e:
        p.error(str(e))
    analyse_batch_files(paths, metas, target_nm, target_note, args, baseline)


if __name__ == "__main__":
    main()
