"""Stage 5b: SiOx particle instances (2D section sizes).

SiOx mask -> fill holes -> Euclidean distance transform -> h-maxima markers (h = 3 px) -> watershed.
Per particle: area, equivalent diameter, aspect ratio, centroid, whether it touches an excluded region.

Writes outputs/segmentation/<batch>/<sid>/siox_instances.png (uint16), outputs/metrics/particle_sizes.csv,
outputs/metrics/siox_summary.csv.

Usage: python -m src.instances
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from . import config as C

H = 3
MIN_AREA_UM2 = 0.1


def run(key):
    batch, sid = key
    lab = np.load(C.proc_dir(sid) / "final_labels.npy")
    excl = lab == C.EXCLUDED
    mask = ndi.binary_fill_holes(lab == C.SIOX)
    dist = ndi.distance_transform_edt(mask)
    markers, _ = ndi.label(h_maxima(dist, H))
    inst = watershed(-dist, markers, mask=mask).astype(np.uint16)
    near_excl = ndi.binary_dilation(excl, iterations=2)
    rows = []
    for p in regionprops(inst):
        area = p.area * C.PX_UM2
        if area < MIN_AREA_UM2:
            inst[inst == p.label] = 0
            continue
        rows.append({"batch": batch, "sample_id": sid, "particle": p.label, "area_um2": round(area, 4),
                     "eq_diam_um": round(2 * np.sqrt(area / np.pi), 3),
                     "aspect_ratio": round(p.major_axis_length / max(p.minor_axis_length, 1e-6), 3),
                     "centroid_y_um": round(p.centroid[0] * C.NM_PER_PX / 1000, 2),
                     "centroid_x_um": round(p.centroid[1] * C.NM_PER_PX / 1000, 2),
                     "touches_excluded": bool(near_excl[inst == p.label].any())})
    Image.fromarray(inst).save(C.OUT_SEG / batch / sid / "siox_instances.png")
    return rows


def main():
    with ProcessPoolExecutor(C.WORKERS) as ex:
        rows = [r for rs in ex.map(run, C.sample_ids()) for r in rs]
    C.write_csv(C.OUT_MET / "particle_sizes.csv", rows)
    summary = []
    valid_area = {}
    for batch, sid in C.sample_ids():
        lab = np.load(C.proc_dir(sid) / "final_labels.npy")
        valid_area[sid] = (lab != C.EXCLUDED).sum() * C.PX_UM2
        d = np.array([r["eq_diam_um"] for r in rows if r["sample_id"] == sid])
        big = d[d >= 2 * np.sqrt(2 / np.pi)]          # >= 2 um^2, comparable to the earlier threshold filter
        summary.append({"batch": batch, "sample_id": sid, "n_particles": len(d), "n_particles_ge_2um2": len(big),
                        "per_1000um2": round(1000 * len(d) / valid_area[sid], 2),
                        "eq_diam_median_um": round(float(np.median(d)), 3) if len(d) else "",
                        "eq_diam_p90_um": round(float(np.percentile(d, 90)), 3) if len(d) else "",
                        "eq_diam_median_ge_2um2_um": round(float(np.median(big)), 3) if len(big) else ""})
    C.write_csv(C.OUT_MET / "siox_summary.csv", summary)
    print(f"{len(rows)} SiOx particles across {len(summary)} samples; "
          f"median eq. diameter {np.median([r['eq_diam_um'] for r in rows]):.2f} um")


if __name__ == "__main__":
    main()
