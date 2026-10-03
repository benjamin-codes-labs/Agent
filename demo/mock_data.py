"""Two mock batteries that exercise opposite paths through the workflow.

BAT-21  a clean medium-confidence case: all three detectors, both branches
        agreeing. Used to show the repair loop fixing a flawed first draft.

BAT-42  the nasty case: a missing detector, the two branches disagreeing, the
        unlike-any-training-type flag raised, and a low confidence label. Used
        to show the CONF* rules forcing disclosure, and the template fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentinel import (
    BatteryResult,
    Classification,
    DetectorMaps,
    ImageRecord,
    MapRegion,
    PhaseFractions,
    Profile,
    StatisticRecord,
    StatsClassifier,
    TypeProbs,
    Versions,
)

VERSIONS = Versions(
    git="9f3c1ab7e5d2408bb16c0f7a2d9e4c81aa37b250",
    segmenters={"BSE": "xgb_bse_v1", "ETD": "xgb_etd_v1", "InLens": "xgb_inlens_v1"},
    backbone="dinov3_vitb16",
    heads="lr_v1",
    llm="claude-sonnet-5-5",
)


def _stat(sid, detector, name, label, value, unit, means, sds, ns, zs, rank, cliffs=None):
    return StatisticRecord(
        id=sid, detector=detector, name=name, label=label, value=value, unit=unit,
        profiles={t: Profile(mean=means[t], sd=sds[t], n=ns[t]) for t in "ABC"},
        z={t: zs[t] for t in "ABC"},
        separates_pred_vs_runner_up_rank=rank,
        cliffs_delta=cliffs,
        primary_detector=detector,
    )


# --------------------------------------------------------------------------- #
# BAT-21 -- the clean case
# --------------------------------------------------------------------------- #
def bat_21() -> BatteryResult:
    return BatteryResult(
        battery_id="BAT-21",
        run_id="2026-10-04T09:14:02Z",
        scope="v1",
        versions=VERSIONS,
        images=[
            ImageRecord(detector="BSE", file="BAT-21_BSE.tif", pixel_size_um=0.0488,
                        pixel_size_source="fei_metadata", kv=5.0, masked_area_frac=0.018),
            ImageRecord(detector="ETD", file="BAT-21_ETD.tif", pixel_size_um=0.0488,
                        pixel_size_source="fei_metadata", kv=5.0,
                        quality_flags=["sigma_edge_high"], masked_area_frac=0.041),
            ImageRecord(detector="InLens", file="BAT-21_InLens.tif", pixel_size_um=0.0488,
                        pixel_size_source="fei_metadata", kv=2.0, masked_area_frac=0.012),
        ],
        classification=Classification(
            per_detector={
                "BSE": TypeProbs(A=0.09, B=0.04, C=0.87),
                "ETD": TypeProbs(A=0.22, B=0.11, C=0.67),
                "InLens": TypeProbs(A=0.17, B=0.09, C=0.74),
            },
            fused=TypeProbs(A=0.16, B=0.08, C=0.76),
            predicted="C",
            runner_up="A",
            detector_agreement=3,
            stats_classifier=StatsClassifier(
                predicted="C", probs=TypeProbs(A=0.21, B=0.07, C=0.72)
            ),
            unlike_any_type=False,
            confidence="medium",
            confidence_reasons=[
                "the ETD image carries a sharpness flag, so its contribution is "
                "down-weighted"
            ],
        ),
        statistics=[
            _stat("stats.BSE.porosity", "BSE", "porosity", "porosity",
                  0.313, "fraction",
                  {"A": 0.352, "B": 0.264, "C": 0.308},
                  {"A": 0.021, "B": 0.018, "C": 0.034},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": -1.857, "B": 2.722, "C": 0.147}, 2, cliffs=-0.41),
            _stat("stats.BSE.d50", "BSE", "d50",
                  "median cross-sectional particle diameter",
                  7.86, "um",
                  {"A": 9.81, "B": 6.15, "C": 7.92},
                  {"A": 1.12, "B": 0.74, "C": 1.41},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": -1.741, "B": 2.311, "C": -0.043}, 1, cliffs=-0.73),
            _stat("stats.BSE.span", "BSE", "span", "particle size span",
                  1.42, "dimensionless",
                  {"A": 0.98, "B": 1.11, "C": 1.39},
                  {"A": 0.14, "B": 0.12, "C": 0.16},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": 3.143, "B": 2.583, "C": 0.188}, 3, cliffs=0.68),
            _stat("stats.InLens.cbd_fraction", "InLens", "cbd_fraction",
                  "carbon-binder domain fraction",
                  0.149, "fraction",
                  {"A": 0.131, "B": 0.179, "C": 0.152},
                  {"A": 0.015, "B": 0.014, "C": 0.022},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": 1.200, "B": -2.143, "C": -0.136}, 4, cliffs=0.33),
            _stat("stats.ETD.crack_density", "ETD", "crack_density", "crack density",
                  0.029, "um^-1",
                  {"A": 0.018, "B": 0.039, "C": 0.026},
                  {"A": 0.009, "B": 0.012, "C": 0.014},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": 1.222, "B": -0.833, "C": 0.214}, 5, cliffs=0.22),
        ],
        maps={
            "BSE": DetectorMaps(
                overview_png="runs/R2/BAT-21/BSE_overview.png",
                crops_png="runs/R2/BAT-21/BSE_crops.png",
                regions=[
                    MapRegion(id=1, box_um=[8.1, 22.4, 26.3, 39.8], area_um2=316.2,
                              share_of_evidence=0.34,
                              phase_fractions=PhaseFractions(pore=0.341, cbd=0.142, am=0.517),
                              minus_whole_image=PhaseFractions(pore=0.028, cbd=-0.007, am=-0.021)),
                    MapRegion(id=2, box_um=[48.7, 55.0, 61.2, 66.4], area_um2=140.0,
                              share_of_evidence=0.21,
                              phase_fractions=PhaseFractions(pore=0.329, cbd=0.151, am=0.520),
                              minus_whole_image=PhaseFractions(pore=0.016, cbd=0.002, am=-0.018)),
                ],
            ),
            "InLens": DetectorMaps(
                overview_png="runs/R2/BAT-21/InLens_overview.png",
                crops_png="runs/R2/BAT-21/InLens_crops.png",
                regions=[
                    MapRegion(id=1, box_um=[31.0, 12.6, 44.1, 25.9], area_um2=174.2,
                              share_of_evidence=0.29,
                              phase_fractions=PhaseFractions(pore=0.318, cbd=0.168, am=0.514),
                              minus_whole_image=PhaseFractions(pore=0.005, cbd=0.019, am=-0.024)),
                ],
            ),
        },
    )


# --------------------------------------------------------------------------- #
# BAT-42 -- the nasty case
# --------------------------------------------------------------------------- #
def bat_42() -> BatteryResult:
    return BatteryResult(
        battery_id="BAT-42",
        run_id="2026-10-04T09:14:02Z",
        scope="v1",
        versions=VERSIONS,
        images=[
            ImageRecord(detector="BSE", file="BAT-42_BSE.tif", pixel_size_um=0.0611,
                        pixel_size_source="txt_sidecar", kv=5.0,
                        quality_flags=["cnr_low"], masked_area_frac=0.093),
            ImageRecord(detector="InLens", file="BAT-42_InLens.tif", pixel_size_um=0.0611,
                        pixel_size_source="txt_sidecar", kv=2.0,
                        entropy_flag=True, masked_area_frac=0.147),
        ],
        classification=Classification(
            # ETD is absent entirely: the sample was imaged with two detectors.
            per_detector={
                "BSE": TypeProbs(A=0.33, B=0.22, C=0.45),
                "InLens": TypeProbs(A=0.29, B=0.28, C=0.43),
            },
            fused=TypeProbs(A=0.31, B=0.25, C=0.44),
            predicted="C",
            runner_up="A",
            detector_agreement=2,
            # The second witness disagrees -- CONF005 will force disclosure.
            stats_classifier=StatsClassifier(
                predicted="A", probs=TypeProbs(A=0.52, B=0.19, C=0.29)
            ),
            unlike_any_type=True,
            confidence="low",
            confidence_reasons=[
                "the fused probability is below the high-confidence threshold",
                "no ETD image was supplied, so the fusion rests on two detectors",
                "the InLens segmentation entropy is above the training range",
                "this sample does not resemble any training type",
            ],
        ),
        statistics=[
            _stat("stats.BSE.porosity", "BSE", "porosity", "porosity",
                  0.411, "fraction",
                  {"A": 0.352, "B": 0.264, "C": 0.308},
                  {"A": 0.021, "B": 0.018, "C": 0.034},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": 2.810, "B": 8.167, "C": 3.029}, 1, cliffs=0.52),
            _stat("stats.BSE.d50", "BSE", "d50",
                  "median cross-sectional particle diameter",
                  12.74, "um",
                  {"A": 9.81, "B": 6.15, "C": 7.92},
                  {"A": 1.12, "B": 0.74, "C": 1.41},
                  {"A": 29, "B": 31, "C": 28},
                  {"A": 2.616, "B": 8.905, "C": 3.418}, 2, cliffs=0.61),
        ],
        maps={
            "BSE": DetectorMaps(
                overview_png="runs/R2/BAT-42/BSE_overview.png",
                crops_png="runs/R2/BAT-42/BSE_crops.png",
                regions=[
                    MapRegion(id=1, box_um=[18.3, 40.2, 39.7, 58.1], area_um2=383.1,
                              share_of_evidence=0.41,
                              phase_fractions=PhaseFractions(pore=0.462, cbd=0.108, am=0.430),
                              minus_whole_image=PhaseFractions(pore=0.051, cbd=-0.014, am=-0.037)),
                ],
            ),
        },
    )


def write_all(folder: str | Path = "examples") -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for result in (bat_21(), bat_42()):
        path = folder / f"{result.battery_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_all():
        print(f"wrote {path}")
