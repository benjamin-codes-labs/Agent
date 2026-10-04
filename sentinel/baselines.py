"""Per-batch baselines computed in code, so the explainer can compare an image
with Batch_1, Batch_2 and Batch_3 using real numbers instead of invented ones.

Why this exists
---------------
A facts.json record carries this image's measurements, but no batch profiles:
``robust_z`` is scaled against the *pooled* training set (median/IQR), and
``push_toward_winner_vs_runner_up`` gives only a direction. A language model has
no knowledge of these particular batches, so asking it to "compare with the
batch baselines" without supplying them invites it to make the profiles up.

The profiles are therefore derived here from the reference outputs of the
*same* pipeline (``lucas-sem-analysis-v3/outputs/batchid``), grouped by each
record's ``true_batch``. The model only narrates them.

Three guarantees
----------------
1. **Like for like.** Only records in the reference format (v3: ``phases``
   plus ``material_model.features_S``) are compared. A legacy record from a
   different segmentation run reports different values for the same image, so
   comparing it with v3 profiles would give wrong verdicts; it gets an explicit
   "not comparable" record instead.
2. **Leave-one-out.** When the image being explained is itself a reference
   image, it is removed from the profiles first, so it is never compared with a
   profile that contains itself.
3. **Code decides, the model narrates.** Every per-measurement verdict
   ("supports the reported batch", "favours Batch_2", "consistent with
   several") is computed here. The explainer can describe the verdict but
   cannot reach a different one, and opposing evidence is listed explicitly so
   it cannot be dropped quietly.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

#: The comparable measurements, with how to read each from a v3 record. These
#: are the composition phases (reliable per the record's own ``reliability``
#: note) and the material features that are not redundant with them --
#: ``pore_all_frac`` and ``siox_frac`` duplicate the pore and SiOx phases, and
#: the experimental graphite-versus-binder split is deliberately left out.
@dataclass(frozen=True)
class Measurement:
    key: str
    meaning: str
    unit: str
    read: Callable[[dict], float | None]
    #: "material" comes from segmentation and is comparable only between records
    #: of the same pipeline version; "imaging" is a raw-pixel statistic.
    group: str = "material"

    @property
    def path(self) -> str:
        """Where the value sits in a record, e.g. ``acquisition.probes.bse_noise_sigma``."""
        return getattr(self.read, "path", "")


def _phase(name: str) -> Callable[[dict], float | None]:
    def read(record: dict) -> float | None:
        value = record.get("phases", {}).get("three_class_pct", {}).get(name)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    read.path = f"phases.three_class_pct.{name}"
    return read


def _feature(name: str) -> Callable[[dict], float | None]:
    def read(record: dict) -> float | None:
        item = record.get("material_model", {}).get("features_S", {}).get(name)
        value = item.get("value") if isinstance(item, dict) else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    read.path = f"material_model.features_S.{name}.value"
    return read


def _probe(name: str) -> Callable[[dict], float | None]:
    def read(record: dict) -> float | None:
        acquisition = record.get("acquisition")
        probes = acquisition.get("probes") if isinstance(acquisition, dict) else None
        value = probes.get(name) if isinstance(probes, dict) else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    read.path = f"acquisition.probes.{name}"
    return read


MATERIAL_MEASUREMENTS: tuple[Measurement, ...] = (
    Measurement("pore_pct", "segmented pore area, percent of the image", "%",
                _phase("pore")),
    Measurement("carbon_pct", "graphite plus binder area, percent of the image", "%",
                _phase("carbon (graphite + binder)")),
    Measurement("siox_pct", "SiOx particle area, percent of the image", "%",
                _phase("SiOx")),
    Measurement("siox_particle_size_um", "area-weighted equivalent diameter of SiOx particles",
                "µm", _feature("siox_ecd_aw")),
    Measurement("open_pore_excess_pp", "all-pore minus deep-pore fraction (open, grey-floored pores)",
                "pp", _feature("pore_open_excess")),
    Measurement("pore_anisotropy", "horizontal over vertical median pore chord (1 = isotropic)",
                "ratio", _feature("pore_chord_aniso")),
)

#: Raw-pixel statistics of the images, computed before any segmentation. They
#: are byte-identical between the legacy and v3 records of the same image (all
#: thirteen probes checked), so they compare like for like across versions --
#: unlike the material measurements. They describe how the image was ACQUIRED,
#: which is why the pipeline warns that they can reflect the imaging session.
#: Left out because they separate nothing in the reference set: width_px
#: (constant), inlens_p99 (saturated at 255), excluded_frac (constant) and
#: bse_file_mb (a proxy for image dimensions).
IMAGING_MEASUREMENTS: tuple[Measurement, ...] = (
    Measurement("bse_noise", "graininess of the BSE image (noise level)", "score",
                _probe("bse_noise_sigma"), "imaging"),
    Measurement("bse_dark_level", "BSE grey level of the darkest 1% of pixels", "grey",
                _probe("bse_p1"), "imaging"),
    Measurement("bse_median_brightness", "median BSE grey level", "grey",
                _probe("bse_p50"), "imaging"),
    Measurement("bse_bright_level", "BSE grey level of the brightest 1% of pixels", "grey",
                _probe("bse_p99"), "imaging"),
    Measurement("inlens_dark_level", "InLens grey level of the darkest 1% of pixels", "grey",
                _probe("inlens_p1"), "imaging"),
    Measurement("inlens_median_brightness", "median InLens grey level", "grey",
                _probe("inlens_p50"), "imaging"),
    Measurement("inlens_sharpness", "sharpness of the InLens image", "score",
                _probe("inlens_sharpness"), "imaging"),
    Measurement("inlens_banding", "line-to-line brightness banding in the InLens image", "score",
                _probe("inlens_banding_power"), "imaging"),
    Measurement("image_height", "image height, an imaging-setup property", "px",
                _probe("height_px"), "imaging"),
)

MEASUREMENTS: tuple[Measurement, ...] = MATERIAL_MEASUREMENTS + IMAGING_MEASUREMENTS


def has_probes(record: dict) -> bool:
    acquisition = record.get("acquisition")
    return isinstance(acquisition, dict) and isinstance(acquisition.get("probes"), dict)

BATCH_PATTERN = re.compile(r"Batch_\d+")


def normalise_sample_id(sample_id: str) -> str:
    """``img_71vgq3fw`` and ``71vgq3fw`` are the same location."""
    return re.sub(r"^img_", "", str(sample_id).strip())


def is_comparable(record: dict) -> bool:
    """True when the record uses the reference set's feature definitions."""
    return (
        isinstance(record.get("phases"), dict)
        and isinstance(record.get("phases", {}).get("three_class_pct"), dict)
        and isinstance(record.get("material_model"), dict)
        and isinstance(record["material_model"].get("features_S"), dict)
    )


def reported_batches(record: dict) -> list[str]:
    """The batch or batches the decision names: specific, pair or leaning."""
    decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    pred_set = decision.get("prediction_set")
    if isinstance(pred_set, list) and all(isinstance(b, str) for b in pred_set) and pred_set:
        return sorted(set(pred_set))
    answer = decision.get("answer") or record.get("predicted_batch") or ""
    found = BATCH_PATTERN.findall(str(answer))
    if not found and isinstance(decision.get("leaning"), str):
        found = BATCH_PATTERN.findall(decision["leaning"])
    return sorted(set(found))


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    n: int
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float
    #: The batch's reference values themselves, so a rank can be computed.
    values: tuple[float, ...] = ()

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    def lower_than(self, value: float) -> int:
        """How many of this batch's reference images have a lower value."""
        return sum(1 for v in self.values if v < value)

    def as_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "median": self.median, "q1": self.q1, "q3": self.q3,
                "min": self.minimum, "max": self.maximum}


def _profile(values: list[float]) -> Profile | None:
    if len(values) < 3:
        return None
    ordered = sorted(values)
    # "inclusive" interpolates within the data, which suits n of about 7.
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return Profile(len(ordered), statistics.median(ordered), q1, q3, ordered[0], ordered[-1], tuple(ordered))


def _round(value: float, unit: str) -> float:
    return round(value, 3 if unit in {"ratio", "µm"} else 2)


@dataclass
class BatchBaselines:
    """Reference records grouped by true batch."""

    records: list[dict] = field(default_factory=list)
    origin: str = ""

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_reference_dir(cls, folder: str | Path) -> "BatchBaselines":
        folder = Path(folder)
        records: list[dict] = []
        for path in sorted(folder.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (isinstance(record, dict) and isinstance(record.get("true_batch"), str)
                    and is_comparable(record)):
                records.append(record)
        if not records:
            raise ValueError(f"no comparable reference records with true_batch in {folder}")
        return cls(records, origin=str(folder))

    @property
    def batches(self) -> list[str]:
        return sorted({r["true_batch"] for r in self.records})

    def profiles(self, exclude_sample: str | None = None) -> dict[str, dict[str, Profile]]:
        """{measurement: {batch: Profile}}, leaving out ``exclude_sample``."""
        excluded = normalise_sample_id(exclude_sample) if exclude_sample else None
        pool = [r for r in self.records
                if normalise_sample_id(r.get("sample_id", "")) != excluded]
        out: dict[str, dict[str, Profile]] = {}
        for m in MEASUREMENTS:
            per_batch: dict[str, Profile] = {}
            for batch in self.batches:
                values = [v for r in pool if r["true_batch"] == batch
                          if (v := m.read(r)) is not None]
                profile = _profile(values)
                if profile is not None:
                    per_batch[batch] = profile
            out[m.key] = per_batch
        return out

    # -- the comparison handed to the explainer --------------------------- #
    def compare(self, record: dict) -> dict[str, Any]:
        """Build the per-sample comparison record. Pure code, no model."""
        sample_id = str(record.get("sample_id", ""))
        material = is_comparable(record)
        imaging = has_probes(record)
        if not (material or imaging):
            return {
                "status": "not_comparable",
                "why": "This record has neither the reference set's material feature definitions "
                       "(phases.three_class_pct and material_model.features_S) nor raw-image "
                       "acquisition probes, so no like-for-like batch comparison is made. Explain "
                       "from the record's own evidence and say that no batch baseline was applied.",
            }

        reported = reported_batches(record)
        in_reference = any(normalise_sample_id(r.get("sample_id", "")) == normalise_sample_id(sample_id)
                           for r in self.records)
        profiles = self.profiles(exclude_sample=sample_id)
        pool_n = {b: max((p.n for m in profiles.values() for bb, p in m.items() if bb == b), default=0)
                  for b in self.batches}

        groups = {"material": material, "imaging": imaging}
        measurements: dict[str, Any] = {}
        summary: dict[str, list[str]] = {
            "supports_reported": [], "favours_an_alternative": [], "consistent_with_several_or_atypical": []}
        for m in MEASUREMENTS:
            if not groups[m.group]:
                continue
            value = m.read(record)
            per_batch = profiles.get(m.key, {})
            if value is None or len(per_batch) < 2:
                continue
            entry = self._measure(m, value, per_batch, reported)
            measurements[m.key] = entry
            verdict = entry["verdict"]
            if verdict == "supports_reported":
                summary["supports_reported"].append(m.key)
            elif verdict.startswith("favours_"):
                summary["favours_an_alternative"].append(m.key)
            else:
                summary["consistent_with_several_or_atypical"].append(m.key)

        by_group = {
            group: {key: [k for k in keys if measurements[k]["group"] == group] for key, keys in summary.items()}
            for group, present in groups.items() if present
        }
        caveats = [
            "Each batch profile rests on only a handful of images, so medians and typical ranges "
            "are approximate.",
            "A measurement inside several batches' typical ranges does not discriminate between them.",
            "Imaging measurements describe how the image was acquired (brightness, noise, sharpness), "
            "so a match can reflect the imaging session rather than the material.",
        ]
        if material:
            caveats.append("Composition is measured from segmentation, not chemistry.")
        else:
            caveats.append(
                "This record's material measurements come from a different segmentation run than the "
                "reference set (the same image reports different values), so only the imaging "
                "measurements, which are raw-pixel statistics identical across runs, are compared.")

        return {
            "status": "compared",
            "scope": "material_and_imaging" if material else "imaging_only",
            "what_this_is": (
                "Per-batch profiles computed in code from reference images grouped by their known "
                "batch, with this image left out. For each measurement: this image's value, each "
                "batch's median and typical range (q1 to q3), and a verdict computed in code. "
                "These are reference statistics, not model output."
            ),
            "reported_batches": reported,
            "reference_set": {
                "n_images": sum(pool_n.values()),
                "per_batch_n": pool_n,
                "this_image_left_out": in_reference,
            },
            "caveats": caveats,
            "measurements": measurements,
            "summary": summary,
            "summary_by_group": by_group,
        }

    def _measure(self, m: Measurement, value: float, per_batch: dict[str, Profile],
                 reported: list[str]) -> dict[str, Any]:
        inside_iqr = [b for b, p in per_batch.items() if p.q1 <= value <= p.q3]
        inside_range = [b for b, p in per_batch.items() if p.minimum <= value <= p.maximum]

        # IQR-scaled distance to each batch median; a zero IQR falls back to the
        # range so a degenerate batch cannot dominate.
        distance: dict[str, float] = {}
        for b, p in per_batch.items():
            scale = p.iqr or (p.maximum - p.minimum) or None
            if scale:
                distance[b] = round((value - p.median) / scale, 2)
        nearest = min(distance, key=lambda b: abs(distance[b])) if distance else None

        alternatives = [b for b in per_batch if b not in reported]
        fits_reported = any(b in inside_iqr for b in reported)
        fits_alternatives = [b for b in alternatives if b in inside_iqr]

        if fits_reported and not fits_alternatives:
            verdict = "supports_reported"
        elif fits_reported:
            verdict = "consistent_with_several"
        elif fits_alternatives:
            verdict = "favours_" + "_and_".join(sorted(fits_alternatives))
        elif nearest and nearest not in reported:
            verdict = "favours_" + nearest
        elif nearest:
            verdict = "closest_to_reported_but_atypical"
        else:
            verdict = "atypical_for_all"

        # Pairs of batches whose typical ranges do not overlap: the measurements
        # that genuinely tell those batches apart. A match on a measurement where
        # every range overlaps means much less than one where they separate.
        ordered = sorted(per_batch.items())
        separate = [f"{a} and {b}" for i, (a, pa) in enumerate(ordered) for b, pb in ordered[i + 1:]
                    if pa.q3 < pb.q1 or pb.q3 < pa.q1]

        # Per batch: centre (median), typical range (q1-q3), observed range
        # (min-max), how many images it rests on (n), and this image's rank
        # among them (lower_n: how many of the batch's images are lower).
        per_batch_out = {}
        for b, p in ordered:
            stats = {k: (_round(v, m.unit) if isinstance(v, float) else v) for k, v in p.as_dict().items()}
            stats["lower_n"] = p.lower_than(value)
            per_batch_out[b] = stats
        return {
            "group": m.group,
            "meaning": m.meaning,
            "unit": m.unit,
            "this_image": _round(value, m.unit),
            **per_batch_out,
            "inside_typical_range_of": sorted(inside_iqr),
            "inside_observed_range_of": sorted(inside_range),
            "typical_ranges_separate": separate,
            "nearest_median": nearest,
            "verdict": verdict,
        }


# --------------------------------------------------------------------------- #
# Display strings for the generated facts
# --------------------------------------------------------------------------- #
_UNIT_BY_MEASUREMENT = {m.key: m.unit for m in MEASUREMENTS}


def baseline_display(key: str, value: Any, fallback: str) -> str:
    """How a ``context.SN.<key>`` fact from a baseline record is shown."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return fallback
    parts = key.split(".")
    if "measurements" not in parts:
        return fallback
    try:
        measurement = parts[parts.index("measurements") + 1]
    except IndexError:
        return fallback
    field = parts[-1]
    if field == "n" or field.endswith("_n"):
        return str(int(value))
    if "distance_to_median_in_iqr" in parts:
        return f"{value:+.2f} IQR"
    unit = _UNIT_BY_MEASUREMENT.get(measurement)
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "pp":
        return f"{value:.1f} pp"
    if unit == "µm":
        return f"{value:.2f} µm"
    if unit == "ratio":
        return f"{value:.3f}"
    if unit == "grey":
        return f"{round(value, 1):g}"           # 0-255 grey level; medians can be half-steps
    if unit == "score":
        return f"{value:.3g}"
    if unit == "px":
        return f"{value:.0f} px"
    return fallback


def find_reference_dir(root: str | Path) -> Path | None:
    """The default reference folder, if this checkout has it."""
    candidate = Path(root) / "lucas-sem-analysis-v3" / "outputs" / "batchid"
    return candidate if candidate.is_dir() else None


def load_default_baselines(root: str | Path) -> BatchBaselines | None:
    folder = find_reference_dir(root)
    if folder is None:
        return None
    try:
        return BatchBaselines.from_reference_dir(folder)
    except ValueError:
        return None


def describe(comparison: dict[str, Any]) -> Iterable[str]:
    """Human-readable lines, for --debug and for checking by eye."""
    if comparison.get("status") != "compared":
        yield f"no baseline: {comparison.get('why', '')}"
        return
    ref = comparison["reference_set"]
    yield (f"reported {comparison['reported_batches']} | reference n={ref['n_images']} "
           f"{ref['per_batch_n']} | this image left out: {ref['this_image_left_out']}")
    for key, m in comparison["measurements"].items():
        medians = "  ".join(f"{b} {m[b]['median']}" for b in sorted(k for k in m if k.startswith("Batch_")))
        yield f"  {key:<24} this {m['this_image']:<8} medians: {medians}  -> {m['verdict']}"
