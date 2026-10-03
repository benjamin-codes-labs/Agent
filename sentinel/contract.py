"""The versioned JSON contract (proposal section 17).

One pydantic model shared by the pipeline, the API and the explainer. Two rules
from the proposal are enforced here rather than left to convention:

  * ``classification`` is written only by Step 4. The explainer reads it and
    cannot change it -- see :meth:`BatteryResult.with_explanation`, which is the
    only sanctioned way for Step 6 to produce a new document.
  * A results file is never modified after writing; a rerun gets a new run_id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Detector = Literal["BSE", "ETD", "InLens"]
BatteryType = Literal["A", "B", "C"]
ConfidenceLabel = Literal["high", "medium", "low"]
DETECTORS: tuple[Detector, ...] = ("BSE", "ETD", "InLens")
TYPES: tuple[BatteryType, ...] = ("A", "B", "C")

Fraction01 = Annotated[float, Field(ge=0.0, le=1.0)]


class Strict(BaseModel):
    """Reject unknown fields so a contract drift fails loudly, at the boundary."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Step 1 output
# --------------------------------------------------------------------------- #
class ImageRecord(Strict):
    detector: Detector
    file: str
    pixel_size_um: float | None = None
    pixel_size_source: str | None = None
    kv: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    entropy_flag: bool = False
    masked_area_frac: float | None = None


# --------------------------------------------------------------------------- #
# Step 4 output -- the prediction. Immutable as far as Step 6 is concerned.
# --------------------------------------------------------------------------- #
class TypeProbs(Strict):
    A: Fraction01
    B: Fraction01
    C: Fraction01

    def as_dict(self) -> dict[str, float]:
        return {"A": self.A, "B": self.B, "C": self.C}

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.as_dict().items(), key=lambda kv: -kv[1])

    def top(self) -> tuple[str, float]:
        return self.ranked()[0]


class StatsClassifier(Strict):
    predicted: BatteryType
    probs: TypeProbs


class Classification(Strict):
    per_detector: dict[Detector, TypeProbs]
    fused: TypeProbs
    predicted: BatteryType
    runner_up: BatteryType
    detector_agreement: int = Field(ge=0, le=3)
    stats_classifier: StatsClassifier | None = None
    unlike_any_type: bool = False
    confidence: ConfidenceLabel
    confidence_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> "Classification":
        if self.predicted == self.runner_up:
            raise ValueError("predicted and runner_up must differ")
        ranked = self.fused.ranked()
        if ranked[0][0] != self.predicted:
            raise ValueError(
                f"predicted={self.predicted} is not the argmax of fused probabilities "
                f"({ranked[0][0]} is). Step 4 must write a coherent document."
            )
        if ranked[1][0] != self.runner_up:
            raise ValueError(
                f"runner_up={self.runner_up} is not the second-highest fused "
                f"probability ({ranked[1][0]} is)."
            )
        return self

    @property
    def fused_top_prob(self) -> float:
        return self.fused.top()[1]

    @property
    def stats_branch_agrees(self) -> bool | None:
        if self.stats_classifier is None:
            return None
        return self.stats_classifier.predicted == self.predicted


# --------------------------------------------------------------------------- #
# Step 3 output
# --------------------------------------------------------------------------- #
class Profile(Strict):
    mean: float | None = None
    sd: float | None = None
    n: int | None = None
    median: float | None = None


class StatisticRecord(Strict):
    """One measured statistic for one detector, with each type's profile."""

    id: str                      # e.g. "stats.BSE.porosity"
    detector: Detector
    name: str                    # e.g. "porosity"
    label: str | None = None     # human phrase, e.g. "porosity"
    value: float | None = None
    unit: str = "fraction"
    profiles: dict[BatteryType, Profile] = Field(default_factory=dict)
    z: dict[BatteryType, float | None] = Field(default_factory=dict)
    separates_pred_vs_runner_up_rank: int | None = None
    cliffs_delta: float | None = None
    primary_detector: Detector | None = None

    @model_validator(mode="after")
    def _id_matches(self) -> "StatisticRecord":
        expected = f"stats.{self.detector}.{self.name}"
        if self.id != expected:
            raise ValueError(f"statistic id {self.id!r} should be {expected!r}")
        return self


# --------------------------------------------------------------------------- #
# Step 5 output
# --------------------------------------------------------------------------- #
class PhaseFractions(Strict):
    pore: float | None = None
    cbd: float | None = None
    am: float | None = None


class MapRegion(Strict):
    id: int = Field(ge=1)
    box_um: list[float] | None = None
    area_um2: float | None = None
    share_of_evidence: Fraction01 | None = None
    phase_fractions: PhaseFractions = Field(default_factory=PhaseFractions)
    minus_whole_image: PhaseFractions = Field(default_factory=PhaseFractions)


class DetectorMaps(Strict):
    overview_png: str | None = None
    crops_png: str | None = None
    regions: list[MapRegion] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Step 6 output -- the explanation
# --------------------------------------------------------------------------- #
class Citation(Strict):
    source: str          # "S3"
    quote: str           # must appear word for word in that source


class EvidenceItem(Strict):
    kind: Literal["statistic", "visual", "both"]
    detector: Detector | None = None
    claim: str
    facts: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class Explanation(Strict):
    headline: str = ""
    why_not_runner_up: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    agreement: str = ""
    caveats: list[str] = Field(default_factory=list)
    generator: Literal["llm", "template"] = "llm"
    validator_passed: bool | None = None
    critic_passed: bool | None = None

    def text_fields(self) -> list[tuple[str, str]]:
        """Every free-text span the validator must police, with a JSON-ish path."""
        out: list[tuple[str, str]] = [
            ("headline", self.headline),
            ("why_not_runner_up", self.why_not_runner_up),
            ("agreement", self.agreement),
        ]
        for i, item in enumerate(self.evidence):
            out.append((f"evidence[{i}].claim", item.claim))
        for i, caveat in enumerate(self.caveats):
            out.append((f"caveats[{i}]", caveat))
        return out


class SignOff(Strict):
    decision: Literal["approve", "override"]
    user: str
    timestamp: str
    reason: str | None = None
    overridden_type: BatteryType | None = None

    @model_validator(mode="after")
    def _override_needs_reason(self) -> "SignOff":
        if self.decision == "override" and not (self.reason and self.reason.strip()):
            raise ValueError("an override needs a reason (design principle 4)")
        return self


class Versions(Strict):
    model_config = ConfigDict(extra="allow")

    git: str | None = None
    segmenters: dict[Detector, str] = Field(default_factory=dict)
    backbone: str | None = None
    heads: str | None = None
    llm: str | None = None
    knowledge_pack: str | None = None
    prompt: str | None = None


class BatteryResult(Strict):
    battery_id: str
    run_id: str
    scope: Literal["v1", "v2"] = "v1"
    versions: Versions = Field(default_factory=Versions)
    images: list[ImageRecord] = Field(default_factory=list)
    classification: Classification
    statistics: list[StatisticRecord] = Field(default_factory=list)
    maps: dict[Detector, DetectorMaps] = Field(default_factory=dict)
    explanation: Explanation | None = None
    signoff: SignOff | None = None

    # -- read helpers ----------------------------------------------------- #
    def available_detectors(self) -> list[Detector]:
        return [d for d in DETECTORS if d in self.classification.per_detector]

    def missing_detectors(self) -> list[Detector]:
        return [d for d in DETECTORS if d not in self.classification.per_detector]

    def quality_flagged(self) -> list[str]:
        flags: list[str] = []
        for img in self.images:
            flags.extend(f"{img.detector}:{f}" for f in img.quality_flags)
            if img.entropy_flag:
                flags.append(f"{img.detector}:segmentation_unreliable")
        return flags

    def statistic(self, stat_id: str) -> StatisticRecord | None:
        for s in self.statistics:
            if s.id == stat_id:
                return s
        return None

    def separating_statistics(self, limit: int = 5) -> list[StatisticRecord]:
        """Top statistics separating the predicted type from the runner-up."""
        ranked = [s for s in self.statistics if s.separates_pred_vs_runner_up_rank]
        ranked.sort(key=lambda s: s.separates_pred_vs_runner_up_rank or 10**6)
        return ranked[:limit]

    def region_ids(self) -> list[str]:
        """Canonical region names, e.g. ``["BSE region 1", ...]``."""
        out: list[str] = []
        for det in DETECTORS:
            maps = self.maps.get(det)
            if not maps:
                continue
            out.extend(f"{det} region {r.id}" for r in maps.regions)
        return out

    def region(self, name: str) -> tuple[Detector, MapRegion] | None:
        for det in DETECTORS:
            maps = self.maps.get(det)
            if not maps:
                continue
            for r in maps.regions:
                if f"{det} region {r.id}" == name:
                    return det, r
        return None

    # -- the only sanctioned write path for Step 6 ------------------------ #
    def with_explanation(self, explanation: Explanation) -> "BatteryResult":
        """Return a new document carrying ``explanation``.

        The classification is copied verbatim. Step 6 never mutates a results
        file in place, so an explanation rerun is always a new document and the
        prediction cannot drift.
        """
        return self.model_copy(update={"explanation": explanation}, deep=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
