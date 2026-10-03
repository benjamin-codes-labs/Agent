"""Evidence builder (proposal section 6.2): turn everything upstream into one
fact sheet per battery, giving every number an ID.

Design notes beyond the proposal
--------------------------------
1. Every fact carries a pre-rendered ``display`` string and the renderer emits
   *only* that. Otherwise the number in the explanation and the number in the
   JSON can differ by rounding, which breaks the audit trail.

2. The proposal forbids numbers outside placeholders, but a natural
   "why not the runner-up" sentence wants a *comparison* ("8 points above type
   A's mean"). If no fact ID exists for that, the explainer invents the number,
   fails the validator, burns both retries and falls back to the template every
   time. So the builder precomputes the comparisons: ``delta_vs_<type>`` for
   every separating statistic and ``<phase>_delta`` for every region.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .contract import (
    DETECTORS,
    TYPES,
    BatteryResult,
    BatteryType,
    Detector,
    StatisticRecord,
)

FactKind = Literal[
    "classification", "measurement", "profile", "zscore", "derived", "region", "flag", "context"
]


class Fact(BaseModel):
    """One addressable number or label. ``display`` is what reaches the screen."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: FactKind
    value: float | int | str | bool | None
    display: str
    unit: str | None = None
    detector: Detector | None = None
    label: str | None = None
    note: str | None = None

    def brief(self) -> str:
        bits = [f"{self.id} = {self.display}"]
        if self.label:
            bits.append(f"({self.label})")
        if self.note:
            bits.append(f"[{self.note}]")
        return " ".join(bits)


# --------------------------------------------------------------------------- #
# Display formatting
# --------------------------------------------------------------------------- #
def _sig(value: float, digits: int = 3) -> str:
    """Round to ``digits`` significant figures without exponent noise."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, digits - 1 - magnitude)
    if decimals > 6:
        return f"{value:.{digits - 1}e}"
    return f"{value:.{decimals}f}"


def format_quantity(value: float | int | None, unit: str | None) -> str:
    """Render a measured quantity the one way it will ever be shown."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    unit = (unit or "").strip()

    if unit in {"fraction", "frac"}:
        return f"{value * 100:.1f}%"
    if unit in {"percentage_points", "pp"}:
        return f"{value * 100:+.1f} pp" if abs(value) < 1 else f"{value:+.1f} pp"
    if unit in {"percent", "%"}:
        return f"{value:.1f}%"
    if unit in {"z", "sd"}:
        return f"{value:+.2f}"
    if unit in {"count", "n"}:
        return f"{int(round(value))}"
    if unit == "dimensionless":
        return _sig(float(value), 3)
    if unit:
        return f"{_sig(float(value), 3)} {unit}"
    return _sig(float(value), 3)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Fact sheet
# --------------------------------------------------------------------------- #
class FactSheet(BaseModel):
    """The explainer's entire permitted universe of numbers."""

    model_config = ConfigDict(extra="forbid")

    battery_id: str
    run_id: str
    predicted: BatteryType
    runner_up: BatteryType
    confidence: str
    confidence_reasons: list[str] = Field(default_factory=list)
    unlike_any_type: bool = False
    available_detectors: list[Detector] = Field(default_factory=list)
    missing_detectors: list[Detector] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    detector_agreement: int = 0
    stats_branch_agrees: bool | None = None
    facts: dict[str, Fact] = Field(default_factory=dict)
    region_ids: list[str] = Field(default_factory=list)
    separating_statistics: list[str] = Field(default_factory=list)

    # -- lookups ---------------------------------------------------------- #
    def __contains__(self, fact_id: object) -> bool:
        return fact_id in self.facts

    def get(self, fact_id: str) -> Fact | None:
        return self.facts.get(fact_id)

    def ids(self) -> list[str]:
        return sorted(self.facts)

    def nearest_ids(self, fact_id: str, limit: int = 4) -> list[str]:
        """Cheap suggestions for a bad ID, so retry feedback is actionable."""
        import difflib

        return difflib.get_close_matches(fact_id, self.ids(), n=limit, cutoff=0.5)

    # -- prompt rendering ------------------------------------------------- #
    def to_prompt_block(self) -> str:
        lines: list[str] = [
            "## Classification (fixed -- you may not change it)",
            f"battery: {self.battery_id}",
            f"predicted type: {self.predicted}   (placeholder: {{pred_type}})",
            f"runner-up type: {self.runner_up}   (placeholder: {{runner_up}})",
            f"confidence label: {self.confidence}",
        ]
        if self.confidence_reasons:
            lines.append("confidence reasons: " + "; ".join(self.confidence_reasons))
        lines.append(f"detectors available: {', '.join(self.available_detectors) or 'none'}")
        if self.missing_detectors:
            lines.append(f"detectors missing: {', '.join(self.missing_detectors)}")
        lines.append(f"detector agreement: {self.detector_agreement} of 3")
        if self.stats_branch_agrees is not None:
            lines.append(
                "statistics branch agrees: "
                + ("yes" if self.stats_branch_agrees else "NO -- say so plainly")
            )
        lines.append(f"unlike any training type: {'yes' if self.unlike_any_type else 'no'}")
        if self.quality_flags:
            lines.append("quality flags: " + ", ".join(self.quality_flags))

        groups: dict[str, list[Fact]] = {}
        for fact in self.facts.values():
            groups.setdefault(fact.kind, []).append(fact)

        order: list[tuple[str, str]] = [
            ("classification", "## Probabilities"),
            ("measurement", "## Measured statistics for this battery"),
            ("profile", "## Type profiles (training batteries)"),
            ("zscore", "## z-scores against each type profile"),
            ("derived", "## Precomputed comparisons -- use these, never do arithmetic"),
            ("region", "## Evidence regions and their local measurements"),
            ("flag", "## Flags"),
            ("context", "## External context -- batch aggregates and reference guidance, NOT this sample's measurements or training profiles"),
        ]
        for kind, heading in order:
            items = sorted(groups.get(kind, []), key=lambda f: f.id)
            if not items:
                continue
            lines.append("")
            lines.append(heading)
            lines.extend(f"- {f.brief()}" for f in items)

        if self.separating_statistics:
            lines.append("")
            lines.append(
                "## Statistics that separate {pred_type} from {runner_up}, best first"
            )
            lines.extend(f"{i}. {s}" for i, s in enumerate(self.separating_statistics, 1))

        if self.region_ids:
            lines.append("")
            lines.append("## Valid region names")
            lines.append(", ".join(self.region_ids))

        return "\n".join(lines)


class EvidenceBuilder:
    """Builds the fact sheet. Pure code: no model involved, so it cannot drift."""

    def __init__(self, *, top_statistics: int = 5, max_regions_per_detector: int = 3):
        self.top_statistics = top_statistics
        self.max_regions_per_detector = max_regions_per_detector

    def build(self, result: BatteryResult) -> FactSheet:
        cls = result.classification
        sheet = FactSheet(
            battery_id=result.battery_id,
            run_id=result.run_id,
            predicted=cls.predicted,
            runner_up=cls.runner_up,
            confidence=cls.confidence,
            confidence_reasons=list(cls.confidence_reasons),
            unlike_any_type=cls.unlike_any_type,
            available_detectors=result.available_detectors(),
            missing_detectors=result.missing_detectors(),
            quality_flags=result.quality_flagged(),
            detector_agreement=cls.detector_agreement,
            stats_branch_agrees=cls.stats_branch_agrees,
            region_ids=result.region_ids(),
        )
        self._add_classification_facts(sheet, result)
        self._add_statistic_facts(sheet, result)
        self._add_profile_summary_facts(sheet, result)
        self._add_region_facts(sheet, result)
        self._add_flag_facts(sheet, result)
        return sheet

    # -- sections --------------------------------------------------------- #
    def _put(self, sheet: FactSheet, fact: Fact) -> None:
        sheet.facts[fact.id] = fact

    def _add_classification_facts(self, sheet: FactSheet, result: BatteryResult) -> None:
        cls = result.classification
        for t in TYPES:
            self._put(sheet, Fact(
                id=f"cls.fused.{t}",
                kind="classification",
                value=cls.fused.as_dict()[t],
                display=_pct(cls.fused.as_dict()[t]),
                unit="fraction",
                label=f"fused probability of type {t}",
            ))
        for det in result.available_detectors():
            probs = cls.per_detector[det].as_dict()
            for t in TYPES:
                self._put(sheet, Fact(
                    id=f"cls.{det}.{t}",
                    kind="classification",
                    value=probs[t],
                    display=_pct(probs[t]),
                    unit="fraction",
                    detector=det,
                    label=f"{det} probability of type {t}",
                ))
            top_type, top_prob = cls.per_detector[det].top()
            self._put(sheet, Fact(
                id=f"cls.{det}.predicted",
                kind="classification",
                value=top_type,
                display=top_type,
                detector=det,
                label=f"type predicted by {det} alone",
                note=f"at {_pct(top_prob)}",
            ))

        self._put(sheet, Fact(
            id="cls.detector_agreement",
            kind="classification",
            value=cls.detector_agreement,
            display=f"{cls.detector_agreement} of 3",
            unit="count",
            label="detectors predicting the fused type",
        ))
        self._put(sheet, Fact(
            id="cls.confidence",
            kind="classification",
            value=cls.confidence,
            display=cls.confidence,
            label="confidence label",
        ))
        if cls.stats_classifier is not None:
            self._put(sheet, Fact(
                id="cls.stats_classifier.predicted",
                kind="classification",
                value=cls.stats_classifier.predicted,
                display=cls.stats_classifier.predicted,
                label="type predicted by the statistics branch",
            ))
            for t in TYPES:
                p = cls.stats_classifier.probs.as_dict()[t]
                self._put(sheet, Fact(
                    id=f"cls.stats_classifier.{t}",
                    kind="classification",
                    value=p,
                    display=_pct(p),
                    unit="fraction",
                    label=f"statistics-branch probability of type {t}",
                ))

    def _add_statistic_facts(self, sheet: FactSheet, result: BatteryResult) -> None:
        separating = result.separating_statistics(self.top_statistics)
        sheet.separating_statistics = [
            f"{s.id} ({s.label or s.name}, primary detector "
            f"{s.primary_detector or s.detector})"
            for s in separating
        ]
        # Facts for the separating statistics, plus any statistic the platform
        # already shows, so the explainer is never tempted off-sheet.
        for stat in separating or result.statistics:
            self._add_one_statistic(sheet, stat)

    def _add_one_statistic(self, sheet: FactSheet, stat: StatisticRecord) -> None:
        label = stat.label or stat.name
        if stat.value is not None:
            self._put(sheet, Fact(
                id=f"{stat.id}.value",
                kind="measurement",
                value=stat.value,
                display=format_quantity(stat.value, stat.unit),
                unit=stat.unit,
                detector=stat.detector,
                label=f"{label} measured on {stat.detector}",
            ))
        for t in TYPES:
            profile = stat.profiles.get(t)
            if profile is None:
                continue
            if profile.mean is not None:
                self._put(sheet, Fact(
                    id=f"{stat.id}.mean_{t}",
                    kind="profile",
                    value=profile.mean,
                    display=format_quantity(profile.mean, stat.unit),
                    unit=stat.unit,
                    detector=stat.detector,
                    label=f"mean {label} of type {t}",
                    note=f"n={profile.n}" if profile.n else None,
                ))
            if profile.sd is not None:
                self._put(sheet, Fact(
                    id=f"{stat.id}.sd_{t}",
                    kind="profile",
                    value=profile.sd,
                    display=format_quantity(profile.sd, stat.unit),
                    unit=stat.unit,
                    detector=stat.detector,
                    label=f"standard deviation of {label} in type {t}",
                    note="an SD from ~30 batteries is itself uncertain by ~13%",
                ))
            if profile.n is not None:
                self._put(sheet, Fact(
                    id=f"{stat.id}.n_{t}",
                    kind="profile",
                    value=profile.n,
                    display=str(profile.n),
                    unit="count",
                    detector=stat.detector,
                    label=f"training batteries of type {t} behind this profile",
                ))
            z = stat.z.get(t)
            if z is not None:
                self._put(sheet, Fact(
                    id=f"{stat.id}.z_{t}",
                    kind="zscore",
                    value=z,
                    display=format_quantity(z, "z"),
                    unit="z",
                    detector=stat.detector,
                    label=f"{label} distance from type {t}, in SDs of type {t}",
                ))
            # Derived comparison: the sentence the explainer actually wants.
            if stat.value is not None and profile.mean is not None:
                delta = stat.value - profile.mean
                unit = "pp" if stat.unit in {"fraction", "frac"} else stat.unit
                self._put(sheet, Fact(
                    id=f"{stat.id}.delta_vs_{t}",
                    kind="derived",
                    value=delta,
                    display=format_quantity(delta, unit),
                    unit=unit,
                    detector=stat.detector,
                    label=f"this battery's {label} minus the mean of type {t}",
                    note="precomputed difference",
                ))
        if stat.cliffs_delta is not None:
            self._put(sheet, Fact(
                id=f"{stat.id}.cliffs_delta",
                kind="derived",
                value=stat.cliffs_delta,
                display=format_quantity(stat.cliffs_delta, "dimensionless"),
                unit="dimensionless",
                detector=stat.detector,
                label=f"Cliff's delta for {label}, predicted vs runner-up",
            ))

    def _add_region_facts(self, sheet: FactSheet, result: BatteryResult) -> None:
        for det in DETECTORS:
            maps = result.maps.get(det)
            if not maps:
                continue
            for region in maps.regions[: self.max_regions_per_detector]:
                base = f"region.{det}.{region.id}"
                name = f"{det} region {region.id}"
                if region.share_of_evidence is not None:
                    self._put(sheet, Fact(
                        id=f"{base}.share_of_evidence",
                        kind="region",
                        value=region.share_of_evidence,
                        display=_pct(region.share_of_evidence),
                        unit="fraction",
                        detector=det,
                        label=f"share of the positive evidence in {name}",
                    ))
                if region.area_um2 is not None:
                    self._put(sheet, Fact(
                        id=f"{base}.area_um2",
                        kind="region",
                        value=region.area_um2,
                        display=format_quantity(region.area_um2, "um2"),
                        unit="um2",
                        detector=det,
                        label=f"area of {name}",
                    ))
                for phase in ("pore", "cbd", "am"):
                    local = getattr(region.phase_fractions, phase)
                    if local is not None:
                        self._put(sheet, Fact(
                            id=f"{base}.{phase}",
                            kind="region",
                            value=local,
                            display=_pct(local),
                            unit="fraction",
                            detector=det,
                            label=f"{phase} fraction inside {name}",
                        ))
                    delta = getattr(region.minus_whole_image, phase)
                    if delta is not None:
                        self._put(sheet, Fact(
                            id=f"{base}.{phase}_delta",
                            kind="derived",
                            value=delta,
                            display=format_quantity(delta, "pp"),
                            unit="pp",
                            detector=det,
                            label=(
                                f"{phase} fraction in {name} minus the whole-image value"
                            ),
                            note="precomputed difference",
                        ))

    def _add_profile_summary_facts(self, sheet: FactSheet, result: BatteryResult) -> None:
        """A citable sample size for the profiles as a whole.

        Proposal 3.2 wants every profile shown with the caveat that an SD from
        about thirty batteries is itself uncertain by roughly 13%, and the
        explainer reliably tries to say so. Per-statistic ``n_A`` facts cannot
        express it, because the claim is about the profiles collectively, so
        without a fact for it the model can only write "about thirty batteries"
        in words -- which the no-numbers-in-words rule then rejects, costing a
        retry on a caveat that ought to be encouraged.
        """
        counts = [
            profile.n
            for stat in result.statistics
            for profile in stat.profiles.values()
            if profile.n
        ]
        if not counts:
            return
        low, high = min(counts), max(counts)
        display = str(low) if low == high else f"{low} to {high}"
        self._put(sheet, Fact(
            id="profiles.batteries_per_type",
            kind="profile",
            value=low if low == high else f"{low}-{high}",
            display=display,
            unit="count",
            label="training batteries behind each type profile",
            note="an SD from about this many batteries is itself uncertain by "
                 "roughly 13%, so profile widths are approximate",
        ))

    def _add_flag_facts(self, sheet: FactSheet, result: BatteryResult) -> None:
        for img in result.images:
            if img.masked_area_frac is not None:
                self._put(sheet, Fact(
                    id=f"image.{img.detector}.masked_area_frac",
                    kind="flag",
                    value=img.masked_area_frac,
                    display=_pct(img.masked_area_frac),
                    unit="fraction",
                    detector=img.detector,
                    label=f"share of the {img.detector} image masked as artefact",
                ))
            if img.pixel_size_um is not None:
                self._put(sheet, Fact(
                    id=f"image.{img.detector}.pixel_size_um",
                    kind="flag",
                    value=img.pixel_size_um,
                    display=format_quantity(img.pixel_size_um, "um"),
                    unit="um",
                    detector=img.detector,
                    label=f"pixel size of the {img.detector} image",
                    note=f"from {img.pixel_size_source}" if img.pixel_size_source else None,
                ))


def build_fact_sheet(result: BatteryResult, **kwargs: Any) -> FactSheet:
    return EvidenceBuilder(**kwargs).build(result)
