"""Jinja2 template explanation -- the fallback when the model cannot pass the
checks, and the only explanation shipped before the Claude call is ready
(proposal section 6.4 and the 15:30-17:00 timeline slot).

The template emits the *same* placeholder syntax as the explainer and is passed
through the same validator and renderer. That is deliberate: the fallback path
is proved safe by the same rules as the model path, and there is only one
renderer to trust.

Jinja delimiters are changed to ``<< >>`` so that the ``{fact.id}`` placeholders
the renderer consumes pass through Jinja untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from jinja2 import Environment, StrictUndefined

from .contract import EvidenceItem, Explanation
from .facts import FactSheet
from .validator import ValidatorConfig

ENV = Environment(
    variable_start_string="<<",
    variable_end_string=">>",
    block_start_string="<%",
    block_end_string="%>",
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

HEADLINE = ENV.from_string(
    "Classified as type {pred_type} with a fused probability of "
    "{cls.fused.<<pred>>}, confidence {confidence}."
)

WHY_NOT = ENV.from_string(
    "The clearest separation from type {runner_up} is <<label>> on <<detector>>: "
    "this battery measures {<<stat>>.value}"
    "<% if has_pred_mean %>, against a type-{pred_type} mean of {<<stat>>.mean_<<pred>>}<% endif %>"
    "<% if has_runner_mean %> and a type-{runner_up} mean of {<<stat>>.mean_<<runner>>}"
    "<% endif %>"
    "<% if has_delta %> (a difference of {<<stat>>.delta_vs_<<runner>>} against type-"
    "{runner_up})<% endif %>."
)

STAT_CLAIM = ENV.from_string(
    "<<Label>> on <<detector>> is {<<stat>>.value}"
    "<% if has_z_pred %>, which sits {<<stat>>.z_<<pred>>} standard deviations from the "
    "type-{pred_type} profile<% endif %>"
    "<% if has_z_runner %> and {<<stat>>.z_<<runner>>} from the type-{runner_up} profile"
    "<% endif %>."
)

REGION_CLAIM = ENV.from_string(
    "<<region>> carries {region.<<det>>.<<rid>>.share_of_evidence} of the positive "
    "evidence for type {pred_type}"
    "<% if has_pore %>; inside it the pore fraction is {region.<<det>>.<<rid>>.pore}"
    "<% if has_pore_delta %> ({region.<<det>>.<<rid>>.pore_delta} against the whole image)"
    "<% endif %><% endif %>"
    "<% if has_cbd %>, and the carbon-binder fraction is {region.<<det>>.<<rid>>.cbd}"
    "<% if has_cbd_delta %> ({region.<<det>>.<<rid>>.cbd_delta} against the whole image)"
    "<% endif %><% endif %>."
)

AGREEMENT = ENV.from_string(
    "Detectors predicting the fused type: {cls.detector_agreement}."
    "<% if stats_agrees is not none %>"
    "<% if stats_agrees %> The statistics branch independently predicts the same type."
    "<% else %> The statistics branch predicts a different type, so the branches "
    "disagree and this result should be reviewed by hand.<% endif %>"
    "<% endif %>"
)


def template_validator_config(base: ValidatorConfig | None = None) -> ValidatorConfig:
    """The template cannot quote the literature, so its citation floor is zero.

    This is the one rule relaxed for the fallback path. The output is labelled
    ``generator="template"`` so the platform can show that it is a degraded
    explanation rather than a cited one.
    """
    base = base or ValidatorConfig()
    return ValidatorConfig(
        min_evidence_items=min(base.min_evidence_items, 2),
        min_fact_backed_items=base.min_fact_backed_items,
        min_citations=0,
        require_why_not_runner_up=base.require_why_not_runner_up,
        require_caveat_below_high=base.require_caveat_below_high,
        require_low_confidence_hedge=base.require_low_confidence_hedge,
        citations_optional_without_pack=True,
        text_rules=base.text_rules,
        max_words=base.max_words,
        max_evidence_items=base.max_evidence_items,
        max_caveats=base.max_caveats,
    )


@dataclass
class TemplateExplainer:
    """Builds a complete, placeholder-based explanation from the fact sheet."""

    max_statistics: int = 3
    max_regions: int = 2

    def build(self, sheet: FactSheet) -> Explanation:
        pred, runner = sheet.predicted, sheet.runner_up
        stats = self._separating_stat_ids(sheet)

        evidence: list[EvidenceItem] = []
        for stat_id in stats[: self.max_statistics]:
            item = self._statistic_item(sheet, stat_id, pred, runner)
            if item:
                evidence.append(item)
        for region_name in sheet.region_ids[: self.max_regions]:
            item = self._region_item(sheet, region_name, pred)
            if item:
                evidence.append(item)

        # The content floor needs a second item; fall back to the probabilities.
        if len(evidence) < 2:
            evidence.append(self._probability_item(sheet, pred, runner))

        return Explanation(
            headline=HEADLINE.render(pred=pred),
            why_not_runner_up=self._why_not(sheet, stats, pred, runner),
            evidence=evidence,
            agreement=AGREEMENT.render(stats_agrees=sheet.stats_branch_agrees),
            caveats=self._caveats(sheet),
            generator="template",
        )

    # ------------------------------------------------------------------ #
    def _separating_stat_ids(self, sheet: FactSheet) -> list[str]:
        """Statistic base IDs in separating order, e.g. ``stats.BSE.porosity``."""
        ids: list[str] = []
        for entry in sheet.separating_statistics:
            base = entry.split(" ", 1)[0]
            if f"{base}.value" in sheet:
                ids.append(base)
        if ids:
            return ids
        seen: list[str] = []
        for fact_id in sheet.ids():
            if fact_id.startswith("stats.") and fact_id.endswith(".value"):
                base = fact_id[: -len(".value")]
                if base not in seen:
                    seen.append(base)
        return seen

    def _meta(self, sheet: FactSheet, stat_id: str) -> tuple[str, str, str]:
        """(label, detector, capitalised label) for a statistic base ID."""
        fact = sheet.get(f"{stat_id}.value")
        detector = (fact.detector if fact else None) or stat_id.split(".")[1]
        label = stat_id.split(".")[-1].replace("_", " ")
        if fact and fact.label:
            label = fact.label.split(" measured on ")[0]
        return label, detector, label[:1].upper() + label[1:]

    def _statistic_item(
        self, sheet: FactSheet, stat_id: str, pred: str, runner: str
    ) -> EvidenceItem | None:
        if f"{stat_id}.value" not in sheet:
            return None
        label, detector, Label = self._meta(sheet, stat_id)
        facts = [f"{stat_id}.value"]
        has_z_pred = f"{stat_id}.z_{pred}" in sheet
        has_z_runner = f"{stat_id}.z_{runner}" in sheet
        if has_z_pred:
            facts.append(f"{stat_id}.z_{pred}")
        if has_z_runner:
            facts.append(f"{stat_id}.z_{runner}")
        claim = STAT_CLAIM.render(
            stat=stat_id, Label=Label, detector=detector, pred=pred, runner=runner,
            has_z_pred=has_z_pred, has_z_runner=has_z_runner,
        )
        return EvidenceItem(kind="statistic", detector=detector, claim=claim, facts=facts)

    def _region_item(
        self, sheet: FactSheet, region_name: str, pred: str
    ) -> EvidenceItem | None:
        detector, _, rid = region_name.partition(" region ")
        base = f"region.{detector}.{rid}"
        if f"{base}.share_of_evidence" not in sheet:
            return None
        facts = [f"{base}.share_of_evidence"]
        flags = {}
        for phase in ("pore", "cbd"):
            flags[f"has_{phase}"] = f"{base}.{phase}" in sheet
            flags[f"has_{phase}_delta"] = f"{base}.{phase}_delta" in sheet
            if flags[f"has_{phase}"]:
                facts.append(f"{base}.{phase}")
            if flags[f"has_{phase}_delta"]:
                facts.append(f"{base}.{phase}_delta")
        claim = REGION_CLAIM.render(
            region=region_name, det=detector, rid=rid, pred=pred, **flags
        )
        return EvidenceItem(
            kind="both", detector=detector, claim=claim,
            facts=facts, regions=[region_name],
        )

    def _probability_item(self, sheet: FactSheet, pred: str, runner: str) -> EvidenceItem:
        facts = [f"cls.fused.{pred}"]
        if f"cls.fused.{runner}" in sheet:
            facts.append(f"cls.fused.{runner}")
        claim = (
            "The fused probability for type {pred_type} is {cls.fused." + pred + "}"
            + (
                ", against {cls.fused." + runner + "} for type {runner_up}."
                if f"cls.fused.{runner}" in sheet
                else "."
            )
        )
        return EvidenceItem(kind="statistic", claim=claim, facts=facts)

    def _why_not(
        self, sheet: FactSheet, stats: list[str], pred: str, runner: str
    ) -> str:
        if not stats:
            return (
                "The fused probability favours type {pred_type} over type "
                "{runner_up}; no separating statistic was available for this battery."
            )
        stat_id = stats[0]
        label, detector, _ = self._meta(sheet, stat_id)
        return WHY_NOT.render(
            stat=stat_id, label=label, detector=detector, pred=pred, runner=runner,
            has_pred_mean=f"{stat_id}.mean_{pred}" in sheet,
            has_runner_mean=f"{stat_id}.mean_{runner}" in sheet,
            has_delta=f"{stat_id}.delta_vs_{runner}" in sheet,
        )

    def _caveats(self, sheet: FactSheet) -> list[str]:
        caveats: list[str] = [
            "This explanation was assembled from the measured facts by a fixed "
            "template, not written by the explainer model."
        ]
        for reason in sheet.confidence_reasons:
            caveats.append(f"Confidence is limited: {reason}.")
        if sheet.missing_detectors:
            caveats.append(
                "No result is available for "
                + ", ".join(sheet.missing_detectors)
                + ", so the fusion rests on the remaining detectors and the "
                "confidence label is lowered."
            )
        if sheet.quality_flags:
            caveats.append(
                "Image quality flags were raised: " + ", ".join(sheet.quality_flags) + "."
            )
        if sheet.unlike_any_type:
            caveats.append(
                "This sample is flagged as unlike any training type, so it may not "
                "be type {pred_type} at all and may be outside the types seen in training."
            )
        if sheet.stats_branch_agrees is False:
            caveats.append(
                "The measurement branch and the classification branch disagree, so "
                "this result is weak and should be reviewed by hand."
            )
        if sheet.confidence == "low":
            caveats.append(
                "The confidence label is low: treat this as tentative and review it "
                "before acting on it."
            )
        caveats.append(
            "Particle-size percentiles are cross-sectional and carry stereological "
            "bias, so they compare samples with each other and not with a supplier's "
            "laser-diffraction figures."
        )
        return caveats


def build_template_explanation(sheet: FactSheet, **kwargs) -> Explanation:
    return TemplateExplainer(**kwargs).build(sheet)
