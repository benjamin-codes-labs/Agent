"""Code validator (proposal section 6.4).

Checks, all deterministic:

* every placeholder resolves to a fact ID on the sheet, and every region ID
  exists                                                     -- ``PH*``/``REGION*``
* no numbers outside placeholders, number words included      -- ``NUM*``
* the predicted type is unchanged and never named literally   -- ``TYPE*``
* every quote appears word for word in its cited source       -- ``CITE*``
* the attention map is not offered as evidence                -- ``ATTN*``
* certainty wording matches the confidence label              -- ``CONF*``
* the explanation actually says something                     -- ``CONTENT*``

The ``CONTENT*`` family is not in the proposal and matters more than it looks.
Every other rule is satisfied most easily by writing nothing: an explanation
with no numbers, no citations and no regions passes all of them. With a retry
loop pushing towards "whatever the validator accepts", the cheapest escape is a
vacuous paragraph -- a small, real case of gaming the checker. ``CONTENT*`` puts
a floor under the output so that escape fails too.

Findings carry a stable ``code``, a JSON-ish ``loc``, what was observed and a
``fix_hint``; that shape is what makes the retry converge instead of oscillate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from .contract import Explanation
from .facts import FactSheet
from .knowledge import KnowledgePack
from .textrules import (
    TextRuleConfig,
    find_region_mentions,
    missing_low_confidence_hedge,
    scan_field,
)

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    loc: str
    message: str
    observed: str | None = None
    fix_hint: str | None = None

    def key(self) -> tuple[str, str, str]:
        """Identity for oscillation detection: code + location + observed text."""
        return (self.code, self.loc, self.observed or "")

    def render(self) -> str:
        line = f"[{self.code}] {self.loc}: {self.message}"
        if self.observed:
            line += f"\n    observed: {self.observed!r}"
        if self.fix_hint:
            line += f"\n    fix: {self.fix_hint}"
        return line


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def progress_key(self) -> tuple[int, int]:
        """Lexicographic progress measure: (errors, warnings), lower is better."""
        return (len(self.errors), len(self.warnings))

    def signature(self) -> tuple[tuple[str, str, str], ...]:
        """Sorted multiset of finding identities, for detecting a stuck loop."""
        return tuple(sorted(f.key() for f in self.findings))

    def feedback(self, limit: int = 20) -> str:
        if self.passed and not self.warnings:
            return "No problems found."
        lines = ["The previous attempt failed these checks. Fix each one:"]
        lines += [f.render() for f in self.errors[:limit]]
        extra = [f for f in self.warnings if f.severity == "warning"][: max(0, limit - len(self.errors))]
        if extra:
            lines.append("Warnings (fix if you can):")
            lines += [f.render() for f in extra]
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidatorConfig:
    #: Minimum evidence items in a delivered explanation.
    min_evidence_items: int = 2
    #: At least this many items must cite a measured fact.
    min_fact_backed_items: int = 1
    #: At least this many citations across the whole explanation.
    min_citations: int = 1
    #: Require the runner-up to be addressed.
    require_why_not_runner_up: bool = True
    #: Require a caveat when the label is not "high".
    require_caveat_below_high: bool = True
    #: Require a hedge phrase when the label is "low".
    require_low_confidence_hedge: bool = True
    #: Citations may be dropped when the pack is empty (template/offline runs).
    citations_optional_without_pack: bool = True
    text_rules: TextRuleConfig = TextRuleConfig()


class Validator:
    def __init__(
        self,
        sheet: FactSheet,
        pack: KnowledgePack | None = None,
        config: ValidatorConfig | None = None,
    ):
        self.sheet = sheet
        self.pack = pack or KnowledgePack.empty()
        self.config = config or ValidatorConfig()
        self.text_rules = self._trust_contract_text(self.config.text_rules, sheet)

    @staticmethod
    def _trust_contract_text(rules: TextRuleConfig, sheet: FactSheet) -> TextRuleConfig:
        """Excuse strings that upstream code wrote, not the model.

        ``confidence_reasons`` and ``quality_flags`` arrive through the JSON
        contract from Step 4, and both the template and the explainer are
        *supposed* to repeat them verbatim. They are code-authored, so policing
        them for literal numbers is wrong: a reason like "the fusion rests on
        two detectors" is trustworthy by construction, and without this the
        validator rejects its own template whenever Lucas writes a numeral in a
        reason string.

        Only these exact strings are excused. Any other number in the same
        field is still caught.
        """
        trusted = tuple(
            text.strip().rstrip(".")
            for text in (*sheet.confidence_reasons, *sheet.quality_flags)
            if text and text.strip()
        )
        return rules.with_extra(phrases=trusted) if trusted else rules

    # ------------------------------------------------------------------ #
    def validate(self, explanation: Explanation) -> ValidationReport:
        report = ValidationReport()
        self._check_text_fields(explanation, report)
        self._check_evidence_structure(explanation, report)
        self._check_citations(explanation, report)
        self._check_content_floor(explanation, report)
        return report

    # -- lexical + placeholder ------------------------------------------ #
    def _check_text_fields(self, explanation: Explanation, report: ValidationReport) -> None:
        confidence = self.sheet.confidence
        for loc, text in explanation.text_fields():
            if not text:
                continue
            allow_attention = loc.startswith("caveats")
            masked, hits = scan_field(
                text,
                self.text_rules,
                allow_attention=allow_attention,
                confidence=confidence,
            )
            for hit in hits:
                report.findings.append(self._finding_for_hit(loc, masked, hit))
            self._check_placeholders(loc, masked, report)
            self._check_region_mentions(loc, text, report)

    def _check_region_mentions(
        self, loc: str, text: str, report: ValidationReport
    ) -> None:
        """A region named in prose must exist, even though it was masked above."""
        valid = set(self.sheet.region_ids)
        for name, _, _ in find_region_mentions(text):
            if name not in valid:
                report.findings.append(Finding(
                    "REGION003", "error", loc,
                    f"the text names {name!r}, which does not exist for this battery",
                    observed=name,
                    fix_hint="valid regions: " + (", ".join(sorted(valid)) or "none"),
                ))

    def _finding_for_hit(self, loc, masked, hit) -> Finding:
        codes = {
            "brace": ("PH003", "error"),
            "digit": ("NUM001", "error"),
            "number_word": ("NUM002", "error"),
            "ordinal_word": ("NUM003", "error"),
            "ordinal_word_warning": ("NUM003", "warning"),
            "bare_type": ("TYPE002", "error"),
            "attention": ("ATTN001", "error"),
            "certainty": ("CONF001", "error"),
        }
        code, severity = codes.get(hit.kind, ("NUM001", "error"))
        hints = {
            "NUM001": "replace the number with a placeholder such as "
                      "{stats.BSE.porosity.value}; every number must come from the fact sheet",
            "NUM002": "the fact sheet has precomputed comparisons "
                      "(delta_vs_A, *_delta); use one instead of saying it in words",
            "NUM003": "rephrase without the ordinal, or name the region "
                      "(\"BSE region 1\") instead of counting",
            "TYPE002": "write {pred_type} or {runner_up}",
            "ATTN001": "the attention map shows only where the model looked; "
                       "cite a numbered evidence region and its local statistics",
            "CONF001": f"the confidence label is {self.sheet.confidence!r}; "
                       "use wording no stronger than \"is consistent with\" / \"indicates\"",
            "PH003": "balance the braces",
        }
        return Finding(
            code=code,
            severity=severity,  # type: ignore[arg-type]
            loc=loc,
            message=hit.detail,
            observed=masked.context(hit.start, hit.end),
            fix_hint=hints.get(code),
        )

    def _check_placeholders(self, loc: str, masked, report: ValidationReport) -> None:
        for name, start, end in masked.placeholders:
            if name in {"pred_type", "runner_up", "confidence"}:
                continue
            if not name:
                report.findings.append(Finding(
                    "PH003", "error", loc, "empty placeholder",
                    observed=masked.context(start, end),
                    fix_hint="remove it or name a fact ID",
                ))
                continue
            if name in self.sheet:
                continue
            suggestions = self.sheet.nearest_ids(name)
            report.findings.append(Finding(
                "PH001", "error", loc,
                f"placeholder {{{name}}} is not a fact on the sheet",
                observed=masked.context(start, end),
                fix_hint=(
                    "use one of: " + ", ".join(suggestions)
                    if suggestions
                    else "only fact IDs listed in the fact sheet may be used"
                ),
            ))

    # -- evidence structure --------------------------------------------- #
    def _check_evidence_structure(
        self, explanation: Explanation, report: ValidationReport
    ) -> None:
        valid_regions = set(self.sheet.region_ids)
        for i, item in enumerate(explanation.evidence):
            loc = f"evidence[{i}]"
            for fact_id in item.facts:
                if fact_id not in self.sheet:
                    report.findings.append(Finding(
                        "PH002", "error", f"{loc}.facts",
                        f"fact ID {fact_id!r} does not exist",
                        observed=fact_id,
                        fix_hint=(
                            "did you mean: " + ", ".join(self.sheet.nearest_ids(fact_id))
                            if self.sheet.nearest_ids(fact_id)
                            else "list only IDs from the fact sheet"
                        ),
                    ))
            for region in item.regions:
                if region not in valid_regions:
                    report.findings.append(Finding(
                        "REGION001", "error", f"{loc}.regions",
                        f"region {region!r} does not exist for this battery",
                        observed=region,
                        fix_hint=(
                            "valid regions: " + (", ".join(sorted(valid_regions)) or "none")
                        ),
                    ))
            if item.detector and item.detector not in self.sheet.available_detectors:
                report.findings.append(Finding(
                    "REGION002", "warning", f"{loc}.detector",
                    f"detector {item.detector} has no result for this battery",
                    observed=item.detector,
                    fix_hint="available: " + ", ".join(self.sheet.available_detectors),
                ))
            if item.kind in {"statistic", "both"} and not item.facts:
                report.findings.append(Finding(
                    "CONTENT003", "error", f"{loc}.facts",
                    f"a {item.kind!r} evidence item must list at least one fact ID",
                    fix_hint="add the fact IDs the claim rests on",
                ))
            if item.kind in {"visual", "both"} and not item.regions:
                report.findings.append(Finding(
                    "CONTENT004", "warning", f"{loc}.regions",
                    f"a {item.kind!r} evidence item should name an evidence region",
                    fix_hint="name a region such as \"BSE region 1\"",
                ))

    # -- citations ------------------------------------------------------- #
    def _check_citations(self, explanation: Explanation, report: ValidationReport) -> None:
        for i, item in enumerate(explanation.evidence):
            for j, citation in enumerate(item.citations):
                loc = f"evidence[{i}].citations[{j}]"
                check = self.pack.verify_quote(citation.source, citation.quote)
                if check.ok:
                    continue
                hint = "quote a sentence from the knowledge pack exactly as written"
                if check.suggestion:
                    hint = f"the nearest real sentence is: {check.suggestion!r}"
                report.findings.append(Finding(
                    "CITE001", "error", loc,
                    check.reason or "the quote could not be verified",
                    observed=citation.quote[:160],
                    fix_hint=hint,
                ))
        # [S#] markers typed into prose must also resolve.
        for loc, text in explanation.text_fields():
            for marker in self.pack.undefined_markers(text):
                report.findings.append(Finding(
                    "CITE003", "error", loc,
                    f"citation marker [{marker}] has no source in the knowledge pack",
                    observed=f"[{marker}]",
                    fix_hint="cite only sources present in the pack: "
                             + (", ".join(self.pack.ids()) or "none"),
                ))

    # -- content floor (anti-degradation) -------------------------------- #
    def _check_content_floor(
        self, explanation: Explanation, report: ValidationReport
    ) -> None:
        cfg = self.config
        if not explanation.headline.strip():
            report.findings.append(Finding(
                "CONTENT001", "error", "headline", "the headline is empty",
                fix_hint="state the predicted type and the fused probability",
            ))
        if len(explanation.evidence) < cfg.min_evidence_items:
            report.findings.append(Finding(
                "CONTENT002", "error", "evidence",
                f"only {len(explanation.evidence)} evidence item(s); "
                f"at least {cfg.min_evidence_items} are required",
                observed=str(len(explanation.evidence)),
                fix_hint="an explanation that says less is not a passing explanation; "
                         "add evidence drawn from the fact sheet",
            ))
        fact_backed = sum(1 for item in explanation.evidence if item.facts)
        if fact_backed < cfg.min_fact_backed_items:
            report.findings.append(Finding(
                "CONTENT003", "error", "evidence",
                f"only {fact_backed} evidence item(s) cite a measured fact; "
                f"at least {cfg.min_fact_backed_items} must",
                observed=str(fact_backed),
                fix_hint="ground at least one claim in a measured statistic",
            ))
        want_citations = cfg.min_citations
        if not len(self.pack) and cfg.citations_optional_without_pack:
            want_citations = 0
        n_citations = sum(len(item.citations) for item in explanation.evidence)
        if n_citations < want_citations:
            report.findings.append(Finding(
                "CITE002", "error", "evidence",
                f"only {n_citations} citation(s); at least {want_citations} required",
                observed=str(n_citations),
                fix_hint="quote the knowledge pack for the materials-science reasoning",
            ))
        if cfg.require_why_not_runner_up and not explanation.why_not_runner_up.strip():
            report.findings.append(Finding(
                "CONTENT005", "error", "why_not_runner_up",
                "the runner-up is not addressed",
                fix_hint="say what distinguishes {pred_type} from {runner_up} here",
            ))
        if (
            cfg.require_caveat_below_high
            and self.sheet.confidence != "high"
            and not [c for c in explanation.caveats if c.strip()]
        ):
            report.findings.append(Finding(
                "CONF002", "error", "caveats",
                f"the confidence label is {self.sheet.confidence!r} but no caveat is given",
                fix_hint="state what limits the confidence: "
                         + (
                             "; ".join(self.sheet.confidence_reasons)
                             or "missing detectors, quality flags or branch disagreement"
                         ),
            ))
        if (
            cfg.require_low_confidence_hedge
            and self.sheet.confidence == "low"
            and missing_low_confidence_hedge(
                " ".join(t for _, t in explanation.text_fields()), self.text_rules
            )
        ):
            report.findings.append(Finding(
                "CONF003", "error", "headline",
                "a low-confidence result must be hedged explicitly",
                fix_hint="say plainly that the result is tentative and should be reviewed",
            ))
        if self.sheet.unlike_any_type and not any(
            "unlike" in c.lower() or "outside" in c.lower() for c in explanation.caveats
        ):
            report.findings.append(Finding(
                "CONF004", "error", "caveats",
                "this battery is flagged unlike any training type, which the "
                "explanation must say",
                fix_hint="add a caveat that the sample does not resemble the "
                         "training types and may not be one of them",
            ))
        if self.sheet.stats_branch_agrees is False and not any(
            "disagree" in t.lower() or "differ" in t.lower()
            for _, t in explanation.text_fields()
        ):
            report.findings.append(Finding(
                "CONF005", "error", "agreement",
                "the two branches disagree, which the explanation must state",
                fix_hint="say that the statistics branch predicts a different type",
            ))


def validate(
    explanation: Explanation,
    sheet: FactSheet,
    pack: KnowledgePack | None = None,
    config: ValidatorConfig | None = None,
) -> ValidationReport:
    return Validator(sheet, pack, config).validate(explanation)
