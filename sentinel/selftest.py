"""Seeded-defect harness: inject known faults into a valid explanation and
check that the deterministic validator catches each one.

This is not in the proposal, and it is the cheapest credible evidence the
explainer guardrails do anything. "100% of delivered explanations pass the
validator" (the proposal's done-when criterion) is satisfied trivially by a
validator that checks nothing. What a judge actually wants to know is the other
direction: *when an explanation is wrong, does the checker notice?*

Each row below is a specific way an explanation can be wrong, paired with the
error code that must fire. Extend the list and the output becomes a
precision/recall table for the checker rather than a pass/fail line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .contract import BatteryResult, Citation, Explanation
from .facts import FactSheet, build_fact_sheet
from .knowledge import KnowledgePack
from .template import TemplateExplainer
from .validator import Validator, ValidatorConfig

Mutator = Callable[[Explanation], Explanation]


@dataclass(frozen=True)
class Defect:
    name: str
    expected_code: str
    mutate: Mutator
    note: str = ""


def _copy(explanation: Explanation) -> Explanation:
    return explanation.model_copy(deep=True)


# --------------------------------------------------------------------------- #
# Building a known-good base explanation
# --------------------------------------------------------------------------- #
def real_quote(pack: KnowledgePack, *, max_len: int = 180) -> tuple[str, str] | None:
    """Pull a genuine sentence out of the pack, to seed a valid citation."""
    for sid in pack.ids():
        body = pack.sources[sid].normalised
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence = sentence.strip()
            if 40 <= len(sentence) <= max_len:
                return sid, sentence
    return None


def build_base(result: BatteryResult, pack: KnowledgePack) -> tuple[Explanation, FactSheet]:
    """A template explanation, upgraded with one real citation so it passes the
    full validator rather than the relaxed template one."""
    sheet = build_fact_sheet(result)
    base = TemplateExplainer().build(sheet)
    base.generator = "llm"          # judge it by the strict rules
    quote = real_quote(pack)
    if quote and base.evidence:
        sid, sentence = quote
        base.evidence[0].citations.append(Citation(source=sid, quote=sentence))
    return base, sheet


# --------------------------------------------------------------------------- #
# The defects
# --------------------------------------------------------------------------- #
def _set_claim(explanation: Explanation, text: str) -> Explanation:
    explanation.evidence[0].claim = text
    return explanation


def _literal_number(e: Explanation) -> Explanation:
    return _set_claim(_copy(e), "Porosity is 34.1%, which is typical of this product.")


def _number_word(e: Explanation) -> Explanation:
    return _set_claim(
        _copy(e), "The pore fraction is roughly twice the value of the other type."
    )


def _bare_type_letter(e: Explanation) -> Explanation:
    c = _copy(e)
    c.headline = "This sample is type A on the balance of the evidence."
    return c


def _attention_as_evidence(e: Explanation) -> Explanation:
    return _set_claim(
        _copy(e),
        "The attention map concentrates on the particle edges, which is why the "
        "model chose this type.",
    )


def _overconfident(e: Explanation) -> Explanation:
    c = _copy(e)
    c.headline = "The microstructure proves this sample is type {pred_type}."
    return c


def _fabricated_fact_id(e: Explanation) -> Explanation:
    c = _copy(e)
    c.evidence[0].facts = ["stats.BSE.tortuosity.value"]
    return c


def _fabricated_placeholder(e: Explanation) -> Explanation:
    return _set_claim(
        _copy(e), "The binder coverage is {stats.InLens.binder_halo.value} here."
    )


def _nonexistent_region(e: Explanation) -> Explanation:
    c = _copy(e)
    c.evidence[0].regions = ["BSE region 99"]
    return c


def _fabricated_quote(e: Explanation) -> Explanation:
    c = _copy(e)
    c.evidence[0].citations = [Citation(
        source="S1",
        quote="Calendering always reduces porosity by exactly one third in "
              "commercial electrodes.",
    )]
    return c


def _unknown_source(e: Explanation) -> Explanation:
    c = _copy(e)
    c.evidence[0].citations = [Citation(source="S99", quote="anything at all")]
    return c


def _undefined_marker(e: Explanation) -> Explanation:
    c = _copy(e)
    c.agreement = c.agreement + " This is consistent with the literature [S97]."
    return c


def _vacuous(e: Explanation) -> Explanation:
    """The reward hack: delete the evidence until every other rule passes."""
    c = _copy(e)
    c.evidence = []
    return c


def _ungrounded_evidence(e: Explanation) -> Explanation:
    c = _copy(e)
    for item in c.evidence:
        item.facts = []
        item.kind = "visual"
    return c


def _citations_stripped(e: Explanation) -> Explanation:
    c = _copy(e)
    for item in c.evidence:
        item.citations = []
    return c


def _runner_up_ignored(e: Explanation) -> Explanation:
    c = _copy(e)
    c.why_not_runner_up = ""
    return c


DEFECTS: tuple[Defect, ...] = (
    Defect("literal number in a claim", "NUM001", _literal_number),
    Defect("number word ('twice')", "NUM002", _number_word),
    Defect("bare type letter", "TYPE002", _bare_type_letter,
           "the explanation must not be able to contradict Step 4"),
    Defect("attention map as evidence", "ATTN001", _attention_as_evidence),
    Defect("certainty beyond the label", "CONF001", _overconfident),
    Defect("fabricated fact ID", "PH002", _fabricated_fact_id),
    Defect("fabricated placeholder", "PH001", _fabricated_placeholder),
    Defect("nonexistent region", "REGION001", _nonexistent_region),
    Defect("fabricated quote", "CITE001", _fabricated_quote),
    Defect("unknown source", "CITE001", _unknown_source),
    Defect("undefined [S#] marker", "CITE003", _undefined_marker),
    Defect("evidence deleted (gaming)", "CONTENT002", _vacuous,
           "passing by saying nothing must fail"),
    Defect("evidence with no facts", "CONTENT003", _ungrounded_evidence),
    Defect("citations stripped", "CITE002", _citations_stripped),
    Defect("runner-up not addressed", "CONTENT005", _runner_up_ignored),
)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_selftest(
    result: BatteryResult,
    pack: KnowledgePack,
    config: ValidatorConfig | None = None,
) -> tuple[list[tuple[str, str, str | None, bool]], int]:
    """Returns (rows, number caught). Each row: (name, expected, got, ok)."""
    base, sheet = build_base(result, pack)
    validator = Validator(sheet, pack, config or ValidatorConfig())

    baseline = validator.validate(base)
    rows: list[tuple[str, str, str | None, bool]] = [(
        "BASELINE (must pass cleanly)",
        "none",
        ", ".join(f.code for f in baseline.errors) or None,
        baseline.passed,
    )]

    for defect in DEFECTS:
        report = validator.validate(defect.mutate(base))
        codes = [f.code for f in report.errors]
        caught = defect.expected_code in codes
        rows.append((defect.name, defect.expected_code, ", ".join(codes) or None, caught))

    return rows, sum(1 for _, _, _, ok in rows if ok)
