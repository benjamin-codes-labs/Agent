"""More than medians: richer batch figures, and the record's own evidence.

The explanation must draw on at least three kinds of evidence the record
offers (batch baseline, composition, measured features, where the top evidence
lies), and the batch comparison now carries rank, observed range and whether
batches separate -- not just medians.
"""

import json
from pathlib import Path

import pytest

from sentinel.baselines import BatchBaselines, load_default_baselines
from sentinel.knowledge import KnowledgePack
from sentinel.pipeline_facts import (
    RECORD_EVIDENCE_RULES,
    FactsResponse,
    _certainty_checks,
    _comparison_claim_checks,
    _displayed_facts,
    _evidence_breadth_checks,
    explain_facts_document,
    parse_facts_document,
    prepare_facts_context,
    validate_facts_response,
)
from .conftest import FakeClient, comparison_payload, critic_pass, evidence_facts, text_response
from .test_baselines import make_record

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "examples" / "facts.json"
V3 = ROOT / "lucas-sem-analysis-v3" / "outputs" / "batchid" / "71vgq3fw.json"
needs_reference = pytest.mark.skipif(not V3.exists(), reason="reference outputs not present")


def _context(path):
    return prepare_facts_context(parse_facts_document(Path(path).read_text()), KnowledgePack.empty(),
                                 baselines=load_default_baselines(ROOT))


def _item(context, points):
    sample = context.samples[0]
    return FactsResponse.model_validate({"explanations": [{
        "sample_id": sample.sample_id, "reported_answer": sample.answer, "points": points}]}).explanations[0]


# --------------------------------------------------------------------------- #
# Richer batch figures
# --------------------------------------------------------------------------- #
def test_rank_observed_range_and_separation_are_computed():
    records = []
    for batch, centre in (("Batch_1", 10.0), ("Batch_2", 20.0), ("Batch_3", 30.0)):
        records += [make_record(f"{batch}_{i}", batch, centre + d) for i, d in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0))]
    pore = BatchBaselines(records).compare(make_record("new", None, 20.6, decision={"answer": "Batch_2"}))[
        "measurements"]["pore_pct"]
    assert pore["Batch_2"]["lower_n"] == 4 and pore["Batch_2"]["n"] == 5          # higher than 4 of 5
    assert pore["Batch_1"]["lower_n"] == 5                                          # higher than all Batch_1
    assert pore["Batch_3"]["lower_n"] == 0                                          # lower than all Batch_3
    assert (pore["Batch_2"]["min"], pore["Batch_2"]["max"]) == (19.0, 21.0)
    assert set(pore["typical_ranges_separate"]) == {
        "Batch_1 and Batch_2", "Batch_1 and Batch_3", "Batch_2 and Batch_3"}


@needs_reference
def test_the_demo_image_is_lower_than_every_batch_1_image_on_bse_graininess():
    m = _context(LEGACY).baselines["img_71vgq3fw"]["measurements"]["bse_noise"]
    assert m["Batch_1"]["lower_n"] == 0 and m["Batch_1"]["n"] == 7
    assert "Batch_1 and Batch_3" in m["typical_ranges_separate"]


# --------------------------------------------------------------------------- #
# The breadth rule
# --------------------------------------------------------------------------- #
@needs_reference
def test_medians_alone_are_not_enough():
    """The live output: baseline comparisons only -> one kind of evidence."""
    context = _context(LEGACY)
    sid = context.samples[0].sample_id
    b = f"context.{context.baseline_sources[sid]}.measurements"
    points = [
        {"kind": "basis", "text": "Noise {" + b + ".bse_noise.this_image} vs {" + b + ".bse_noise.Batch_3.median}.",
         "facts": [f"{b}.bse_noise.this_image", f"{b}.bse_noise.Batch_3.median"]},
        {"kind": "alternatives", "text": "Dark level {" + b + ".inlens_dark_level.this_image} vs {"
         + b + ".inlens_dark_level.Batch_1.median}.",
         "facts": [f"{b}.inlens_dark_level.this_image", f"{b}.inlens_dark_level.Batch_1.median"]},
        {"kind": "limitation", "text": "Few images per batch.",
         "citations": [{"source": context.baseline_sources[sid], "quote": "Each batch profile rests on only a handful of images"}]},
    ]
    shown = {f for f, _ in _displayed_facts(context)}
    errors = _evidence_breadth_checks(_item(context, points), context, shown)
    assert errors and "draw on at least 3 kinds" in errors[0]


@needs_reference
@pytest.mark.parametrize("path", [LEGACY, V3])
def test_baseline_plus_two_record_kinds_passes(path):
    """The availability bug: the baseline must count, though not in the fact table."""
    context = _context(path)
    response = FactsResponse.model_validate(comparison_payload(context))
    assert validate_facts_response(response, context) == []


@needs_reference
def test_only_kinds_the_record_offers_are_required():
    """A bare record with one kind of evidence must not be asked for three."""
    record = {"sample_id": "img_bare", "predicted_batch": "Batch_1", "confidence": {"tier": "Medium"},
              "phase_fractions_pct": {"pore": 14.0, "SiOx": 6.0}}
    context = prepare_facts_context(parse_facts_document(record), KnowledgePack.empty())
    fid = f"context.{context.sample_sources['img_bare']}.phase_fractions_pct.pore"
    item = _item(context, [{"kind": "basis", "text": "Pore {" + fid + "}.", "facts": [fid]}])
    assert _evidence_breadth_checks(item, context, {f for f, _ in _displayed_facts(context)}) == []


@needs_reference
def test_the_prompt_lists_the_four_kinds_and_the_richer_figures():
    context = _context(LEGACY)
    client = FakeClient([text_response(comparison_payload(context)), critic_pass()])
    explain_facts_document(LEGACY.read_text(), KnowledgePack.empty(), client=client,
                           baselines=load_default_baselines(ROOT))
    system = client.messages.requests[0]["system"]
    assert RECORD_EVIDENCE_RULES.splitlines()[0] in system
    assert "lower_n" in system and "observed range" in system and "do not overlap" in system
    assert "Decisive area" not in system                       # see test below
    prompt = "\n".join(b["text"] for b in client.messages.requests[0]["messages"][0]["content"])
    assert "images lower" in prompt and "typical ranges separate" in prompt


# --------------------------------------------------------------------------- #
# Rank and observed-range claims are checked by code
# --------------------------------------------------------------------------- #
@needs_reference
def test_lower_than_all_is_checked_against_the_rank():
    context = _context(LEGACY)
    sid = context.samples[0].sample_id
    m = f"context.{context.baseline_sources[sid]}.measurements.bse_noise"
    ok = _item(context, [{"kind": "basis", "facts": [f"{m}.this_image", f"{m}.Batch_1.n"],
                          "text": "BSE graininess {" + m + ".this_image} lower than all {" + m + ".Batch_1.n} Batch_1 images."}])
    assert _comparison_claim_checks(ok, context) == []          # lower_n is 0
    bad = _item(context, [{"kind": "basis", "facts": [f"{m}.this_image", f"{m}.Batch_3.n"],
                           "text": "BSE graininess {" + m + ".this_image} lower than all {" + m + ".Batch_3.n} Batch_3 images."}])
    errors = _comparison_claim_checks(bad, context)
    assert errors and "are lower than it" in errors[0]          # 3 of 16 are lower


@needs_reference
def test_inside_the_observed_range_is_checked():
    context = _context(LEGACY)
    sid = context.samples[0].sample_id
    m = f"context.{context.baseline_sources[sid]}.measurements.bse_noise"
    text = "BSE graininess {" + m + ".this_image}: inside Batch_1 observed range {" + m + ".Batch_1.min} to {" + m + ".Batch_1.max}"
    errors = _comparison_claim_checks(_item(context, [{"kind": "basis", "text": text, "facts": [
        f"{m}.this_image", f"{m}.Batch_1.min", f"{m}.Batch_1.max"]}]), context)
    assert errors and "observed range" in errors[0]


# --------------------------------------------------------------------------- #
# "decisive": a location word in the field name, a certainty word in a claim
# --------------------------------------------------------------------------- #
def _certainty(text, tier="Medium"):
    context = prepare_facts_context(parse_facts_document(LEGACY.read_text()), KnowledgePack.empty())
    sample = context.samples[0]
    sample = type(sample)(**{**sample.__dict__, "confidence": tier})
    return _certainty_checks(_item(context, [{"kind": "basis", "text": text, "citations": []}]), sample)


@pytest.mark.parametrize("text", [
    "Decisive area: graphite-rich, SiOx-poor.",
    "The decisive-area share of SiOx is low.",
    "Decisive regions sit on graphite.",
])
def test_decisive_naming_an_area_is_not_overconfidence(text):
    assert _certainty(text) == []


def test_decisive_as_a_claim_is_still_overconfidence():
    assert _certainty("The imaging evidence is decisive for Batch_3.")
