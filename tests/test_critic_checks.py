"""The critic is advisory; the checks code can do exactly are in the validator.

Built from a live run in which the critic rejected an explanation three times
over, and all three findings were wrong:

  C  "7.34 is at the exact q1 boundary, not clearly inside the range"
     -> the typical range includes its ends.
  D  "32.75 below 44.25 ... verdict says atypical, not outside the range"
     -> no causal claim at all, and "atypical" means outside every range.
  E  "verdict favours_Batch_1, confirming the statement, but presents this as
     evidence against Batch_3" -> opposing evidence, which the rules require.
"""

import json
from pathlib import Path

import pytest

from sentinel.baselines import load_default_baselines
from sentinel.explainer import ExplainerError
from sentinel.knowledge import KnowledgePack
from sentinel.pipeline_facts import (
    CRITIC_BREVITY,
    WHOLE_FILE_CRITIC_MAX_TOKENS,
    FactsResponse,
    _certainty_checks,
    _comparison_claim_checks,
    explain_facts_document,
    parse_facts_document,
    prepare_facts_context,
)
from .conftest import (
    FakeClient, FakeResponse, FakeToolUse, comparison_payload, critic_fail, critic_pass, text_response, tool_response,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "examples" / "facts.json"
V3 = ROOT / "lucas-sem-analysis-v3" / "outputs" / "batchid" / "71vgq3fw.json"
needs_reference = pytest.mark.skipif(not V3.exists(), reason="reference outputs not present")


def _cut_off(_):
    return FakeResponse(content=[FakeToolUse("report_grounding", {"passed": False, "problems": []})],
                        stop_reason="max_tokens", model="fake-haiku")


def _boom(_):
    raise RuntimeError("529 overloaded")


def _live_false_findings():
    """The three findings from the live run, verbatim in substance."""
    return tool_response("report_grounding", {"passed": False, "problems": [
        {"kind": "C", "loc": "explanations[0].points[0].text", "quote": "inside Batch_3 typical range",
         "why": "7.34 is at the exact q1 boundary, not clearly inside the range."},
        {"kind": "D", "loc": "explanations[0].points[2].text", "quote": "below Batch_3 typical range start",
         "why": "32.75 is below 44.25, but the verdict says atypical, not outside the range."},
        {"kind": "E", "loc": "explanations[0].points[2].text", "quote": "nearer Batch_1 median",
         "why": "verdict favours_Batch_1, confirming the statement, but presented as evidence against Batch_3."},
    ]})


def _run(*critic_steps, path=LEGACY, mode="advisory", retries=0):
    doc = Path(path).read_text()
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(doc), KnowledgePack.empty(), baselines=baselines)
    client = FakeClient([text_response(comparison_payload(context)), *critic_steps])
    outcome = explain_facts_document(doc, KnowledgePack.empty(), client=client, baselines=baselines,
                                     max_retries=retries, critic_mode=mode)
    return outcome, client


# --------------------------------------------------------------------------- #
# Advisory by default
# --------------------------------------------------------------------------- #
@needs_reference
def test_the_live_false_findings_no_longer_block():
    outcome, _ = _run(_live_false_findings())
    assert outcome.response.explanations                      # delivered
    assert outcome.audit["critic_mode"] == "advisory"
    assert outcome.audit["critic_passed"] is False            # the objection is on record
    assert len(outcome.audit["critic_advisory_findings"]) == 3


@needs_reference
def test_blocking_mode_restores_the_gate():
    with pytest.raises(ExplainerError, match="grounding critic flagged"):
        _run(_live_false_findings(), mode="blocking")


@needs_reference
def test_off_mode_makes_no_critic_call():
    outcome, client = _run(mode="off")
    assert len(client.messages.requests) == 1
    assert outcome.audit["semantic_critic_ran"] is False and outcome.audit["critic_passed"] is None


@needs_reference
def test_an_unavailable_advisory_critic_is_recorded_not_fatal():
    outcome, _ = _run(_cut_off, _cut_off)
    assert outcome.response.explanations
    assert outcome.audit["semantic_critic_ran"] is False
    assert "cut off at the 4000-token limit" in outcome.audit["critic_error"]


def test_main_exposes_the_critic_flag():
    import main
    import inspect
    assert inspect.signature(main.main).parameters["critic"].default == "advisory"


# --------------------------------------------------------------------------- #
# Critic robustness (restored after test_speed.py was overwritten on disk)
# --------------------------------------------------------------------------- #
@needs_reference
def test_a_cut_off_critic_is_retried_not_fatal():
    outcome, client = _run(_cut_off, critic_pass(), mode="blocking")
    assert outcome.audit["critic_passed"] is True
    assert len(client.messages.requests) == 3                 # writer, critic, critic again


@needs_reference
def test_a_transport_error_in_the_critic_is_retried():
    outcome, _ = _run(_boom, critic_pass(), mode="blocking")
    assert outcome.audit["critic_passed"] is True


@needs_reference
def test_two_blocking_critic_failures_name_the_real_cause():
    with pytest.raises(ExplainerError) as caught:
        _run(_cut_off, _cut_off, mode="blocking")
    assert "cut off at the 4000-token limit" in str(caught.value)
    assert "failed 2 time(s)" in str(caught.value)


@needs_reference
def test_one_malformed_objection_does_not_void_the_verdict():
    step = tool_response("report_grounding", {"passed": True, "problems": [
        {"kind": "Z", "loc": "points[0]", "quote": "x", "why": "not a real kind"}]})
    outcome, _ = _run(step, mode="blocking")
    assert outcome.response.explanations


@needs_reference
def test_a_rejection_with_only_malformed_objections_fails_closed_in_blocking_mode():
    step = tool_response("report_grounding", {"passed": False, "problems": [
        {"kind": "Z", "loc": "points[0]", "quote": "x", "why": "unusable"}]})
    with pytest.raises(ExplainerError, match="without detailed findings"):
        _run(step, mode="blocking")


@needs_reference
def test_critic_gets_room_and_is_asked_to_be_brief():
    _, client = _run(critic_pass())
    critic_request = client.messages.requests[1]
    assert critic_request["max_tokens"] >= WHOLE_FILE_CRITIC_MAX_TOKENS
    assert CRITIC_BREVITY.strip() in critic_request["system"]


# --------------------------------------------------------------------------- #
# Certainty, now checked by code
# --------------------------------------------------------------------------- #
def _legacy_item(text, tier):
    context = prepare_facts_context(parse_facts_document(LEGACY.read_text()), KnowledgePack.empty())
    sample = context.samples[0]
    sample = type(sample)(**{**sample.__dict__, "confidence": tier})
    item = FactsResponse.model_validate({"explanations": [{
        "sample_id": sample.sample_id, "reported_answer": sample.answer,
        "points": [{"kind": "basis", "text": text, "citations": []}]}]}).explanations[0]
    return item, sample


@pytest.mark.parametrize("text,tier", [
    ("The imaging proves Batch_3.", "High"),
    ("Definitely Batch_3 on every measurement.", "Medium"),
    ("Clearly shows a Batch_3 microstructure.", "Medium"),
    ("Strongly Batch_3 on imaging.", "Low"),
])
def test_overstated_certainty_is_caught_by_code(text, tier):
    item, sample = _legacy_item(text, tier)
    assert _certainty_checks(item, sample)


@pytest.mark.parametrize("text,tier", [
    ("Clearly shows a Batch_3 imaging match.", "High"),
    ("Strongly matches Batch_3 on imaging.", "Medium"),
    ("Consistent with Batch_3 on imaging; composition does not single it out.", "Low"),
])
def test_wording_within_the_tier_passes(text, tier):
    item, sample = _legacy_item(text, tier)
    assert _certainty_checks(item, sample) == []


# --------------------------------------------------------------------------- #
# Comparison claims, now checked by code (inclusive ranges)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def v3():
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(V3.read_text()), KnowledgePack.empty(), baselines=baselines)
    sid = context.samples[0].sample_id
    base = f"context.{context.baseline_sources[sid]}.measurements"
    return context, sid, base


def _claims(v3, text):
    context, sid, _ = v3
    item = FactsResponse.model_validate({"explanations": [{
        "sample_id": sid, "reported_answer": context.samples[0].answer,
        "points": [{"kind": "basis", "text": text, "citations": []}]}]}).explanations[0]
    return _comparison_claim_checks(item, context)


@needs_reference
def test_a_value_on_the_range_boundary_is_inside(v3):
    """The critic's finding C: 7.34 equals q1, and the range includes its ends."""
    _, _, b = v3
    m = f"{b}.bse_noise"
    facts = v3[0].pack.context_facts
    assert facts[f"{m}.this_image"].value == facts[f"{m}.Batch_3.q1"].value
    assert _claims(v3, "BSE graininess {" + m + ".this_image}: inside Batch_3 typical range {"
                   + m + ".Batch_3.q1} to {" + m + ".Batch_3.q3}") == []


@needs_reference
def test_below_the_typical_range_start_is_true(v3):
    """The critic's finding D: 32.75 is below Batch_3's q1."""
    _, _, b = v3
    m = f"{b}.inlens_sharpness"
    assert _claims(v3, "InLens sharpness {" + m + ".this_image} is below Batch_3 typical range start {"
                   + m + ".Batch_3.q1}") == []


@needs_reference
def test_above_the_batch_3_median_is_true(v3):
    """The critic's finding E: 62 is above Batch_3's median of 57."""
    _, _, b = v3
    m = f"{b}.bse_median_brightness"
    assert _claims(v3, "BSE median brightness {" + m + ".this_image}: above Batch_3 median {"
                   + m + ".Batch_3.median}, nearer Batch_1 median") == []


@needs_reference
def test_a_wrong_direction_is_caught(v3):
    _, _, b = v3
    m = f"{b}.inlens_sharpness"
    errors = _claims(v3, "InLens sharpness {" + m + ".this_image}: above Batch_3 median {" + m + ".Batch_3.median}")
    assert errors and "but it is lower" in errors[0]


@needs_reference
def test_a_false_inside_claim_is_caught(v3):
    _, _, b = v3
    m = f"{b}.bse_noise"
    errors = _claims(v3, "BSE graininess {" + m + ".this_image}: inside Batch_1 typical range {"
                     + m + ".Batch_1.q1} to {" + m + ".Batch_1.q3}")
    assert errors and "lies outside it" in errors[0]


@needs_reference
def test_a_false_outside_claim_is_caught(v3):
    _, _, b = v3
    m = f"{b}.bse_noise"
    errors = _claims(v3, "BSE graininess {" + m + ".this_image}: outside Batch_3 typical range {"
                     + m + ".Batch_3.q1} to {" + m + ".Batch_3.q3}")
    assert errors and "range includes its ends" in errors[0]


@needs_reference
def test_a_comparison_word_in_another_clause_is_not_misread(v3):
    """'... ; Batch_1 median above ...' is about the median, not this image."""
    _, _, b = v3
    m = f"{b}.bse_noise"
    assert _claims(v3, "BSE graininess {" + m + ".this_image}; Batch_1 median above it at {"
                   + m + ".Batch_1.median}") == []


@needs_reference
def test_advisory_is_the_default_without_being_asked():
    """Every other test passes the mode explicitly; this one pins the default."""
    doc = LEGACY.read_text()
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(doc), KnowledgePack.empty(), baselines=baselines)
    client = FakeClient([text_response(comparison_payload(context)), _live_false_findings()])
    outcome = explain_facts_document(doc, KnowledgePack.empty(), client=client, baselines=baselines, max_retries=0)
    assert outcome.audit["critic_mode"] == "advisory"
    assert outcome.response.explanations
