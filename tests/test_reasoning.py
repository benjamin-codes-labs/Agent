"""Explanations say WHY an image is in its category, not HOW SURE a model is,
and a facts file with N images yields exactly N explanations, in order."""

import json
from pathlib import Path

import pytest

from sentinel.knowledge import KnowledgePack
from sentinel.pipeline_facts import (
    REASONING_RULES,
    FactsResponse,
    explain_facts_document,
    is_model_score_fact,
    parse_facts_document,
    prepare_facts_context,
    validate_facts_response,
)
from sentinel.baselines import BatchBaselines
from .conftest import FakeClient, comparison_payload, critic_pass, evidence_facts, text_response

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "examples" / "facts.json"
REFERENCE_DIR = ROOT / "lucas-sem-analysis-v3" / "outputs" / "batchid"
needs_reference = pytest.mark.skipif(not REFERENCE_DIR.is_dir(), reason="reference outputs not present")


def _legacy_context():
    return prepare_facts_context(parse_facts_document(LEGACY.read_text()), KnowledgePack.empty())


def _breadth_point(context, sample_id, already):
    """An alternatives point adding the record's other kinds of evidence.

    Tests here are about the basis point; this keeps them clear of the breadth
    rule (at least three kinds of evidence) without changing what they test.
    """
    from sentinel.pipeline_facts import _evidence_kinds
    have = _evidence_kinds(already, context, sample_id)
    extra = [fid for kind, fid in evidence_facts(context, sample_id).items() if kind not in have]
    text = "Nothing in the composition rules out the runner-up" + "".join(
        "; record figure {" + fid + "}" for fid in extra) + "."
    return {"kind": "alternatives", "text": text, "facts": extra,
            "citations": [{"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}]}


def _legacy_answer(context, basis_text, basis_facts):
    sample = context.samples[0]
    return FactsResponse.model_validate({"explanations": [{
        "sample_id": sample.sample_id, "reported_answer": sample.answer, "points": [
            {"kind": "basis", "text": basis_text, "facts": basis_facts},
            _breadth_point(context, sample.sample_id, basis_facts),
            {"kind": "limitation", "text": "No batch comparison was available for this record.",
             "citations": [{"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}]},
        ]}]})


# --------------------------------------------------------------------------- #
# Which facts count as model scores
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fid", [
    "context.S9.probabilities.fused_F.Batch_3",
    "context.S9.probabilities.dino_head.Batch_3",
    "context.S9.decision.average_calibrated_probabilities.Batch_3",
    "context.S9.features_S.siox_frac.push_toward_winner_vs_runner_up",
    "context.S9.fingerprint_model.top_measurements.0.push_toward_winner",
    "context.S9.validation_reference.LOGO_F",
    "context.S9.decision.track_record.LOO.this_confidence_tier.accuracy",
    "context.S9.decision.high_tier_threshold",
    "context.S9.loo_prediction.p_F.2",
    "context.S9.evidence.reconstruction_error",
])
def test_model_scores_are_recognised(fid):
    assert is_model_score_fact(fid)


@pytest.mark.parametrize("fid", [
    "context.S9.features_S.siox_frac.value",
    "context.S9.features_S.siox_frac.robust_z",
    "context.S9.evidence.phase_share_in_top10pct_evidence_pct.SiOx",
    "context.S9.phase_fractions_pct.pore",
    "context.S9.material_range_rule.phases.0.ranges.Batch_1.0",
    "context.S12.measurements.pore_pct.Batch_3.median",
])
def test_characteristics_are_not_model_scores(fid):
    assert not is_model_score_fact(fid)


# --------------------------------------------------------------------------- #
# The regression: a legacy record explained entirely by its probabilities
# --------------------------------------------------------------------------- #
def test_legacy_record_cannot_be_explained_by_probabilities():
    """The cap used to live inside the baseline checks, which skip legacy records."""
    context = _legacy_context()
    own = context.sample_sources[context.samples[0].sample_id]
    fused = f"context.{own}.probabilities.fused_F.Batch_3"
    response = _legacy_answer(context, "Fused support is {" + fused + "}.", [fused])
    errors = validate_facts_response(response, context)
    assert any("not HOW SURE the model is" in e for e in errors)


def test_push_values_are_not_reasons_either():
    context = _legacy_context()
    own = context.sample_sources[context.samples[0].sample_id]
    push = f"context.{own}.features_S.siox_frac.push_toward_winner_vs_runner_up"
    errors = validate_facts_response(_legacy_answer(context, "SiOx pushes by {" + push + "}.", [push]), context)
    assert any(push in e for e in errors)


def test_naming_model_branches_without_numbers_is_still_rejected():
    """'The DINO branch favours it' is a model score offered as a reason, in words."""
    context = _legacy_context()
    own = context.sample_sources[context.samples[0].sample_id]
    siox = f"context.{own}.features_S.siox_frac.value"
    errors = validate_facts_response(_legacy_answer(
        context, "SiOx covers {" + siox + "} and the DINO branch strongly favours this batch.", [siox]), context)
    assert any("model branches" in e and "DINO branch" in e for e in errors)


def test_characteristic_reasons_pass_on_a_legacy_record():
    context = _legacy_context()
    own = context.sample_sources[context.samples[0].sample_id]
    top = f"context.{own}.evidence.phase_share_in_top10pct_evidence_pct.SiOx"
    whole = f"context.{own}.evidence.phase_share_whole_image_pct.SiOx"
    response = _legacy_answer(
        context,
        "The area that drove the call holds {" + top + "} SiOx, against {" + whole + "} across the whole image, "
        "so the decisive evidence lies away from the SiOx particles.",
        [top, whole])
    assert validate_facts_response(response, context) == []


def test_evidence_map_wording_is_a_reason_not_a_score():
    context = _legacy_context()
    own = context.sample_sources[context.samples[0].sample_id]
    top = f"context.{own}.evidence.phase_share_in_top10pct_evidence_pct.graphite"
    response = _legacy_answer(context, "In the DINOv2 evidence map, graphite takes {" + top + "} of the decisive area.", [top])
    assert not any("model branches" in e for e in validate_facts_response(response, context))


def test_dino_mode_is_exempt_from_the_ban():
    """DINO-only narration is about the score vector by definition."""
    record = json.loads(LEGACY.read_text())
    context = prepare_facts_context(parse_facts_document(record, decision_mode="dino"), KnowledgePack.empty())
    sample = context.samples[0]
    own = context.sample_sources[sample.sample_id]
    fid = f"context.{own}.probabilities.dino_head.Batch_3"
    assert fid in context.pack.context_facts
    response = FactsResponse.model_validate({"explanations": [{
        "sample_id": sample.sample_id, "reported_answer": sample.answer, "points": [
            {"kind": "basis", "text": "The top score is {" + fid + "}.", "facts": [fid]},
            {"kind": "alternatives", "text": "The other scores are lower.", "facts": [fid.replace("Batch_3", "Batch_2")],
             },
            {"kind": "limitation", "text": "Scores do not show visual features.", "facts": [fid.replace("Batch_3", "Batch_1")]},
        ]}]})
    errors = validate_facts_response(response, context)
    assert not any("HOW SURE" in e or "model branches" in e for e in errors)


def test_reasoning_rules_reach_the_pipeline_prompt_but_not_dino():
    record = LEGACY.read_text()
    context = _legacy_context()
    sample = context.samples[0]
    payload = _legacy_answer(context, "SiOx covers {context." + context.sample_sources[sample.sample_id]
                             + ".features_S.siox_frac.value}.",
                             [f"context.{context.sample_sources[sample.sample_id]}.features_S.siox_frac.value"])
    client = FakeClient([text_response(payload.model_dump()), critic_pass()])
    explain_facts_document(record, KnowledgePack.empty(), client=client)
    assert "WHY, not HOW SURE" in client.messages.requests[0]["system"]
    assert REASONING_RULES.split("\n")[0] in client.messages.requests[0]["system"]


# --------------------------------------------------------------------------- #
# N images in, N explanations out, in input order
# --------------------------------------------------------------------------- #
def _baseline_payload(context):
    return comparison_payload(context)


def _one_of_each_answer_type():
    picked, seen = [], set()
    for path in sorted(REFERENCE_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        kind = record["decision"].get("answer_type")
        if kind not in seen:
            seen.add(kind)
            picked.append(record)
    return picked


@needs_reference
def test_one_list_element_per_image_in_input_order(monkeypatch, capsys, tmp_path):
    import main

    records = _one_of_each_answer_type()
    assert len(records) >= 3                       # specific, pair and unsure
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(records))

    def explain(document, pack, **kwargs):
        context = prepare_facts_context(parse_facts_document(document), pack, baselines=kwargs.get("baselines"))
        return explain_facts_document(document, pack,
            client=FakeClient([text_response(_baseline_payload(context)), critic_pass()]), **kwargs)

    monkeypatch.setattr(main, "explain_facts_document", explain)
    monkeypatch.setattr(main.KnowledgePack, "from_project", lambda root: KnowledgePack.empty())
    values = main.main(facts=path, quiet=True)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == values
    assert captured.err == ""                                    # quiet by default: result only
    assert list(values) == [r["sample_id"] for r in records]    # one key per image, input order
    for record in records:
        value = values[record["sample_id"]]
        assert isinstance(value, str)                               # one string per image
        lines = value.split("\n")
        assert len(lines) == 3 and all(line.startswith("- ") for line in lines)
        others = [r["sample_id"] for r in records if r is not record]
        assert not any(other in value for other in others)          # no cross-talk


@needs_reference
def test_each_image_gets_its_own_left_out_baseline():
    records = _one_of_each_answer_type()
    baselines = BatchBaselines.from_reference_dir(REFERENCE_DIR)
    context = prepare_facts_context(parse_facts_document(records), KnowledgePack.empty(), baselines=baselines)
    assert len(set(context.baseline_sources.values())) == len(records)
    for record in records:
        ref = context.baselines[record["sample_id"]]["reference_set"]
        assert ref["this_image_left_out"] is True
        assert ref["per_batch_n"][record["true_batch"]] == sum(
            1 for r in baselines.records if r["true_batch"] == record["true_batch"]) - 1


# --------------------------------------------------------------------------- #
# Regression: the critic judged characteristic wording against probabilities
# --------------------------------------------------------------------------- #
def _critic_request(record, decision_mode="pipeline"):
    context = prepare_facts_context(parse_facts_document(record, decision_mode=decision_mode), KnowledgePack.empty())
    sample = context.samples[0]
    own = context.sample_sources[sample.sample_id]
    if decision_mode == "dino":
        fid = f"context.{own}.probabilities.dino_head.Batch_3"
        points = [{"kind": k, "text": "The score is {" + fid + "}.", "facts": [fid]}
                  for k in ("basis", "alternatives", "limitation")]
    else:
        siox = f"context.{own}.features_S.siox_frac.value"
        quote = {"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}
        points = [
            {"kind": "basis", "text": "The measurements favour this batch only weakly: SiOx covers {" + siox + "}.",
             "facts": [siox]},
            _breadth_point(context, sample.sample_id, [siox]),
            {"kind": "limitation", "text": "No batch comparison was available.", "citations": [quote]},
        ]
    payload = {"explanations": [{"sample_id": sample.sample_id, "reported_answer": sample.answer, "points": points}]}
    client = FakeClient([text_response(payload), critic_pass()])
    explain_facts_document(record, KnowledgePack.empty(), client=client, decision_mode=decision_mode)
    return client.messages.requests[1]


def test_critic_never_sees_probabilities_in_pipeline_mode():
    """It rejected an honest 'only weakly' for understating a 0.707 fused probability."""
    request = _critic_request(LEGACY.read_text())
    content = request["messages"][0]["content"]
    document = json.loads(content)["pipeline_document"]
    assert "probabilities" not in document
    assert "validation_reference" not in document
    assert all("push_toward_winner_vs_runner_up" not in f for f in document["features_S"].values())
    assert document["features_S"]["siox_frac"]["value"] == 6.469     # characteristics are kept
    assert "0.707" not in content


def test_critic_is_told_weak_support_is_honest_not_understatement():
    system = _critic_request(LEGACY.read_text())["system"]
    assert "Never object that wording understates the model's confidence" in system
    assert "OVERSTATING" in system


def test_dino_critic_still_sees_the_scores_it_checks():
    content = _critic_request(LEGACY.read_text(), decision_mode="dino")["messages"][0]["content"]
    assert "dino_head" in content


# --------------------------------------------------------------------------- #
# Regression: an advisory critic finding must not block delivery
# --------------------------------------------------------------------------- #
from sentinel.critic import ClaudeCritic, CriticConfig
from sentinel.explainer import ModelCall
from .conftest import critic_fail, tool_response


def _verdict(step, blocking_kinds=("A", "B", "D", "E")):
    critic = ClaudeCritic(FakeClient([step]), CriticConfig(model="fake", blocking_kinds=blocking_kinds))
    return critic._read(step(None), ModelCall(model="fake"))


def test_an_advisory_only_objection_ships_but_is_recorded():
    """blocking_kinds used to be ignored: any finding failed the run."""
    verdict = _verdict(critic_fail("C", "quote is about the category of imaging cues"))
    assert verdict.passed is True                         # delivery not blocked
    assert verdict.declared_passed is False               # the objection is on record
    assert [f.severity for f in verdict.findings] == ["warning"]


def test_a_blocking_objection_still_blocks():
    verdict = _verdict(critic_fail("E", "contradicts the fact sheet"))
    assert verdict.passed is False
    assert [f.severity for f in verdict.findings] == ["error"]


def test_a_mixed_verdict_blocks_on_the_blocking_finding():
    step = tool_response("report_grounding", {"passed": False, "problems": [
        {"kind": "C", "loc": "points[1]", "quote": "x", "why": "weak quote"},
        {"kind": "B", "loc": "points[0]", "quote": "y", "why": "claims proof"}]})
    verdict = _verdict(step)
    assert verdict.passed is False
    assert sorted(f.severity for f in verdict.findings) == ["error", "warning"]


def test_an_unexplained_rejection_still_blocks():
    verdict = _verdict(tool_response("report_grounding", {"passed": False, "problems": []}))
    assert verdict.passed is False


def _legacy_run(critic_step):
    record = LEGACY.read_text()
    from sentinel.baselines import load_default_baselines
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(record), KnowledgePack.empty(), baselines=baselines)
    client = FakeClient([text_response(comparison_payload(context)), critic_step])
    return explain_facts_document(record, KnowledgePack.empty(), client=client, baselines=baselines)


@needs_reference
def test_the_exact_rejection_from_the_live_run_no_longer_kills_the_file():
    """Kind C on a caveat citation: delivered, with the objection in the audit."""
    outcome = _legacy_run(critic_fail(
        "C", "This quote is a general caveat about imaging measurements ... it does not justify why this "
             "particular dark-level measurement simultaneously supports a batch classification"))
    assert outcome.audit["critic_passed"] is False
    assert len(outcome.audit["critic_advisory_findings"]) == 1
    assert "CRIT_QUOTE" in outcome.audit["critic_advisory_findings"][0]


@needs_reference
def test_a_contradiction_finding_still_forces_a_repair():
    from sentinel.explainer import ExplainerError
    with pytest.raises(ExplainerError, match="after 1 attempt"):
        record = LEGACY.read_text()
        from sentinel.baselines import load_default_baselines
        baselines = load_default_baselines(ROOT)
        context = prepare_facts_context(parse_facts_document(record), KnowledgePack.empty(), baselines=baselines)
        client = FakeClient([text_response(comparison_payload(context)), critic_fail("E", "wrong number")])
        explain_facts_document(record, KnowledgePack.empty(), client=client, baselines=baselines, max_retries=0,
                               critic_mode="blocking")


def test_the_critic_is_told_caveat_citations_need_not_justify_numbers():
    from sentinel.pipeline_facts import CRITIC_REASONING_POLICY, FACTS_CRITIC_BLOCKING_KINDS
    assert "Do not read a caveat citation as the justification" in CRITIC_REASONING_POLICY
    assert "C" not in FACTS_CRITIC_BLOCKING_KINDS and {"A", "B", "D", "E"} <= set(FACTS_CRITIC_BLOCKING_KINDS)
