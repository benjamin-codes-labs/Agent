"""Batch baselines: code-computed comparisons the explainer narrates."""

import copy
import json
from pathlib import Path

import pytest

from sentinel.baselines import (
    BatchBaselines,
    baseline_display,
    is_comparable,
    normalise_sample_id,
    reported_batches,
)
from sentinel.knowledge import KnowledgePack
from sentinel.pipeline_facts import (
    explain_facts_document,
    parse_facts_document,
    prepare_facts_context,
    validate_facts_response,
    FactsResponse,
)
from .conftest import FakeClient, comparison_payload, critic_pass, prompt_text, text_response

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "lucas-sem-analysis-v3" / "outputs" / "batchid"


# --------------------------------------------------------------------------- #
# Synthetic reference set: pore separates the batches cleanly, SiOx does not
# --------------------------------------------------------------------------- #
def make_record(sample_id, batch, pore, siox=6.0, carbon=None, size=3.9,
                open_excess=7.0, aniso=1.06, decision=None):
    carbon = 100.0 - pore - siox if carbon is None else carbon
    record = {
        "sample_id": sample_id,
        "phases": {"three_class_pct": {"pore": pore, "carbon (graphite + binder)": carbon, "SiOx": siox}},
        "material_model": {"features_S": {
            "siox_ecd_aw": {"value": size, "robust_z": 0.0, "push_toward_winner_vs_runner_up": 0.0},
            "pore_open_excess": {"value": open_excess, "robust_z": 0.0, "push_toward_winner_vs_runner_up": 0.0},
            "pore_chord_aniso": {"value": aniso, "robust_z": 0.0, "push_toward_winner_vs_runner_up": 0.0},
        }},
        "decision": decision or {"answer": batch, "answer_type": "specific", "confidence": "High"},
    }
    if batch is not None:
        record["true_batch"] = batch
    return record


@pytest.fixture
def synthetic():
    records = []
    for batch, centre in (("Batch_1", 10.0), ("Batch_2", 20.0), ("Batch_3", 30.0)):
        for i, offset in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            records.append(make_record(f"{batch}_{i}", batch, centre + offset, siox=6.0 + offset * 0.1))
    return BatchBaselines(records, origin="synthetic")


def test_profiles_have_median_and_typical_range(synthetic):
    pore = synthetic.profiles()["pore_pct"]
    assert pore["Batch_2"].median == 20.0
    assert pore["Batch_2"].q1 == 19.5 and pore["Batch_2"].q3 == 20.5
    assert pore["Batch_2"].n == 5


def test_a_clean_match_supports_the_reported_batch(synthetic):
    record = make_record("new", None, 20.2, decision={"answer": "Batch_2"})
    pore = synthetic.compare(record)["measurements"]["pore_pct"]
    assert pore["verdict"] == "supports_reported"
    assert pore["inside_typical_range_of"] == ["Batch_2"]
    assert pore["nearest_median"] == "Batch_2"


def test_overlapping_ranges_do_not_discriminate(synthetic):
    record = make_record("new", None, 20.2, siox=6.0, decision={"answer": "Batch_2"})
    assert synthetic.compare(record)["measurements"]["siox_pct"]["verdict"] == "consistent_with_several"


def test_a_measurement_typical_of_another_batch_is_reported_as_opposing(synthetic):
    record = make_record("new", None, 30.1, decision={"answer": "Batch_2"})
    comparison = synthetic.compare(record)
    assert comparison["measurements"]["pore_pct"]["verdict"] == "favours_Batch_3"
    assert "pore_pct" in comparison["summary"]["favours_an_alternative"]


def test_an_atypical_value_is_not_dressed_up_as_support(synthetic):
    record = make_record("new", None, 45.0, decision={"answer": "Batch_3"})
    verdict = synthetic.compare(record)["measurements"]["pore_pct"]["verdict"]
    assert verdict == "closest_to_reported_but_atypical"


def test_pair_decisions_count_both_batches_as_reported(synthetic):
    record = make_record("new", None, 20.2, decision={
        "answer": "Batch_2 or Batch_3", "answer_type": "pair", "prediction_set": ["Batch_2", "Batch_3"]})
    comparison = synthetic.compare(record)
    assert comparison["reported_batches"] == ["Batch_2", "Batch_3"]
    assert comparison["measurements"]["pore_pct"]["verdict"] == "supports_reported"


def test_unsure_leaning_is_parsed():
    record = {"decision": {"answer": "unsure (leaning Batch_2)", "answer_type": "unsure"}}
    assert reported_batches(record) == ["Batch_2"]


def test_the_image_is_left_out_of_its_own_baseline(synthetic):
    """Never compare an image with a profile that contains itself."""
    member = copy.deepcopy(synthetic.records[7])                       # a Batch_2 record
    comparison = synthetic.compare(member)
    assert comparison["reference_set"]["this_image_left_out"] is True
    assert comparison["reference_set"]["per_batch_n"]["Batch_2"] == 4
    assert comparison["reference_set"]["per_batch_n"]["Batch_1"] == 5


def test_the_img_prefix_does_not_defeat_leave_one_out(synthetic):
    member = copy.deepcopy(synthetic.records[7])
    member["sample_id"] = "img_" + member["sample_id"]
    assert synthetic.compare(member)["reference_set"]["per_batch_n"]["Batch_2"] == 4
    assert normalise_sample_id("img_71vgq3fw") == normalise_sample_id("71vgq3fw")


def test_a_legacy_record_is_compared_on_imaging_only():
    """Different segmentation run: material values differ, raw-pixel probes do not."""
    legacy = json.loads((ROOT / "examples" / "facts.json").read_text())
    assert not is_comparable(legacy)
    refs = [make_record(f"r{i}", b, 10.0 + i) for i, b in enumerate(["Batch_1"] * 3 + ["Batch_2"] * 3)]
    for i, r in enumerate(refs):
        r["acquisition"] = {"probes": dict(legacy["acquisition"]["probes"], bse_noise_sigma=7.0 + i)}
    comparison = BatchBaselines(refs).compare(legacy)
    assert comparison["status"] == "compared" and comparison["scope"] == "imaging_only"
    assert {m["group"] for m in comparison["measurements"].values()} == {"imaging"}
    assert "pore_pct" not in comparison["measurements"]


def test_a_record_with_neither_is_not_compared():
    bare = {"sample_id": "x", "decision": {"answer": "Batch_1"}}
    comparison = BatchBaselines([make_record("r", "Batch_1", 10.0)] * 3).compare(bare)
    assert comparison["status"] == "not_comparable" and "measurements" not in comparison


def test_display_strings_carry_units():
    assert baseline_display("measurements.pore_pct.Batch_2.median", 16.37, "x") == "16.4%"
    assert baseline_display("measurements.siox_particle_size_um.this_image", 3.987, "x") == "3.99 µm"
    assert baseline_display("measurements.open_pore_excess_pp.this_image", 8.2, "x") == "8.2 pp"   # short units
    assert baseline_display("measurements.pore_anisotropy.this_image", 1.0625, "x") == "1.062"
    assert baseline_display("measurements.pore_pct.Batch_2.n", 7, "x") == "7"
    assert baseline_display("measurements.pore_pct.distance_to_median_in_iqr.Batch_2", 0.25, "x") == "+0.25 IQR"


# --------------------------------------------------------------------------- #
# The real reference set, when this checkout has it
# --------------------------------------------------------------------------- #
needs_reference = pytest.mark.skipif(not REFERENCE_DIR.is_dir(), reason="reference outputs not present")


@needs_reference
def test_real_reference_set_loads_all_three_batches():
    baselines = BatchBaselines.from_reference_dir(REFERENCE_DIR)
    assert baselines.batches == ["Batch_1", "Batch_2", "Batch_3"]
    assert len(baselines.records) == 31


@needs_reference
def test_demo_image_material_evidence_does_not_support_its_high_confidence_call():
    """The finding the baseline exists to surface: the call is not material-driven."""
    baselines = BatchBaselines.from_reference_dir(REFERENCE_DIR)
    record = json.loads((REFERENCE_DIR / "71vgq3fw.json").read_text())
    comparison = baselines.compare(record)
    assert record["decision"]["answer"] == "Batch_3" and record["decision"]["confidence"] == "High"
    assert comparison["reference_set"]["this_image_left_out"] is True
    material, imaging = comparison["summary_by_group"]["material"], comparison["summary_by_group"]["imaging"]
    assert material["supports_reported"] == []            # no material measurement singles out Batch_3
    assert material["favours_an_alternative"]
    assert set(imaging["supports_reported"]) >= {"bse_noise", "inlens_dark_level"}   # the imaging does


# --------------------------------------------------------------------------- #
# Pipeline integration and enforcement
# --------------------------------------------------------------------------- #
@pytest.fixture
def v3_document(synthetic):
    record = make_record("img_new", None, 30.1, decision={
        "answer": "Batch_2", "answer_type": "specific", "confidence": "Medium",
        "average_calibrated_probabilities": {"Batch_1": 0.1, "Batch_2": 0.6, "Batch_3": 0.3}})
    return record


def _context(document, baselines):
    return prepare_facts_context(parse_facts_document(document), KnowledgePack.empty(), baselines=baselines)


def _good_payload(context):
    """Two image-versus-batch comparisons, one of them the opposing measurement."""
    return comparison_payload(context)


def test_pipeline_adds_a_baseline_record_per_sample(v3_document, synthetic):
    context = _context(v3_document, synthetic)
    sid = context.baseline_sources["img_new"]
    assert context.has_baseline("img_new")
    assert sid in context.for_sample("img_new").sources
    fact = context.pack.context_facts[f"context.{sid}.measurements.pore_pct.Batch_3.median"]
    assert fact.display == "30.0%"


def test_dino_mode_never_receives_baselines(synthetic):
    record = make_record("img_new", None, 30.1)
    record["probabilities"] = {"dino_head": {"Batch_1": 0.1, "Batch_2": 0.2, "Batch_3": 0.7}}
    context = prepare_facts_context(parse_facts_document(record, decision_mode="dino"),
                                    KnowledgePack.empty(), baselines=synthetic)
    assert context.baselines == {} and context.baseline_sources == {}


def test_a_well_grounded_answer_passes(v3_document, synthetic):
    context = _context(v3_document, synthetic)
    response = FactsResponse.model_validate(_good_payload(context))
    assert validate_facts_response(response, context) == []


def test_an_answer_that_ignores_the_baseline_is_rejected(v3_document, synthetic):
    context = _context(v3_document, synthetic)
    payload = _good_payload(context)
    for point in payload["explanations"][0]["points"][:2]:
        point["text"], point["facts"] = "The model preferred this batch.", []
        point["citations"] = [{"source": context.baseline_sources["img_new"],
                               "quote": "Each batch profile rests on only a handful of images"}]
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    # Not every point must compare now, but at least two measurements must.
    assert any("compare at least 2 different measurements" in e for e in errors)


def test_opposing_evidence_cannot_be_dropped(v3_document, synthetic):
    """pore_pct favours Batch_3 here, so it must be mentioned."""
    context = _context(v3_document, synthetic)
    b = context.baseline_sources["img_new"]
    s = f"context.{b}.measurements.siox_pct"
    payload = _good_payload(context)
    payload["explanations"][0]["points"][0] = {
        "kind": "basis", "text": "SiOx area is {" + s + ".this_image}.", "facts": [f"{s}.this_image"]}
    payload["explanations"][0]["points"][1] = {
        "kind": "alternatives", "text": "SiOx overlaps every batch, median {" + s + ".Batch_3.median}.",
        "facts": [f"{s}.Batch_3.median"]}
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    assert any("opposing evidence must not be omitted" in e for e in errors)


def test_even_one_probability_is_rejected(v3_document, synthetic):
    """The page shows the probabilities; the explanation says WHY, not HOW SURE."""
    context = _context(v3_document, synthetic)
    own = context.sample_sources["img_new"]
    p = f"context.{own}.decision.average_calibrated_probabilities"
    payload = _good_payload(context)
    payload["explanations"][0]["points"][2] = {
        "kind": "limitation", "text": "Support is {" + p + ".Batch_2}.", "facts": [f"{p}.Batch_2"]}
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    assert any("not HOW SURE the model is" in e for e in errors)


def test_end_to_end_prompt_and_critic_carry_the_baseline(v3_document, synthetic):
    client = FakeClient([])
    context = _context(v3_document, synthetic)
    client.messages.script = [text_response(_good_payload(context)), critic_pass()]
    outcome = explain_facts_document(v3_document, KnowledgePack.empty(), client=client, baselines=synthetic)

    generation, critic = client.messages.requests
    prompt = prompt_text(generation)
    assert "Batch baseline comparisons (computed in code" in prompt
    assert "compare this image's numbers with the batches' numbers" in generation["system"]
    assert "batch_baselines_computed_in_code" in critic["messages"][0]["content"]

    audit = outcome.audit["batch_baselines"]["img_new"]
    assert audit["status"] == "compared"
    assert "pore_pct" in audit["favours_an_alternative"]
    text = " ".join(p.text for p in outcome.response.explanations[0].points)
    assert "30.1%" in text and "{" not in text


def test_without_baselines_nothing_changes(v3_document):
    context = _context(v3_document, None)
    assert context.baselines == {}
    response = FactsResponse.model_validate({"explanations": [{
        "sample_id": "img_new", "reported_answer": "Batch_2", "points": [
            {"kind": "basis", "text": "Reported pore area is {context." + context.sample_sources["img_new"]
             + ".phases.three_class_pct.pore}.",
             "facts": [f"context.{context.sample_sources['img_new']}.phases.three_class_pct.pore"]},
            {"kind": "alternatives", "text": "No batch profiles were supplied.",
             "citations": [{"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}]},
            {"kind": "limitation", "text": "Composition comes from segmentation.",
             "citations": [{"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}]},
        ]}]})
    assert not any("baseline" in e for e in validate_facts_response(response, context))


# --------------------------------------------------------------------------- #
# The comparison rule: this image's number against the batches' numbers
# --------------------------------------------------------------------------- #
from sentinel.pipeline_facts import compared_measurements


def test_a_comparison_needs_both_sides_of_the_same_measurement():
    s = "context.S5.measurements"
    assert compared_measurements([f"{s}.pore_pct.this_image", f"{s}.pore_pct.Batch_2.median"], "S5") == {"pore_pct"}
    assert compared_measurements([f"{s}.pore_pct.this_image", f"{s}.pore_pct.Batch_1.q1"], "S5") == {"pore_pct"}
    assert compared_measurements([f"{s}.pore_pct.this_image"], "S5") == set()          # image only
    assert compared_measurements([f"{s}.pore_pct.Batch_2.median"], "S5") == set()     # batch only
    # Two halves of different measurements are not a comparison of either.
    assert compared_measurements([f"{s}.pore_pct.this_image", f"{s}.siox_pct.Batch_2.median"], "S5") == set()
    # Another sample's baseline does not count.
    assert compared_measurements(["context.S9.measurements.pore_pct.this_image",
                                  "context.S9.measurements.pore_pct.Batch_2.median"], "S5") == set()


def _point(m, *keys):
    facts = [f"{m}.{k}" for k in keys]
    return {"text": " ".join("{" + f + "}" for f in facts) + " here.", "facts": facts}


def test_a_point_quoting_only_this_images_number_is_rejected(v3_document, synthetic):
    context = _context(v3_document, synthetic)
    b = context.baseline_sources["img_new"]
    payload = comparison_payload(context)
    payload["explanations"][0]["points"][0].update(_point(f"context.{b}.measurements.pore_pct", "this_image"))
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    assert any("compare at least 2 different measurements" in e for e in errors)


def test_a_point_quoting_only_batch_numbers_is_rejected(v3_document, synthetic):
    context = _context(v3_document, synthetic)
    b = context.baseline_sources["img_new"]
    payload = comparison_payload(context)
    payload["explanations"][0]["points"][1].update(
        _point(f"context.{b}.measurements.pore_pct", "Batch_1.median", "Batch_3.median"))
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    assert any("compare at least 2 different measurements" in e for e in errors)


def test_comparing_a_single_measurement_is_not_enough(v3_document, synthetic):
    context = _context(v3_document, synthetic)
    b = context.baseline_sources["img_new"]
    m = f"context.{b}.measurements.pore_pct"
    payload = comparison_payload(context)
    for point in payload["explanations"][0]["points"][:2]:
        point.update(_point(m, "this_image", "Batch_3.median"))
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    assert any("compare at least 2 different measurements" in e for e in errors)


def test_refused_answers_are_not_forced_into_comparisons(synthetic):
    record = make_record("img_ref", None, 20.0, decision={
        "answer": "refused (training image)", "answer_type": "refused", "reasons": ["This is a training image."]})
    context = _context(record, synthetic)
    sid = context.sample_sources["img_ref"]
    response = FactsResponse.model_validate({"explanations": [{
        "sample_id": "img_ref", "reported_answer": "refused (training image)", "points": [
            {"kind": "limitation", "text": "This is a training image, so no new classification is made.",
             "citations": [{"source": sid, "quote": "This is a training image."}]}]}]})
    assert not any("compare" in e for e in validate_facts_response(response, context))


def test_the_prompt_asks_for_side_by_side_numbers(v3_document, synthetic):
    client = FakeClient([])
    context = _context(v3_document, synthetic)
    client.messages.script = [text_response(comparison_payload(context)), critic_pass()]
    explain_facts_document(v3_document, KnowledgePack.empty(), client=client, baselines=synthetic)
    system = client.messages.requests[0]["system"]
    assert "Compare at least two different measurements this way" in system
    assert "lower_n" in system and "observed range" in system
    assert "Compare at least two different measurements" in system
