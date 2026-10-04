import copy
import hashlib
import json
from pathlib import Path

import pytest

from sentinel.explainer import ExplainerError
from sentinel.pipeline_facts import (
    FactsResponse, explain_facts_document, parse_facts_document, prepare_facts_context, validate_facts_response,
)
from .conftest import FakeClient, comparison_payload, critic_pass, evidence_facts, prompt_text, text_response
from .test_questions import project_knowledge


@pytest.fixture
def facts_document():
    record = json.loads((Path(__file__).resolve().parents[1] / "examples" / "facts.json").read_text())
    second = copy.deepcopy(record)
    second["sample_id"] = "img_test_second"
    second["predicted_batch"] = "Batch_1"
    second["runner_up"] = "Batch_3"
    second["features_S"]["siox_frac"]["value"] = 9.25
    return {"samples": [record, second], "run_id": "test-run"}


def _payload(context, reverse=False):
    results = []
    for sample in context.samples:
        sid = context.sample_sources[sample.sample_id]
        fid = f"context.{sid}.features_S.siox_frac.value"
        # Three kinds of evidence: measured feature, composition, evidence location.
        record = evidence_facts(context, sample.sample_id)
        comp, where = record.get("composition"), record.get("evidence location")
        basis_facts = [fid] + ([comp] if comp else [])
        alt_facts = [where] if where else []
        results.append({"sample_id": sample.sample_id, "reported_answer": sample.answer, "points": [
            {"kind": "basis", "text": "Reported SiOx area fraction is {" + fid + "}"
             + ("; pore share {" + comp + "}" if comp else "") + ".", "facts": basis_facts},
            {"kind": "alternatives", "text": "Comparable profiles are needed to exclude the alternatives"
             + ("; decisive-area share {" + where + "}" if where else "") + ".", "facts": alt_facts,
             "citations": [{"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}]},
            {"kind": "limitation", "text": "Acquisition conditions can confound the material interpretation.",
             "citations": [{"source": sid, "quote": "a prediction that matches the nearest-acquisition batch may reflect imaging session, not material"}]},
        ]})
    return {"explanations": list(reversed(results)) if reverse else results}


def test_example_has_original_structure_and_values():
    data = (Path(__file__).resolve().parents[1] / "examples" / "facts.json").read_bytes()
    assert hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == "8aefc83468a7659f39767e20d87de8acf3dda0caaa9d520c76017a0f9032943d"
    assert json.loads(data)["sample_id"] == "img_71vgq3fw"
    assert "samples" not in json.loads(data)


@pytest.mark.parametrize("shape", ["single", "list", "samples", "results", "keyed"])
def test_accepts_native_records_without_mutating_them(facts_document, shape):
    records = facts_document["samples"]
    payload = {"single": records[0], "list": records, "samples": facts_document,
               "results": {"results": records}, "keyed": {r["sample_id"]: r for r in records}}[shape]
    before = copy.deepcopy(payload)
    parsed = parse_facts_document(payload)
    assert len(parsed.samples) == (1 if shape == "single" else 2)
    assert payload == before
    assert parsed.samples[0].data == records[0]


def test_final_decision_overrides_material_and_legacy_picks():
    payload = {"sample_id": "img_new", "predicted_batch": "Batch_3", "confidence": {"tier": "High"},
               "decision": {"answer": "Batch_1 or Batch_2", "answer_type": "pair", "confidence": "Medium"},
               "material_model": {"predicted_batch": "Batch_3"}, "forced_mode": {"answer": "Batch_1"}}
    record = parse_facts_document(payload).samples[0]
    assert record.answer == "Batch_1 or Batch_2"
    assert record.confidence == "Medium"
    assert record.answer_type == "pair"


@pytest.mark.parametrize("bad", [[], {}, {"samples": []}, {"sample_id": "x", "decision": None},
                                  {"sample_id": "x"}, {"samples": [], "results": []}])
def test_invalid_documents_fail_before_api(bad):
    with pytest.raises(ValueError):
        parse_facts_document(bad)


def test_duplicate_ids_and_duplicate_json_keys_are_rejected(facts_document):
    with pytest.raises(ValueError, match="duplicate sample_id"):
        parse_facts_document([facts_document["samples"][0]] * 2)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_facts_document('{"sample_id":"one","sample_id":"two","predicted_batch":"Batch_1"}')


def test_units_and_meanings_do_not_change_raw_measurements(project_knowledge, facts_document):
    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    sid = context.sample_sources["img_71vgq3fw"]
    facts = context.pack.context_facts
    assert facts[f"context.{sid}.features_S.siox_frac.value"].display == "6.47%"
    assert facts[f"context.{sid}.features_S.siox_ecd_aw.value"].display == "3.85 µm"
    assert facts[f"context.{sid}.evidence.top_regions.0.evidence_share"].display == "0.4%"
    assert facts[f"context.{sid}.features_S.pore_open_excess.push_toward_winner_vs_runner_up"].value == -0.264
    assert context.samples[0].data == facts_document["samples"][0]
    guide = context.pack.sources[context.guide_source].body
    assert "median" in guide and "IQR" in guide
    assert "decision.answer" in guide and "does not vote" in guide
    assert "0.859" not in guide


def test_whole_document_is_sent_together_and_order_is_restored(project_knowledge, facts_document):
    original = json.dumps(facts_document, indent=2)
    context = prepare_facts_context(parse_facts_document(original), project_knowledge)
    client = FakeClient([text_response(_payload(context, reverse=True)), critic_pass()])
    outcome = explain_facts_document(original, project_knowledge, client=client)
    assert len(client.messages.requests) == 2
    generation = client.messages.requests[0]
    assert original in prompt_text(generation)
    assert all(block["type"] == "text" for block in generation["messages"][0]["content"])
    assert [item.sample_id for item in outcome.response.explanations] == [r["sample_id"] for r in facts_document["samples"]]
    assert outcome.audit["critic_passed"] is True
    assert outcome.audit["sample_count"] == 2
    assert outcome.samples[0].data == facts_document["samples"][0]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_answer", "foreign_fact", "foreign_citation"])
def test_incomplete_or_cross_sample_output_is_rejected(project_knowledge, facts_document, failure):
    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    payload = _payload(context)
    first, second = payload["explanations"]
    if failure == "missing":
        payload["explanations"] = [first]
    elif failure == "duplicate":
        payload["explanations"] = [first, first]
    elif failure == "wrong_answer":
        first["reported_answer"] = "Batch_2"
    elif failure == "foreign_fact":
        first["points"][0] = copy.deepcopy(second["points"][0])
    else:
        first["points"][2] = copy.deepcopy(second["points"][2])
    assert validate_facts_response(FactsResponse.model_validate(payload), context)


def test_retries_the_whole_answer_and_never_returns_a_partial_list(project_knowledge, facts_document):
    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    partial = _payload(context)
    partial["explanations"].pop()
    client = FakeClient([text_response(partial), text_response(_payload(context)), critic_pass()])
    outcome = explain_facts_document(facts_document, project_knowledge, client=client)
    assert outcome.audit["attempts"] == 2
    assert len(outcome.response.explanations) == 2
    with pytest.raises(ExplainerError):
        explain_facts_document(facts_document, project_knowledge,
                               client=FakeClient([text_response(partial)]), max_retries=0)


def _with_reasoning_words(context, index, count):
    from sentinel.questions import _display_text

    payload = _payload(context)
    item = payload["explanations"][index]
    local = context.for_sample(item["sample_id"])
    current = sum(len(_display_text(point["text"], local).split()) for point in item["points"])
    assert count >= current
    item["points"][0]["text"] += " observation" * (count - current)
    return payload


def _spread_reasoning_words(context, index, count):
    """Pad to `count` reasoning words spread over short points.

    Points are capped at HARD_POINT_WORDS each, so a total above the target has
    to come from more points, not one long one.
    """
    from sentinel.pipeline_facts import MAX_POINTS, MAX_POINT_WORDS
    from sentinel.questions import _display_text

    payload = _payload(context)
    item = payload["explanations"][index]
    local = context.for_sample(item["sample_id"])
    quote = {"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}
    while len(item["points"]) < MAX_POINTS:
        item["points"].append({"kind": "limitation", "text": "Further observation.", "citations": [quote]})
    for point in item["points"]:
        words = len(_display_text(point["text"], local).split())
        total = sum(len(_display_text(q["text"], local).split()) for q in item["points"])
        point["text"] += " observation" * max(0, min(MAX_POINT_WORDS - words, count - total))
    return payload


def test_small_word_overrun_does_not_reject_grounded_output(project_knowledge, facts_document):
    """Above the 110-word target but under the ceiling, in short points: accepted."""
    from sentinel.pipeline_facts import TARGET_REASONING_WORDS
    from sentinel.questions import _display_text

    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    payload = _spread_reasoning_words(context, 0, 140)
    local = context.for_sample(payload["explanations"][0]["sample_id"])
    total = sum(len(_display_text(p["text"], local).split()) for p in payload["explanations"][0]["points"])
    assert total > TARGET_REASONING_WORDS
    assert validate_facts_response(FactsResponse.model_validate(payload), context) == []
    client = FakeClient([text_response(payload), critic_pass()])
    outcome = explain_facts_document(facts_document, project_knowledge, client=client)
    assert outcome.audit["attempts"] == 1
    assert len(client.messages.requests) == 2


def test_manifest_shows_exact_heading_and_available_word_budget(project_knowledge, facts_document):
    facts_document["samples"][1]["confidence"]["tier"] = "Medium with an uncertain acquisition match"
    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    client = FakeClient([text_response(_payload(context)), critic_pass()])
    explain_facts_document(facts_document, project_knowledge, client=client)
    content = prompt_text(client.messages.requests[0])
    manifest_text = content.split("# Authoritative sample/source manifest\n", 1)[1].split("\n\n# Complete input", 1)[0]
    manifest = json.loads(manifest_text)
    for entry, sample in zip(manifest, context.samples):
        budget = entry["word_budget"]
        assert budget["heading"] == sample.header()
        assert budget["max_reasoning_words"] + budget["heading_words"] == 280
        assert budget["target_reasoning_words"] <= 150
        assert budget["target_reasoning_words"] < budget["max_reasoning_words"]
    assert manifest[0]["word_budget"]["max_reasoning_words"] > manifest[1]["word_budget"]["max_reasoning_words"]


def test_overlong_repair_targets_safe_margin_without_truncating(project_knowledge, facts_document):
    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    payload = _with_reasoning_words(context, 0, 350)
    draft = FactsResponse.model_validate(payload)
    before = draft.model_dump_json()
    assert any("words" in error for error in validate_facts_response(draft, context))
    assert draft.model_dump_json() == before
    client = FakeClient([text_response(payload), text_response(_payload(context)), critic_pass()])
    outcome = explain_facts_document(facts_document, project_knowledge, client=client)
    feedback = client.messages.requests[1]["messages"][-1]["content"]
    assert '"current_reasoning_words": 350' in feedback
    assert '"target_reasoning_words": 95' in feedback
    assert "after placeholder substitution" in feedback
    assert "Do not remove opposing evidence" in feedback
    assert outcome.audit["attempts"] == 2
    assert len(outcome.response.explanations) == 2


def test_relaxed_style_budget_does_not_relax_fact_ownership(project_knowledge, facts_document):
    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    payload = _with_reasoning_words(context, 0, 250)
    foreign = payload["explanations"][1]["points"][0]["facts"][0]
    own = payload["explanations"][0]["points"][0]["facts"][0]
    point = payload["explanations"][0]["points"][0]
    point["text"] = point["text"].replace(own, foreign)
    point["facts"] = [foreign]
    assert any("unknown fact" in error for error in validate_facts_response(FactsResponse.model_validate(payload), context))


def test_authentication_failure_stops_without_retry_or_secret_echo(project_knowledge, facts_document):
    class RejectedCredential(Exception):
        status_code = 401

    def fail(_):
        raise RejectedCredential("private provider details")

    client = FakeClient([fail])
    with pytest.raises(ExplainerError, match="HTTP 401") as error:
        explain_facts_document(facts_document, project_knowledge, client=client)
    assert len(client.messages.requests) == 1
    assert "private provider details" not in str(error.value)


def test_refused_decision_is_not_replaced_by_component_prediction(project_knowledge):
    from main import format_facts_response

    record = {"sample_id": "img_refused", "predicted_batch": "Batch_3",
              "decision": {"answer": "refused (training image)", "answer_type": "refused",
                           "confidence": "n/a", "reasons": ["This is a training image."]}}
    context = prepare_facts_context(parse_facts_document(record), project_knowledge)
    sid = context.sample_sources["img_refused"]
    payload = {"explanations": [{"sample_id": "img_refused", "reported_answer": "refused (training image)", "points": [
        {"kind": "limitation", "text": "This is a training image, not an independent material classification.",
         "citations": [{"source": sid, "quote": "This is a training image."}]}]}]}
    outcome = explain_facts_document(record, project_knowledge,
                                    client=FakeClient([text_response(payload), critic_pass()]))
    output = format_facts_response(outcome.response.model_dump_json(), outcome.samples)
    # format_facts_response raises if the decision changed; the text must not claim the component pick.
    assert list(output) == ["img_refused"]
    assert "Batch_3" not in " ".join(output["img_refused"])


def test_malformed_critic_verdict_never_passes(project_knowledge, facts_document):
    from .conftest import tool_response

    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    client = FakeClient([text_response(_payload(context)), tool_response("report_grounding", {}),
                         tool_response("report_grounding", {})])
    # The critic blocks only in blocking mode; advisory mode is covered in test_speed.py.
    with pytest.raises(ExplainerError, match="critic unavailable"):
        explain_facts_document(facts_document, project_knowledge, client=client, critic_mode="blocking")


def test_truncated_response_never_returns_prefix_of_results(project_knowledge, facts_document):
    from .conftest import FakeResponse, FakeTextBlock

    client = FakeClient([lambda _: FakeResponse(content=[FakeTextBlock('{"explanations":[')], stop_reason="max_tokens")])
    with pytest.raises(ExplainerError, match="no incomplete list"):
        explain_facts_document(facts_document, project_knowledge, client=client)
    assert len(client.messages.requests) == 1


def test_guide_format_decision_ranges_and_maps_are_kept_intact(project_knowledge):
    record = {"sample_id": "img_v3", "decision": {"answer": "Batch_1 or Batch_2", "answer_type": "pair", "confidence": "Medium"},
              "material_model": {"predicted_batch": "Batch_3", "map_text": {"sentences": ["SiOx is enriched centrally."]}},
              "texture_model": {"used_in_decision": False},
              "material_range_rule": {"answer": None, "phases": [{"phase": "pore", "fits_ranges_of": ["Batch_1", "Batch_2"]}]},
              "phases": {"three_class_pct": {"SiOx": 6.4}, "reliability": "The binder split is experimental."}}
    original = copy.deepcopy(record)
    context = prepare_facts_context(parse_facts_document(record), project_knowledge)
    assert context.samples[0].answer == "Batch_1 or Batch_2"
    assert context.samples[0].data == original
    source = context.pack.sources[context.sample_sources["img_v3"]]
    assert json.loads(source.body) == original
    assert "used_in_decision" in source.body and "fits_ranges_of" in source.body


def test_facts_mode_rejects_raw_image_attachments(monkeypatch, project_knowledge):
    import main

    monkeypatch.setattr(main.KnowledgePack, "from_project", lambda root: project_knowledge)
    with pytest.raises(ValueError, match="text-only"):
        main.main(facts="examples/facts.json", images=["BSE=anything.tif"])


def test_parser_returns_one_string_per_image_keyed_by_id(project_knowledge, facts_document):
    from main import format_facts_response

    context = prepare_facts_context(parse_facts_document(facts_document), project_knowledge)
    outcome = explain_facts_document(facts_document, project_knowledge,
                                    client=FakeClient([text_response(_payload(context)), critic_pass()]))
    values = format_facts_response(outcome.response.model_dump_json(), outcome.samples)
    assert list(values) == ["img_71vgq3fw", "img_test_second"]
    first, second = values["img_71vgq3fw"], values["img_test_second"]
    assert isinstance(first, str) and isinstance(second, str)
    assert "6.47%" in first and "9.25%" not in first
    assert "9.25%" in second and "6.47%" not in second
    # Three points -> three lines, each a "- " bullet, no blank lines.
    assert all(len(v.split("\n")) == 3 and all(l.startswith("- ") for l in v.split("\n"))
               for v in values.values())
    assert "\\n" in json.dumps(values)            # newlines are escaped in the JSON output


def test_default_main_reads_only_facts_example_and_returns_a_dict(project_knowledge, monkeypatch, capsys):
    import main

    monkeypatch.setattr(main.KnowledgePack, "from_project", lambda root: project_knowledge)
    calls = []

    def explain(document, pack, **kwargs):
        calls.append(document)
        context = prepare_facts_context(parse_facts_document(document), pack, baselines=kwargs.get("baselines"))
        return explain_facts_document(document, pack,
            client=FakeClient([text_response(comparison_payload(context)), critic_pass()]), **kwargs)

    monkeypatch.setattr(main, "explain_facts_document", explain)
    values = main.main()
    captured = capsys.readouterr()
    assert isinstance(values, dict) and list(values) == ["img_71vgq3fw"]
    assert json.loads(captured.out) == values
    assert len(calls) == 1 and json.loads(calls[0])["sample_id"] == "img_71vgq3fw"
    assert captured.err == ""
