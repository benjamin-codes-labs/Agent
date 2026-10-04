import copy
import json
from pathlib import Path

import pytest

from sentinel import KnowledgePack
from sentinel.pipeline_facts import (
    FactsResponse, explain_facts_document, parse_facts_document, prepare_facts_context, validate_facts_response,
)
from .conftest import FakeClient, critic_pass, text_response


def record(sample_id="img_dino"):
    return {"sample_id": sample_id, "predicted_batch": "Batch_1", "runner_up": "Batch_3",
            "confidence": {"tier": "High"},
            "decision": {"answer": "Batch_1", "answer_type": "specific", "confidence": "High"},
            "probabilities": {"dino_head": {"Batch_1": 0.022, "Batch_2": 0.047, "Batch_3": 0.931},
                              "fused_F": {"Batch_1": 0.9, "Batch_2": 0.05, "Batch_3": 0.05}},
            "features_S": {"siox_frac": {"value": 99.0}},
            "acquisition": {"warning": "OTHER_METRIC_MARKER"}}


def payload(context):
    output = []
    for sample in context.samples:
        sid = context.sample_sources[sample.sample_id]
        top = f"context.{sid}.dino_selection.top_probability"
        margin = f"context.{sid}.dino_selection.margin_to_next_pp"
        output.append({"sample_id": sample.sample_id, "reported_answer": sample.answer, "points": [
            {"kind": "basis", "text": "The highest DINO probability is {" + top + "}.", "facts": [top]},
            {"kind": "alternatives", "text": "The gap to the next-ranked class is {" + margin + "}.", "facts": [margin]},
            {"kind": "limitation", "text": "Class scores do not identify the visual features behind the prediction.",
             "citations": [{"source": context.guide_source,
                            "quote": "Class scores do not identify the visual features behind the prediction."}]},
        ]})
    return {"explanations": output}


def test_dino_argmax_overrides_other_decisions_without_modifying_input():
    original = record()
    before = copy.deepcopy(original)
    document = parse_facts_document(original, decision_mode="dino")
    sample = document.samples[0]
    assert sample.answer == "Batch_3"
    assert sample.confidence is None
    assert "DINO-only" in sample.header()
    assert "High" not in sample.header()
    assert original == before and document.data == original
    assert sample.data["decision"]["answer"] == "Batch_1"


def test_nested_material_dino_scores_are_supported():
    original = {"sample_id": "img_nested", "material_model": {"probabilities": {
        "dino_head": {"Batch_1": 0.8, "Batch_2": 0.1, "Batch_3": 0.1}}}}
    assert parse_facts_document(original, decision_mode="dino").samples[0].answer == "Batch_1"


@pytest.mark.parametrize("scores", [None, {}, {"Batch_1": 1.0},
    {"Batch_1": -0.1, "Batch_2": 1.1}, {"Batch_1": "0.9", "Batch_2": 0.1},
    {"Batch_1": True, "Batch_2": 0.0}, {"Batch_1": 0.1, "Batch_2": 0.2}])
def test_invalid_or_missing_dino_never_falls_back_to_other_metrics(scores):
    sample = record()
    sample["probabilities"]["dino_head"] = scores
    with pytest.raises(ValueError):
        parse_facts_document(sample, decision_mode="dino")
    sample["probabilities"].pop("dino_head")
    with pytest.raises(ValueError, match="dino_head"):
        parse_facts_document(sample, decision_mode="dino")


def test_conflicting_dino_locations_are_rejected():
    sample = record()
    sample["material_model"] = {"probabilities": {"dino_head": {"Batch_1": 0.8, "Batch_2": 0.1, "Batch_3": 0.1}}}
    with pytest.raises(ValueError, match="conflicting"):
        parse_facts_document(sample, decision_mode="dino")


def test_ties_are_not_broken_with_other_models():
    sample = record()
    sample["probabilities"]["dino_head"] = {"Batch_1": 0.45, "Batch_2": 0.45, "Batch_3": 0.10}
    parsed = parse_facts_document(sample, decision_mode="dino").samples[0]
    assert parsed.answer_type == "dino_tie"
    assert parsed.answer == "Batch_1 or Batch_2"
    assert parsed.confidence is None


def test_projection_contains_only_dino_data_and_computed_margins():
    context = prepare_facts_context(parse_facts_document(record(), decision_mode="dino"), KnowledgePack.empty())
    view = context.model_document["samples"][0]
    assert set(view) == {"sample_id", "probabilities", "dino_selection"}
    assert set(view["probabilities"]) == {"dino_head"}
    assert view["dino_selection"]["runner_up"] == "Batch_2"
    assert view["dino_selection"]["margin_to_next_pp"] == pytest.approx(88.4)
    assert view["dino_selection"]["alternatives"][1]["gap_from_top_pp"] == pytest.approx(90.9)
    assert "OTHER_METRIC_MARKER" not in context.pack.to_prompt_block()
    assert not any("features_S" in fid or "fused_F" in fid for fid in context.pack.context_facts)


def test_generation_and_critic_never_receive_other_metrics():
    requests = []
    for full_context in (False, True):
        original = record()
        if full_context:
            original["decision"] = None
            original["features_S"] = {"siox_frac": {"value": 1234.0}}
            original["acquisition"]["warning"] = "CHANGED_OTHER_METRIC"
        context = prepare_facts_context(parse_facts_document(original, decision_mode="dino"), KnowledgePack.empty(), full_context=full_context)
        client = FakeClient([text_response(payload(context)), critic_pass()])
        outcome = explain_facts_document(original, KnowledgePack.empty(), client=client, full_context=full_context,
                                         decision_mode="dino")
        assert outcome.samples[0].answer == "Batch_3"
        assert outcome.audit["classification_basis"] == "dino_head_only"
        sent = json.dumps(client.messages.requests, ensure_ascii=False)
        assert "OTHER_METRIC_MARKER" not in sent and "CHANGED_OTHER_METRIC" not in sent
        assert "1234" not in sent
        requests.append(client.messages.requests)
    assert requests[0] == requests[1]


def test_non_dino_facts_and_quotes_cannot_be_used():
    context = prepare_facts_context(parse_facts_document(record(), decision_mode="dino"), KnowledgePack.empty())
    bad = payload(context)
    sid = context.sample_sources["img_dino"]
    bad["explanations"][0]["points"][0].update(
        text="SiOx is {context." + sid + ".features_S.siox_frac.value}.",
        facts=[f"context.{sid}.features_S.siox_frac.value"],
        citations=[{"source": sid, "quote": "OTHER_METRIC_MARKER"}])
    assert validate_facts_response(FactsResponse.model_validate(bad), context)
    bad = payload(context)
    bad["explanations"][0]["reported_answer"] = "Batch_1"
    assert any("reported_answer" in error for error in validate_facts_response(FactsResponse.model_validate(bad), context))


def test_multi_sample_order_and_dict_format_are_preserved():
    from main import format_facts_response

    first, second = record("img_first"), record("img_second")
    second["probabilities"]["dino_head"] = {"Batch_1": 0.8, "Batch_2": 0.1, "Batch_3": 0.1}
    document = {"samples": [first, second]}
    before = copy.deepcopy(document)
    context = prepare_facts_context(parse_facts_document(document, decision_mode="dino"), KnowledgePack.empty())
    response = payload(context)
    response["explanations"].reverse()
    outcome = explain_facts_document(document, KnowledgePack.empty(), decision_mode="dino",
                                    client=FakeClient([text_response(response), critic_pass()]))
    values = format_facts_response(outcome.response.model_dump_json(), outcome.samples)
    assert list(values) == ["img_first", "img_second"]          # input order, not response order
    assert "93.1%" in values["img_first"] and "80%" in values["img_second"]
    # One string per image: a "- " bullet per line.
    assert all(isinstance(v, str) and all(line.startswith("- ") for line in v.split("\n"))
               for v in values.values())
    assert document == before


def test_main_dino_mode_does_not_require_other_knowledge_files(monkeypatch, capsys):
    import main

    monkeypatch.setattr(main.KnowledgePack, "from_project", lambda *a: pytest.fail("DINO classification must not load other metrics"))

    def explain(document, knowledge, **kwargs):
        context = prepare_facts_context(parse_facts_document(document, decision_mode="dino"), knowledge)
        return explain_facts_document(document, knowledge,
            client=FakeClient([text_response(payload(context)), critic_pass()]), **kwargs)

    monkeypatch.setattr(main, "explain_facts_document", explain)
    path = Path(main.ROOT) / "examples" / "facts.json"
    original = path.read_bytes()
    values = main.main(quiet=True, dino_only=True)
    assert json.loads(capsys.readouterr().out) == values
    assert list(values) == ["img_71vgq3fw"] and "93.1%" in values["img_71vgq3fw"]
    assert path.read_bytes() == original
