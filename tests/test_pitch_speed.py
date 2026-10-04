import copy
import json
from pathlib import Path

import pytest

from sentinel.baselines import load_default_baselines
from sentinel.explainer import ExplainerConfig
from sentinel.knowledge import KnowledgePack
from sentinel.pipeline_facts import (
    FactsResponse, _baseline_table, _fact_aliases, _expand_fact_aliases,
    explain_facts_document, parse_facts_document, prepare_facts_context, validate_facts_response,
)
from sentinel.response_cache import ResponseCache, response_cache_key
from .conftest import FakeClient, comparison_payload, critic_pass, text_response


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case():
    text = (ROOT / "examples" / "facts.json").read_text()
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(text), KnowledgePack.empty(), baselines=baselines)
    return text, baselines, context


def _aliased_response(context):
    aliases = _fact_aliases(context)
    payload = comparison_payload(context)
    for item in payload["explanations"]:
        for point in item["points"]:
            for fid in point.get("facts", []):
                point["text"] = point["text"].replace("{" + fid + "}", "{" + aliases[fid] + "}")
            point.pop("facts", None)
    return payload


def test_short_aliases_reduce_output_without_changing_checked_claims(case):
    _, _, context = case
    full = FactsResponse.model_validate(comparison_payload(context))
    short = FactsResponse.model_validate(_aliased_response(context))
    expanded = _expand_fact_aliases(short, _fact_aliases(context))
    assert expanded == full
    assert len(short.model_dump_json()) < len(full.model_dump_json())
    assert validate_facts_response(expanded, context) == []
    assert short != expanded


def test_alias_table_identifies_every_numeric_cell(case):
    _, _, context = case
    aliases = _fact_aliases(context)
    for sample in context.samples:
        table = _baseline_table(context, sample.sample_id, fact_aliases=aliases)
        source = context.baseline_sources[sample.sample_id]
        for name in context.baselines[sample.sample_id]["measurements"]:
            fid = f"context.{source}.measurements.{name}.this_image"
            assert "{" + aliases[fid] + "}" in table
            assert context.pack.context_facts[fid].display in table


def test_unknown_aliases_still_fail_grounding(case):
    _, _, context = case
    payload = _aliased_response(context)
    payload["explanations"][0]["points"][0].update(text="Value {f999999}.", facts=["f999999"])
    draft = _expand_fact_aliases(FactsResponse.model_validate(payload), _fact_aliases(context))
    assert any("unknown fact" in error for error in validate_facts_response(draft, context))


def test_short_aliases_do_not_allow_cross_sample_evidence(case):
    text, baselines, _ = case
    first = json.loads(text)
    second = copy.deepcopy(first)
    second["sample_id"] = "img_second"
    context = prepare_facts_context(parse_facts_document([first, second]), KnowledgePack.empty(), baselines=baselines)
    payload = _aliased_response(context)
    payload["explanations"][1]["points"][0]["text"] = payload["explanations"][0]["points"][0]["text"]
    draft = _expand_fact_aliases(FactsResponse.model_validate(payload), _fact_aliases(context))
    assert any("unknown fact" in error for error in validate_facts_response(draft, context))


def test_unused_metadata_cannot_supply_missing_evidence(case):
    _, _, context = case
    payload = _aliased_response(context)
    point = payload["explanations"][0]["points"][0]
    point["text"] = "The image resembles the reported category."
    point["facts"] = comparison_payload(context)["explanations"][0]["points"][0]["facts"]
    draft = _expand_fact_aliases(FactsResponse.model_validate(payload), _fact_aliases(context))
    assert draft.explanations[0].points[0].facts == []
    assert validate_facts_response(draft, context)


def test_aliases_are_expanded_before_validator_critic_and_rendering(case):
    text, baselines, context = case
    client = FakeClient([text_response(_aliased_response(context)), critic_pass()])
    outcome = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines, client=client, short_ids=True)
    assert outcome.audit["validator_passed"]
    assert outcome.audit["critic_passed"]
    assert "Short fact aliases" in client.messages.requests[0]["system"]
    assert "context." in client.messages.requests[1]["messages"][0]["content"]
    assert "{f" not in outcome.response.explanations[0].points[0].text


def test_identical_request_reuses_checked_result_without_network(case, tmp_path):
    text, baselines, context = case
    client = FakeClient([text_response(_aliased_response(context)), critic_pass()])
    first = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines, client=client,
                                   cache_dir=tmp_path, short_ids=True)
    no_network = FakeClient([])
    second = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines, client=no_network,
                                    cache_dir=tmp_path, short_ids=True)
    assert first.response == second.response
    assert first.audit["cache_hit"] is False
    assert second.audit["cache_hit"] is True
    assert second.audit["api_calls_this_run"] == 0
    assert second.audit["cached_critic_verdict"] is True
    assert no_network.messages.requests == []
    assert text == (ROOT / "examples" / "facts.json").read_text()


@pytest.mark.parametrize("change", ["document", "question", "effort", "model", "critic_mode", "refresh", "knowledge", "baselines"])
def test_changed_request_or_refresh_does_not_reuse_result(case, tmp_path, change):
    text, baselines, context = case
    explain_facts_document(text, KnowledgePack.empty(), baselines=baselines,
        client=FakeClient([text_response(comparison_payload(context)), critic_pass()]), cache_dir=tmp_path)
    options = {}
    knowledge = KnowledgePack.empty()
    if change == "document":
        document = json.loads(text)
        document["timings_s"]["total_s"] += 1
        text = json.dumps(document)
    elif change == "question":
        options["question"] = "Focus on acquisition."
    elif change == "effort":
        options["config"] = ExplainerConfig(effort="low")
    elif change == "model":
        options["config"] = ExplainerConfig(model="other-model")
    elif change == "critic_mode":
        options["critic_mode"] = "blocking"
    elif change == "knowledge":
        knowledge = KnowledgePack({}, raw="updated reference revision")
        knowledge._add_record("SEM imaging reference", {"guidance": "Imaging cues are not material measurements."},
                              "reference", "test reference")
    elif change == "baselines":
        baselines = copy.deepcopy(baselines)
        for record in baselines.records:
            record["acquisition"]["probes"]["bse_noise_sigma"] += 10
    else:
        options["refresh_cache"] = True
    context = prepare_facts_context(parse_facts_document(text), knowledge, baselines=baselines)
    client = FakeClient([text_response(comparison_payload(context)), critic_pass()])
    result = explain_facts_document(text, knowledge, baselines=baselines, client=client,
                                    cache_dir=tmp_path, **options)
    assert result.audit["cache_hit"] is False
    assert len(client.messages.requests) == 2


def test_dino_cache_also_keys_the_original_document(case, tmp_path):
    from .test_dino_only import payload

    text, _, _ = case
    record = json.loads(text)
    context = prepare_facts_context(parse_facts_document(record, decision_mode="dino"), KnowledgePack.empty())
    options = {"decision_mode": "dino", "short_ids": True, "cache_dir": tmp_path}
    first = explain_facts_document(record, client=FakeClient([text_response(payload(context)), critic_pass()]), **options)
    record["timings_s"]["total_s"] += 1
    second = explain_facts_document(record, client=FakeClient([text_response(payload(context)), critic_pass()]), **options)
    assert first.audit["cache_key"] != second.audit["cache_key"]
    assert second.audit["cache_hit"] is False


def test_cache_revalidates_claims_not_just_integrity(case, tmp_path):
    text, baselines, context = case
    first = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines,
        client=FakeClient([text_response(comparison_payload(context)), critic_pass()]), cache_dir=tmp_path)
    cache = ResponseCache(tmp_path)
    entry = cache.get(first.audit["cache_key"])
    entry["draft"]["explanations"][0]["reported_answer"] = "invented category"
    cache.put(first.audit["cache_key"], entry)
    client = FakeClient([text_response(comparison_payload(context)), critic_pass()])
    second = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines, client=client, cache_dir=tmp_path)
    assert not second.audit["cache_hit"]
    assert len(client.messages.requests) == 2


def test_cache_corruption_expiry_and_version_changes_are_misses(tmp_path):
    key = response_cache_key({"prompt": "a"}, code_version="one")
    assert key != response_cache_key({"prompt": "a"}, code_version="two")
    assert key != response_cache_key({"prompt": "b"}, code_version="one")
    cache = ResponseCache(tmp_path, max_age_seconds=60)
    cache.put(key, {"answer": "checked"}, now=100)
    assert cache.get(key, now=101) == {"answer": "checked"}
    assert cache.get(key, now=161) is None
    path = tmp_path / f"{key}.json"
    path.write_text("not json")
    assert cache.get(key, now=101) is None


def test_failed_critic_is_not_cached(case, tmp_path):
    text, baselines, context = case

    def fail(_):
        raise RuntimeError("critic unavailable")

    client = FakeClient([text_response(comparison_payload(context)), fail, fail])
    outcome = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines, client=client, cache_dir=tmp_path)
    assert outcome.audit["critic_error"]
    assert not list(tmp_path.glob("*.json"))


def test_failed_generation_is_not_cached(case, tmp_path):
    from sentinel.explainer import ExplainerError

    text, baselines, context = case
    bad = comparison_payload(context)
    bad["explanations"][0]["reported_answer"] = "wrong category"
    with pytest.raises(ExplainerError):
        explain_facts_document(text, KnowledgePack.empty(), baselines=baselines,
            client=FakeClient([text_response(bad)]), cache_dir=tmp_path, max_retries=0)
    assert not list(tmp_path.glob("*.json"))


def test_optional_cache_write_failure_does_not_discard_checked_answer(case, tmp_path, monkeypatch):
    text, baselines, context = case

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ResponseCache, "put", fail)
    result = explain_facts_document(text, KnowledgePack.empty(), baselines=baselines,
        client=FakeClient([text_response(comparison_payload(context)), critic_pass()]), cache_dir=tmp_path)
    assert result.audit["validator_passed"]
    assert result.audit["cache_write_error"] == "OSError"


def test_main_model_override_keeps_validators_and_output_contract(case, monkeypatch, capsys, tmp_path):
    import main
    from sentinel.pipeline_facts import FactsOutcome

    _, _, context = case
    response = FactsResponse.model_validate(comparison_payload(context))
    calls = []

    def explain(document, knowledge, **kwargs):
        calls.append(kwargs)
        return FactsOutcome(context.samples, response, {"cache_hit": False})

    monkeypatch.setattr(main, "explain_facts_document", explain)
    main.main(model="claude-haiku-4-5-20251001", cache_dir=tmp_path)
    assert calls[0]["config"].model == "claude-haiku-4-5-20251001"
    assert calls[0]["config"].effort is None
    assert calls[0]["critic_mode"] == "advisory"
    assert calls[0]["short_ids"] is True
    assert calls[0]["cache_dir"] == tmp_path
    assert isinstance(json.loads(capsys.readouterr().out), dict)
    main.main(use_cache=False)
    assert calls[1]["cache_dir"] is None
    assert calls[1]["config"].effort == "low"
