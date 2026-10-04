"""Contract, fact sheet, validator, renderer, template and the retry loop."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from sentinel import (
    BatteryResult,
    Citation,
    EvidenceItem,
    Explanation,
    ExplainerWorkflow,
    KnowledgePack,
    NullCritic,
    TemplateExplainer,
    Validator,
    ValidatorConfig,
    WorkflowConfig,
    build_fact_sheet,
    render,
)
from sentinel.critic import ClaudeCritic, CriticConfig
from sentinel.explainer import (
    ClaudeExplainer,
    ExplainerConfig,
    ExplainerError,
    extract_json_object,
)
from sentinel.renderer import UnresolvedPlaceholder
from sentinel.selftest import run_selftest
from sentinel.template import template_validator_config
from sentinel.tools import ReadOnlyTools

from .conftest import (
    FakeClient,
    critic_fail,
    critic_pass,
    fenced_response,
    text_response,
    tool_response,
)


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_example_document_validates(result):
    assert result.battery_id == "BAT-07"
    assert result.classification.predicted == "B"
    assert result.available_detectors() == ["BSE", "ETD", "InLens"]


def test_predicted_type_must_be_the_argmax(result):
    payload = json.loads(result.model_dump_json())
    payload["classification"]["predicted"] = "A"
    payload["classification"]["runner_up"] = "B"
    with pytest.raises(ValidationError, match="not the argmax"):
        BatteryResult.model_validate(payload)


def test_runner_up_must_be_second(result):
    payload = json.loads(result.model_dump_json())
    payload["classification"]["runner_up"] = "C"
    with pytest.raises(ValidationError, match="not the second-highest"):
        BatteryResult.model_validate(payload)


def test_unknown_field_is_rejected(result):
    payload = json.loads(result.model_dump_json())
    payload["sneaky_extra"] = 1
    with pytest.raises(ValidationError):
        BatteryResult.model_validate(payload)


def test_override_signoff_requires_a_reason():
    from sentinel import SignOff

    with pytest.raises(ValidationError, match="needs a reason"):
        SignOff(decision="override", user="zoe", timestamp="2026-10-03T19:00:00Z")


def test_with_explanation_does_not_mutate_the_classification(result):
    updated = result.with_explanation(Explanation(headline="x"))
    assert updated.classification == result.classification
    assert result.explanation is None
    assert updated.explanation is not None


# --------------------------------------------------------------------------- #
# Fact sheet
# --------------------------------------------------------------------------- #
def test_fact_sheet_has_the_ids_the_proposal_names(sheet):
    assert "stats.BSE.porosity.z_B" in sheet
    assert "stats.BSE.porosity.value" in sheet
    assert "cls.fused.B" in sheet
    assert "region.BSE.1.pore" in sheet


def test_fractions_display_as_percentages(sheet):
    assert sheet.get("stats.BSE.porosity.value").display == "27.1%"
    assert sheet.get("cls.fused.B").display == "72.7%"


def test_micron_statistic_keeps_its_unit(sheet):
    assert sheet.get("stats.BSE.d50.value").display == "6.42 um"


def test_zscores_are_signed(sheet):
    assert sheet.get("stats.BSE.porosity.z_A").display == "-3.86"
    assert sheet.get("stats.InLens.cbd_fraction.z_A").display == "+3.67"


def test_derived_comparisons_exist_so_the_model_need_not_subtract(sheet):
    delta = sheet.get("stats.BSE.porosity.delta_vs_A")
    assert delta is not None
    assert delta.display == "-8.1 pp"
    assert sheet.get("region.BSE.1.pore_delta").display == "-5.7 pp"


def test_region_ids_are_canonical(sheet):
    assert sheet.region_ids == ["BSE region 1", "BSE region 2", "InLens region 1"]


def test_separating_statistics_are_ordered(sheet):
    assert sheet.separating_statistics[0].startswith("stats.BSE.porosity")
    assert sheet.separating_statistics[1].startswith("stats.BSE.d50")


def test_prompt_block_mentions_every_group(sheet):
    block = sheet.to_prompt_block()
    for heading in ("Probabilities", "Measured statistics", "Type profiles",
                    "z-scores", "Precomputed comparisons", "Evidence regions"):
        assert heading in block


def test_nearest_ids_suggests_a_real_id(sheet):
    assert "stats.BSE.porosity.value" in sheet.nearest_ids("stats.BSE.porosity.val")


# --------------------------------------------------------------------------- #
# Template path
# --------------------------------------------------------------------------- #
def test_template_passes_its_own_validator(sheet, pack):
    draft = TemplateExplainer().build(sheet)
    report = Validator(sheet, pack, template_validator_config()).validate(draft)
    assert report.passed, report.feedback()


def test_template_renders_without_leftover_placeholders(sheet):
    draft = TemplateExplainer().build(sheet)
    rendered = render(draft, sheet)
    text = " ".join(t for _, t in rendered.text_fields())
    assert "{" not in text and "}" not in text
    assert "27.1%" in text


def test_template_names_the_runner_up_and_the_prediction(sheet):
    rendered = render(TemplateExplainer().build(sheet), sheet)
    assert "type B" in rendered.headline
    assert "type A" in rendered.why_not_runner_up


def test_template_states_the_template_caveat(sheet):
    draft = TemplateExplainer().build(sheet)
    assert any("fixed template" in c for c in draft.caveats)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def test_renderer_substitutes_display_strings(sheet):
    draft = Explanation(
        headline="Porosity {stats.BSE.porosity.value} for type {pred_type}.",
        why_not_runner_up="x",
    )
    out = render(draft, sheet)
    assert out.headline == "Porosity 27.1% for type B."


def test_renderer_is_strict_about_unknown_facts(sheet):
    draft = Explanation(headline="{stats.BSE.nonsense.value}")
    with pytest.raises(UnresolvedPlaceholder):
        render(draft, sheet)


def test_renderer_leaves_citations_untouched(sheet, pack):
    quote = pack.sources["S2"].body.splitlines()[-1]
    draft = Explanation(
        headline="x",
        evidence=[EvidenceItem(
            kind="statistic", claim="y", facts=["cls.fused.B"],
            citations=[Citation(source="S2", quote=quote)],
        )],
    )
    out = render(draft, sheet)
    assert out.evidence[0].citations[0].quote == quote


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def test_good_draft_passes(good_draft, sheet, pack):
    report = Validator(sheet, pack).validate(good_draft)
    assert report.passed, report.feedback()


def test_findings_carry_code_location_and_hint(sheet, pack):
    draft = Explanation(
        headline="Porosity is 34%.", why_not_runner_up="x",
        evidence=[EvidenceItem(kind="statistic", claim="z", facts=["cls.fused.B"])],
    )
    report = Validator(sheet, pack).validate(draft)
    finding = next(f for f in report.errors if f.code == "NUM001")
    assert finding.loc == "headline"
    assert finding.fix_hint and "placeholder" in finding.fix_hint
    assert "34%" in (finding.observed or "")


def test_bad_fact_id_suggests_the_real_one(sheet, pack):
    draft = Explanation(
        headline="x", why_not_runner_up="y",
        evidence=[EvidenceItem(kind="statistic", claim="z",
                               facts=["stats.BSE.porositty.value"])],
    )
    report = Validator(sheet, pack).validate(draft)
    finding = next(f for f in report.errors if f.code == "PH002")
    assert "stats.BSE.porosity.value" in (finding.fix_hint or "")


def test_low_confidence_requires_a_hedge(result, pack):
    payload = json.loads(result.model_dump_json())
    payload["classification"]["confidence"] = "low"
    low = BatteryResult.model_validate(payload)
    low_sheet = build_fact_sheet(low)
    draft = Explanation(
        headline="Classified as type {pred_type}.",
        why_not_runner_up="The separation is visible.",
        evidence=[EvidenceItem(kind="statistic", claim="a", facts=["cls.fused.B"])],
        caveats=["A detector flag was raised."],
    )
    codes = {f.code for f in Validator(low_sheet, pack).validate(draft).errors}
    assert "CONF003" in codes


def test_unlike_any_type_must_be_disclosed(result, pack):
    payload = json.loads(result.model_dump_json())
    payload["classification"]["unlike_any_type"] = True
    odd = BatteryResult.model_validate(payload)
    odd_sheet = build_fact_sheet(odd)
    draft = Explanation(
        headline="x", why_not_runner_up="y",
        evidence=[EvidenceItem(kind="statistic", claim="a", facts=["cls.fused.B"])],
        caveats=["A quality flag was raised."],
    )
    codes = {f.code for f in Validator(odd_sheet, pack).validate(draft).errors}
    assert "CONF004" in codes


def test_branch_disagreement_must_be_stated(result, pack):
    payload = json.loads(result.model_dump_json())
    payload["classification"]["stats_classifier"] = {
        "predicted": "A", "probs": {"A": 0.6, "B": 0.3, "C": 0.1}
    }
    split = BatteryResult.model_validate(payload)
    split_sheet = build_fact_sheet(split)
    draft = Explanation(
        headline="x", why_not_runner_up="y", agreement="Everything lines up.",
        evidence=[EvidenceItem(kind="statistic", claim="a", facts=["cls.fused.B"])],
        caveats=["A flag was raised."],
    )
    codes = {f.code for f in Validator(split_sheet, pack).validate(draft).errors}
    assert "CONF005" in codes


def test_progress_key_and_signature_are_stable(sheet, pack):
    draft = Explanation(headline="Porosity is 34%.")
    a = Validator(sheet, pack).validate(draft)
    b = Validator(sheet, pack).validate(draft)
    assert a.progress_key() == b.progress_key()
    assert a.signature() == b.signature()


# --------------------------------------------------------------------------- #
# Anti-degradation: the vacuous explanation must not pass
# --------------------------------------------------------------------------- #
def test_empty_explanation_fails_the_content_floor(sheet, pack):
    report = Validator(sheet, pack).validate(Explanation(headline="", why_not_runner_up=""))
    codes = {f.code for f in report.errors}
    assert {"CONTENT001", "CONTENT002", "CONTENT005"} <= codes


def test_explanation_with_no_facts_fails(sheet, pack):
    draft = Explanation(
        headline="Classified as type {pred_type}.",
        why_not_runner_up="It differs from type {runner_up}.",
        evidence=[
            EvidenceItem(kind="visual", claim="It looks right.", regions=["BSE region 1"]),
            EvidenceItem(kind="visual", claim="It also looks right.",
                         regions=["BSE region 2"]),
        ],
        caveats=["A flag was raised."],
    )
    codes = {f.code for f in Validator(sheet, pack).validate(draft).errors}
    assert "CONTENT003" in codes


def test_seeded_defects_are_all_caught(result, pack):
    rows, caught = run_selftest(result, pack)
    missed = [name for name, _, _, ok in rows if not ok]
    assert not missed, f"missed: {missed}"
    assert caught == len(rows)


# --------------------------------------------------------------------------- #
# Explainer response parsing
# --------------------------------------------------------------------------- #
def test_main_formats_review_json_as_bullet_points(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    import sentinel

    def unexpected_call(*args, **kwargs):
        pytest.fail("Importing main must not call the explainer")

    monkeypatch.setattr(sentinel, "explain_battery", unexpected_call)
    spec = importlib.util.spec_from_file_location("main_import_test", Path(__file__).resolve().parents[1] / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    format_review = module.format_review

    explanation = Explanation(
        headline="Classified as type B at 72.7%.",
        why_not_runner_up="Type A is less consistent with the measurements.",
        evidence=[EvidenceItem(
            kind="statistic", claim="Porosity is 27.1%.",
            facts=["stats.BSE.porosity.value"],
            citations=[Citation(source="S1", quote="Supporting source text.")],
        )],
        agreement="The detectors agree.",
        caveats=["Confidence is medium.", "Image quality limits interpretation."],
        validator_passed=True,
    )
    assert format_review(explanation.model_dump_json()) == (
        "- Classified as type B at 72.7%. The detectors agree.\n\n"
        "- Porosity is 27.1%.\n\n"
        "- Type A is less consistent with the measurements.\n\n"
        "- Confidence is medium.\n\n- Image quality limits interpretation."
    )
    assert capsys.readouterr().out == ""


def test_main_formats_empty_sections_and_unicode():
    from main import format_review

    explanation = Explanation(
        headline="  Median diameter is 6.42 µm.  ",
        evidence=[EvidenceItem(kind="visual", claim=" ")],
        caveats=["", "  Review is tentative.  "],
    )
    assert format_review(explanation.model_dump_json()) == (
        "- Median diameter is 6.42 µm.\n\n- Review is tentative."
    )


def test_main_prints_only_review(result, pack, monkeypatch, capsys, tmp_path):
    import main

    workflow, _ = _workflow(pack, [text_response(_payload()), critic_pass()])
    outcome = workflow.run(result)
    def explain(*args, config, progress):
        assert config.allow_template_fallback is False
        assert config.use_critic is True
        assert config.max_retries == 2  # three attempts by default
        assert config.explainer.max_tokens == 8000
        assert config.explainer.max_transport_retries == 0
        assert config.validator.max_words == 280
        assert config.validator.max_evidence_items == 3
        assert config.validator.max_caveats == 2
        assert progress is None
        return outcome

    monkeypatch.setattr(main, "explain_battery", explain)
    monkeypatch.chdir(tmp_path)
    main.main(quiet=True, legacy_result=result)
    captured = capsys.readouterr()
    assert captured.out == main.format_review(outcome.explanation.model_dump_json()) + "\n"
    assert captured.err == ""
    main.main(debug=True, quiet=True, legacy_result=result)
    captured = capsys.readouterr()
    assert captured.out == main.format_review(outcome.explanation.model_dump_json()) + "\n"
    assert "llm" in captured.err
    assert "knowledge_pack" in captured.err


def test_api_clients_use_explicit_timeouts_without_hidden_retries(monkeypatch):
    import anthropic
    import sentinel.explainer
    import sentinel.critic

    captured = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: captured.append(kwargs) or FakeClient([]))
    monkeypatch.setattr(sentinel.explainer, "ensure_api_key", lambda: True)
    monkeypatch.setattr(sentinel.critic, "ensure_api_key", lambda: True)
    ClaudeExplainer(config=ExplainerConfig(timeout_seconds=42)).client
    ClaudeCritic(config=CriticConfig(timeout_seconds=24)).client
    assert captured == [{"timeout": 42, "max_retries": 0}, {"timeout": 24, "max_retries": 0}]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_request_timeouts_are_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        ExplainerConfig(timeout_seconds=timeout)
    with pytest.raises(ValueError, match="timeout"):
        CriticConfig(timeout_seconds=timeout)


def test_progress_reports_stages_and_repairs_without_repeating_facts(result, pack):
    events = []
    workflow, client = _workflow(pack, [text_response(_payload(headline="Porosity is 99%.")),
                                       text_response(_payload()), critic_pass()])
    workflow.progress = events.append
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    assert any("attempt 1" in event for event in events)
    assert any("attempt 2" in event for event in events)
    assert any("Validating" in event for event in events)
    assert any("critic" in event for event in events)
    initial = client.messages.requests[0]["messages"][0]
    repair = client.messages.requests[1]["messages"][-1]
    assert "# Fact sheet" in json.dumps(initial)
    assert "# Fact sheet" not in json.dumps(repair)
    assert "NUM001" in json.dumps(repair)
    assert initial == client.messages.requests[1]["messages"][0]


def test_progress_heartbeat_stops_when_activity_finishes():
    from threading import Event
    from sentinel.explainer import report_activity

    waiting = Event()
    events = []

    def progress(message):
        events.append(message)
        if "still waiting" in message:
            waiting.set()

    with report_activity(progress, "Generating", interval=0.001):
        assert waiting.wait(2)
    assert events[0] == "Generating"
    assert len(events) >= 2
    count = len(events)
    Event().wait(0.02)
    assert len(events) == count


def test_request_timeout_reaches_explainer_and_critic(result, pack):
    workflow, client = _workflow(pack, [text_response(_payload()), critic_pass()])
    workflow.run(result)
    assert all(request["timeout"] == 90.0 for request in client.messages.requests)


def test_main_progress_is_on_stderr_and_context_can_be_expanded(result, pack, monkeypatch, capsys):
    import main

    workflow, _ = _workflow(pack, [text_response(_payload()), critic_pass()])
    outcome = workflow.run(result)
    selected = []

    def explain(result, knowledge, images, **kwargs):
        selected.append(knowledge)
        if kwargs["progress"]:
            kwargs["progress"]("Generating AI review")
        return outcome

    monkeypatch.setattr(main, "explain_battery", explain)
    main.main(quiet=False, legacy_result=result)          # progress is opt-in now
    captured = capsys.readouterr()
    assert "Loading knowledge" in captured.err and "Generating AI review" in captured.err
    assert "Review ready" in captured.err
    assert "[Sentinel" not in captured.out
    assert not any(source.metadata.get("batches") for source in selected[0].sources.values())
    main.main(quiet=True, full_context=True, legacy_result=result)
    assert capsys.readouterr().err == ""
    assert any(source.metadata.get("batches") for source in selected[1].sources.values())


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"max_attempts": 0}, {"max_tokens": 0}])
def test_main_rejects_invalid_call_limits(kwargs):
    import main

    with pytest.raises(ValueError):
        main.main(**kwargs)


def test_ai_only_workflow_does_not_replace_errors_with_a_template(result, pack):
    def fail(_):
        raise RuntimeError("provider unavailable")

    workflow, _ = _workflow(pack, [fail], allow_template_fallback=False)
    with pytest.raises(ExplainerError, match="provider unavailable") as error:
        workflow.run(result)
    assert "template fallback is disabled" in str(error.value)
    assert all(attempt.source == "llm" for attempt in error.value.attempts)


def test_ai_only_workflow_reports_validation_errors(result, pack):
    workflow, _ = _workflow(pack, [text_response(_payload(headline="Porosity is 99%."))],
                            allow_template_fallback=False, max_retries=0)
    with pytest.raises(ExplainerError, match="NUM001"):
        workflow.run(result)


def test_ai_only_workflow_keeps_the_critic(result, pack):
    workflow, _ = _workflow(pack, [text_response(_payload()), critic_fail()],
                            allow_template_fallback=False, max_retries=0)
    with pytest.raises(ExplainerError, match="CRIT_CONF"):
        workflow.run(result)


def test_main_missing_credentials_never_prints_a_template(result, monkeypatch, capsys):
    import main
    import sentinel.explainer

    monkeypatch.setattr(main, "load_dotenv", lambda **kwargs: [])
    monkeypatch.setattr(sentinel.explainer, "ensure_api_key", lambda: False)
    with pytest.raises(ExplainerError, match="ANTHROPIC_API_KEY is not set"):
        main.main(legacy_result=result)
    assert capsys.readouterr().out == ""


def test_main_rejects_an_unexpected_template_result(result, pack, monkeypatch, capsys):
    import main

    workflow, _ = _workflow(pack, [text_response(_payload(headline="Porosity is 99%."))], max_retries=0)
    outcome = workflow.run(result)
    assert outcome.generator == "template"
    monkeypatch.setattr(main, "explain_battery", lambda *a, **kw: outcome)
    with pytest.raises(ExplainerError, match="AI-generated"):
        main.main(legacy_result=result)
    assert capsys.readouterr().out == ""


def test_main_loads_configuration_from_project_directory(result, pack, tmp_path, monkeypatch):
    import main

    calls = []
    workflow, _ = _workflow(pack, [text_response(_payload()), critic_pass()])
    outcome = workflow.run(result)
    monkeypatch.setattr(main, "load_dotenv", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(main, "explain_battery", lambda *a, **kw: outcome)
    monkeypatch.chdir(tmp_path)
    main.main(legacy_result=result)
    assert calls == [{"start": main.ROOT}]


def test_main_formats_template_fallback(sheet):
    from main import format_review

    explanation = render(TemplateExplainer().build(sheet), sheet)
    text = format_review(explanation.model_dump_json())
    assert "fixed template" in text
    assert "27.1%" in text
    assert "{" not in text


def test_remaining_type_has_a_grounded_label(result, sheet):
    assert sheet.get("cls.other_type").display == "C"
    payload = result.model_dump()
    payload["classification"].update(predicted="C", runner_up="B", fused={"A": 0.1, "B": 0.2, "C": 0.7})
    other_sheet = build_fact_sheet(BatteryResult.model_validate(payload))
    assert other_sheet.get("cls.other_type").display == "A"
    draft = Explanation(headline="Type {pred_type} is favoured; type {cls.other_type} is not excluded.")
    assert "type A" in render(draft, other_sheet).headline


def test_concise_review_budget_checks_all_displayed_text(good_draft, sheet, pack):
    draft = good_draft.model_copy(deep=True)
    draft.agreement = "The detectors agree. " * 100
    report = Validator(sheet, pack, ValidatorConfig(max_words=280)).validate(draft)
    assert "STYLE001" in {finding.code for finding in report.errors}
    assert draft.agreement.endswith("The detectors agree. ")
    assert "STYLE001" not in {finding.code for finding in Validator(sheet, pack).validate(draft).errors}


def test_concise_review_limits_inventory_without_dropping_grounding(good_draft, sheet, pack):
    draft = good_draft.model_copy(deep=True)
    draft.evidence *= 3
    draft.caveats = ["A quality flag limits confidence."] * 3
    report = Validator(sheet, pack, ValidatorConfig(max_evidence_items=3, max_caveats=2)).validate(draft)
    assert {"STYLE002", "STYLE003"} <= {finding.code for finding in report.errors}
    empty = Explanation(headline="A short assessment.")
    report = Validator(sheet, pack, ValidatorConfig(max_words=280)).validate(empty)
    assert "CONTENT002" in {finding.code for finding in report.errors}


def test_template_config_preserves_requested_length_limits():
    config = template_validator_config(ValidatorConfig(max_words=280, max_evidence_items=3, max_caveats=2))
    assert (config.max_words, config.max_evidence_items, config.max_caveats) == (280, 3, 2)
    assert config.min_citations == 0


def test_critic_distinguishes_causal_claims_from_cautions():
    from sentinel.prompts import CRITIC_ROLE

    assert "NOT an asserted mechanism" in CRITIC_ROLE
    assert "Continue to reject actual unsupported" in CRITIC_ROLE
    assert "style editing, not a grounding error" in CRITIC_ROLE


def test_review_prompt_requests_depth_without_invented_visuals(pack):
    from sentinel.prompts import build_system_prompt

    instructions = build_system_prompt(pack)[0]["text"]
    assert "professional technical review" in instructions
    assert "bullet point" in instructions
    assert "180-230 words" in instructions
    assert "observed or measured characteristic" in instructions
    assert "remaining category" in instructions
    assert "not zero probability" in instructions
    assert "450-750 words" not in instructions
    assert "Do not claim to have inspected an image" in instructions
    assert "EVERY NUMBER COMES FROM A PLACEHOLDER" in instructions
    assert "THREE items, four at the very most" not in instructions


@pytest.mark.parametrize("field", ["headline", "why_not_runner_up", "agreement"])
def test_sentence_lists_in_prose_fields_are_normalised(field):
    from sentinel.explainer import parse_explanation

    draft = parse_explanation(_payload(**{field: ["First statement.", "Second statement."]}))
    assert getattr(draft, field) == "First statement. Second statement."


@pytest.mark.parametrize("value", [[], ["A statement.", 99], [{"text": "A statement."}]])
def test_non_text_lists_are_not_coerced_into_prose(value):
    from sentinel.explainer import parse_explanation

    with pytest.raises(ExplainerError, match="contract"):
        parse_explanation(_payload(why_not_runner_up=value))


def test_normalised_prose_still_requires_grounded_numbers(sheet, pack):
    from sentinel.explainer import parse_explanation

    draft = parse_explanation(_payload(why_not_runner_up=["Porosity is 99%.", "Review is needed."]))
    assert "NUM001" in {finding.code for finding in Validator(sheet, pack).validate(draft).errors}


def test_json_extracted_from_a_fence():
    assert extract_json_object('text\n```json\n{"a": 1}\n```\n') == {"a": 1}


def test_json_extracted_from_surrounding_prose():
    assert extract_json_object('Sure. {"a": 2} Hope that helps.') == {"a": 2}


def test_no_json_raises():
    with pytest.raises(ExplainerError, match="no JSON object"):
        extract_json_object("I would rather not.")


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def _payload(**overrides):
    base = {
        "headline": "Classified as type {pred_type} at {cls.fused.B}.",
        "why_not_runner_up": "Porosity sits {stats.BSE.porosity.delta_vs_A} from type "
                             "{runner_up}.",
        "evidence": [
            {
                "kind": "statistic",
                "detector": "BSE",
                "claim": "Porosity is {stats.BSE.porosity.value}, "
                         "{stats.BSE.porosity.z_B} from the type {pred_type} profile.",
                "facts": ["stats.BSE.porosity.value", "stats.BSE.porosity.z_B"],
                "regions": [],
                "citations": [],
            },
            {
                "kind": "both",
                "detector": "BSE",
                "claim": "BSE region 1 holds {region.BSE.1.share_of_evidence} of the "
                         "evidence, with a pore fraction of {region.BSE.1.pore}.",
                "facts": ["region.BSE.1.share_of_evidence", "region.BSE.1.pore"],
                "regions": ["BSE region 1"],
                "citations": [{
                    "source": "S1",
                    "quote": "Because the signal depends on composition rather than "
                             "on surface shape, a backscattered electron image is the "
                             "preferred basis for measuring the active material "
                             "fraction, particle size and particle shape.",
                }],
            },
        ],
        "agreement": "Detectors agreeing: {cls.detector_agreement}.",
        "caveats": ["A charging-streak flag was raised on one of the images."],
    }
    base.update(overrides)
    return base


def _workflow(pack, script, **cfg):
    client = FakeClient(script)
    explainer = ClaudeExplainer(client, ExplainerConfig(model="fake-sonnet"))
    critic = ClaudeCritic(client, CriticConfig(model="fake-haiku"))
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(**cfg), explainer=explainer, critic=critic
    )
    return workflow, client


def test_happy_path_first_attempt(result, pack):
    workflow, client = _workflow(pack, [text_response(_payload()), critic_pass()])
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    assert outcome.validator_passed and outcome.critic_passed
    assert len(outcome.attempts) == 1
    assert "72.7%" in outcome.explanation.headline
    assert "{" not in outcome.explanation.headline


def test_tool_use_path_is_accepted(result, pack):
    workflow, _ = _workflow(
        pack,
        [tool_response("submit_explanation", _payload()), critic_pass()],
    )
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    assert outcome.attempts[0].call.used_tool


def test_fenced_json_is_accepted(result, pack):
    workflow, _ = _workflow(pack, [fenced_response(_payload()), critic_pass()])
    assert workflow.run(result).generator == "llm"


def test_tool_choice_is_not_forced_for_sonnet(result, pack):
    workflow, client = _workflow(pack, [text_response(_payload()), critic_pass()])
    workflow.run(result)
    request = client.messages.requests[0]
    assert "tool_choice" not in request
    assert "temperature" not in request


def test_images_precede_text_and_the_pack_is_cached(result, pack, tmp_path):
    from sentinel import ImageAsset

    png = tmp_path / "BSE_overview.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    workflow, client = _workflow(pack, [text_response(_payload()), critic_pass()])
    workflow.run(result, [ImageAsset(detector="BSE", kind="overview", path=png)])

    request = client.messages.requests[0]
    kinds = [block["type"] for block in request["messages"][0]["content"]]
    assert kinds.index("image") < kinds.index("text", kinds.index("image"))
    assert any("cache_control" in block for block in request["system"])


def test_retry_after_a_validator_failure(result, pack):
    bad = _payload(headline="Porosity is 27.1% so this is type B.")
    workflow, client = _workflow(
        pack, [text_response(bad), text_response(_payload()), critic_pass()]
    )
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    assert len(outcome.attempts) == 2
    assert not outcome.attempts[0].ok and outcome.attempts[1].ok

    feedback = client.messages.requests[1]["messages"][-1]["content"][-1]["text"]
    assert "NUM001" in feedback and "TYPE002" in feedback

    # The retry conversation must still begin with a user turn (an
    # assistant-first message list is an API error), and the rejected draft is
    # replayed in between so the retry is a diff rather than a fresh draft.
    roles = [m["role"] for m in client.messages.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_retry_feedback_warns_against_deleting_evidence(result, pack):
    workflow, client = _workflow(
        pack,
        [text_response(_payload(headline="34% porosity")), text_response(_payload()),
         critic_pass()],
    )
    workflow.run(result)
    feedback = client.messages.requests[1]["messages"][-1]["content"][-1]["text"]
    assert "do not delete evidence" in feedback


def test_critic_failure_triggers_a_retry(result, pack):
    workflow, _ = _workflow(
        pack,
        [text_response(_payload()), critic_fail(),
         text_response(_payload()), critic_pass()],
    )
    outcome = workflow.run(result)
    assert len(outcome.attempts) == 2
    assert outcome.critic_passed is True


def test_non_blocking_critic_ships_but_records_the_objection(result, pack):
    """blocking=False must not silently launder a critic failure into a pass."""
    client = FakeClient([text_response(_payload()), critic_fail()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake", blocking=False)),
    )
    outcome = workflow.run(result)
    assert outcome.generator == "llm"            # it shipped
    assert outcome.critic_passed is False        # and the objection is on record
    assert len(outcome.attempts) == 1            # no retry was triggered
    findings = outcome.attempts[0].critic.findings
    assert [f.severity for f in findings] == ["warning"]
    assert any(f.code == "CRIT_CONF" for f in findings)


def test_falls_back_to_the_template_after_the_retry_budget(result, pack):
    # Three *different* failures, so the oscillation guard does not fire first
    # and the full retry budget is spent.
    workflow, _ = _workflow(pack, [
        text_response(_payload(headline="Porosity is 34% here.")),
        text_response(_payload(headline="This is definitely type A.")),
        text_response(_payload(headline="It is twice the binder of the other.")),
    ])
    outcome = workflow.run(result)
    assert outcome.generator == "template"
    assert outcome.validator_passed
    assert len(outcome.attempts) == 4            # 3 model attempts + the template
    assert "27.1%" in " ".join(t for _, t in outcome.explanation.text_fields())


def test_oscillation_stops_the_loop_early(result, pack):
    """Identical failures twice means the model is stuck; stop paying for it."""
    bad = _payload(headline="Porosity is 34%.")
    workflow, client = _workflow(pack, [text_response(bad)] * 3, max_retries=5)
    outcome = workflow.run(result)
    assert outcome.generator == "template"
    assert len(client.messages.requests) == 2    # stopped well before 6 attempts


def test_unparseable_response_is_retried_then_templated(result, pack):
    def refuse(_):
        from .conftest import FakeResponse, FakeTextBlock

        return FakeResponse(content=[FakeTextBlock("I cannot help with that.")])

    workflow, _ = _workflow(pack, [refuse, refuse, refuse])
    outcome = workflow.run(result)
    assert outcome.generator == "template"
    assert all(a.error for a in outcome.attempts if a.source == "llm")


def test_api_exception_falls_back_to_the_template(result, pack):
    def boom(_):
        raise RuntimeError("503 overloaded")

    workflow, _ = _workflow(pack, [boom])
    outcome = workflow.run(result)
    assert outcome.generator == "template"
    assert "503" in (outcome.attempts[0].error or "")


def test_critic_exception_does_not_break_the_run(result, pack):
    def boom(kwargs):
        if kwargs.get("model") == "fake-haiku":
            raise RuntimeError("critic down")
        raise AssertionError("unexpected call")

    client = FakeClient([text_response(_payload()), boom])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake-sonnet")),
        critic=ClaudeCritic(client, CriticConfig(model="fake-haiku")),
    )
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    assert outcome.critic_passed is None
    assert "critic" in (outcome.audit["critic_skipped_reason"] or "")


def test_missing_api_key_falls_back_to_the_template(result, pack, monkeypatch, tmp_path):
    """No key must degrade to the template, never reach the network.

    This test previously relied on the environment having no key, which meant
    that once a key WAS configured it made a real, billed API call on every
    run. The key is now removed explicitly, and the working directory moved so
    no .env is discovered.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    workflow = ExplainerWorkflow(pack, WorkflowConfig(use_critic=False))
    workflow.explainer = None
    outcome = workflow.run(result)

    assert outcome.generator == "template"
    assert outcome.validator_passed
    assert "ANTHROPIC_API_KEY" in (outcome.attempts[0].error or "")


def test_null_critic_records_no_verdict(result, pack):
    workflow, _ = _workflow(pack, [text_response(_payload())], use_critic=False)
    workflow.critic = NullCritic()
    outcome = workflow.run(result)
    assert outcome.critic_passed is None


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
def test_audit_records_versions_and_fact_usage(result, pack):
    workflow, _ = _workflow(pack, [text_response(_payload()), critic_pass()])
    outcome = workflow.run(result)
    audit = outcome.audit
    assert audit["knowledge_pack"].startswith("sha256:")
    assert audit["prompt"].startswith("sha256:")
    # The audit names what actually ran, not what the config defaulted to.
    assert audit["explainer_model"] == "fake-sonnet"
    assert audit["critic_model"] == "fake-haiku"
    assert audit["critic_blocking"] is True
    assert audit["critic_declared_passed"] is True
    assert "stats.BSE.porosity.value" in audit["facts_used"]
    assert audit["citations_used"] == ["S1"]
    assert 0 < audit["fact_coverage"] <= 1


def test_outcome_serialises(result, pack):
    workflow, _ = _workflow(pack, [text_response(_payload()), critic_pass()])
    payload = json.loads(workflow.run(result).to_json())
    assert payload["explanation"]["generator"] == "llm"
    assert payload["draft"]["headline"].count("{") >= 1   # draft keeps placeholders


# --------------------------------------------------------------------------- #
# v2 read-only tools
# --------------------------------------------------------------------------- #
def test_zoom_region_returns_facts(result, pack, sheet):
    tools = ReadOnlyTools(sheet=sheet, result=result, pack=pack)
    out = tools.dispatch("zoom_region", {"detector": "BSE", "id": 1})
    assert out["region"] == "BSE region 1"
    assert "region.BSE.1.pore" in out["facts"]


def test_zoom_region_rejects_a_bad_region(result, pack, sheet):
    tools = ReadOnlyTools(sheet=sheet, result=result, pack=pack)
    out = tools.dispatch("zoom_region", {"detector": "BSE", "id": 99})
    assert "error" in out and out["available"]


def test_stat_distribution_lists_profiles(result, pack, sheet):
    tools = ReadOnlyTools(sheet=sheet, result=result, pack=pack)
    out = tools.dispatch("get_stat_distribution", {"stat": "stats.BSE.porosity"})
    assert out["facts"]["stats.BSE.porosity.mean_A"] == "35.2%"
    assert out["separating_rank"] == 1


def test_tool_budget_is_enforced(result, pack, sheet):
    tools = ReadOnlyTools(sheet=sheet, result=result, pack=pack, max_calls=2)
    for _ in range(2):
        tools.dispatch("search_knowledge", {"query": "calendering"})
    assert "error" in tools.dispatch("search_knowledge", {"query": "porosity"})
    assert tools.budget_left == 0


def test_tools_have_no_write_path(result, pack, sheet):
    tools = ReadOnlyTools(sheet=sheet, result=result, pack=pack)
    assert set(tools.handlers()) == {
        "zoom_region", "get_stat_distribution", "search_knowledge"
    }
    tools.dispatch("get_stat_distribution", {"stat": "stats.BSE.porosity"})
    assert result.classification.predicted == "B"
    assert result.explanation is None


# --------------------------------------------------------------------------- #
# Regression: text the contract supplied is code-authored, not model output
# --------------------------------------------------------------------------- #
def test_confidence_reasons_may_contain_numerals(result, pack):
    """Step 4 writes these strings; repeating them verbatim must not fail.

    Without this the validator rejects its own template whenever an upstream
    reason string contains a numeral.
    """
    payload = json.loads(result.model_dump_json())
    payload["classification"]["confidence_reasons"] = [
        "no ETD image was supplied, so the fusion rests on two detectors"
    ]
    doc = BatteryResult.model_validate(payload)
    sheet = build_fact_sheet(doc)

    draft = TemplateExplainer().build(sheet)
    report = Validator(sheet, pack, template_validator_config()).validate(draft)
    assert report.passed, report.feedback()
    assert any("two detectors" in c for c in draft.caveats)


def test_an_unrelated_numeral_in_the_same_field_is_still_caught(result, pack):
    payload = json.loads(result.model_dump_json())
    payload["classification"]["confidence_reasons"] = ["the fusion rests on two detectors"]
    doc = BatteryResult.model_validate(payload)
    sheet = build_fact_sheet(doc)
    draft = Explanation(
        headline="Classified as type {pred_type}.",
        why_not_runner_up="It differs from type {runner_up}.",
        evidence=[EvidenceItem(kind="statistic", claim="a", facts=["cls.fused.B"])],
        caveats=["the fusion rests on two detectors, and porosity is 34%"],
    )
    codes = {f.code for f in Validator(sheet, pack).validate(draft).errors}
    assert "NUM001" in codes


# --------------------------------------------------------------------------- #
# .env loading: convenience without putting a key in source
# --------------------------------------------------------------------------- #
def test_dotenv_is_found_and_parsed(tmp_path, monkeypatch):
    from sentinel.env import load_dotenv, parse_dotenv

    (tmp_path / ".env").write_text(
        '# a comment\n'
        'export ANTHROPIC_API_KEY="sk-ant-from-file"\n'
        'OTHER=plain value\n'
        'BAD LINE\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    applied = load_dotenv(start=tmp_path / "nested" / "deeper", override=True)
    assert "ANTHROPIC_API_KEY" in applied
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
    assert os.environ["OTHER"] == "plain value"
    assert "BAD LINE" not in parse_dotenv("BAD LINE")


def test_a_real_env_var_beats_the_file(tmp_path, monkeypatch):
    from sentinel.env import load_dotenv

    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
    load_dotenv(start=tmp_path, override=False)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-shell"


def test_missing_key_gives_advice_and_never_prints_a_key(tmp_path, monkeypatch):
    from sentinel.env import api_key_hint

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    hint = api_key_hint()
    assert ".env" in hint and "template" in hint
    assert "sk-" not in hint


# --------------------------------------------------------------------------- #
# Regression: a response cut off at max_tokens is truncation, not bad format
# --------------------------------------------------------------------------- #
def test_max_tokens_stop_reason_is_reported_as_truncation(result, pack):
    from sentinel.explainer import TruncatedResponse

    from .conftest import FakeResponse, FakeTextBlock

    def cut_off(_):
        return FakeResponse(
            content=[FakeTextBlock('{"headline":"Classified as type {pred_t')],
            stop_reason="max_tokens",
        )

    client = FakeClient([cut_off])
    explainer = ClaudeExplainer(client, ExplainerConfig(model="fake"))
    sheet = build_fact_sheet(result)
    with pytest.raises(TruncatedResponse, match="output limit"):
        explainer.generate(sheet, pack)


def test_leaked_tool_syntax_is_reported_as_truncation(result, pack):
    from sentinel.explainer import TruncatedResponse

    from .conftest import FakeResponse, FakeTextBlock

    def leaked(_):
        return FakeResponse(content=[FakeTextBlock(
            '{"headline":"x","evidence":"\n<parameter name="item">{"kind":"statistic"'
        )])

    explainer = ClaudeExplainer(FakeClient([leaked]), ExplainerConfig(model="fake"))
    with pytest.raises(TruncatedResponse, match="tool-call syntax"):
        explainer.generate(build_fact_sheet(result), pack)


def test_truncation_feedback_asks_for_brevity_not_valid_json(result, pack):
    from .conftest import FakeResponse, FakeTextBlock

    def cut_off(_):
        return FakeResponse(content=[FakeTextBlock('{"headline":"cut')],
                            stop_reason="max_tokens")

    client = FakeClient([cut_off, text_response(_payload()), critic_pass()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    feedback = client.messages.requests[1]["messages"][-1]["content"][-1]["text"]
    assert "SHORTER" in feedback
    assert "could not be parsed" not in feedback


def test_default_token_budget_fits_a_full_explanation():
    """3000 was too small and silently cost every battery its model draft."""
    assert ExplainerConfig().max_tokens >= 8000


# --------------------------------------------------------------------------- #
# Regression: tool arguments encoded as pseudo-XML instead of JSON
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_evidence", [
    '\n<item>\n<kind>both</kind>\n<claim>x.</claim></item>\n</caveats>\n',
    '\n<parameter name="item">{"kind":"statistic"}\n',
    '\n<invoke name="x">\n</invoke>\n',
])
def test_list_field_as_a_string_is_a_malformed_call(bad_evidence):
    """Detection is structural, so it survives a new tag spelling."""
    from sentinel.explainer import MalformedToolCall, parse_explanation

    with pytest.raises(MalformedToolCall):
        parse_explanation({"headline": "x", "evidence": bad_evidence})


def test_malformed_call_feedback_names_the_encoding_not_the_content(result, pack):
    from .conftest import FakeResponse, FakeTextBlock, FakeToolUse

    def xml_args(_):
        return FakeResponse(content=[FakeToolUse(
            "submit_explanation",
            {"headline": "x", "evidence": "\n<item>\n<kind>both</kind>\n</item>\n"},
        )], stop_reason="tool_use")

    client = FakeClient([xml_args, text_response(_payload()), critic_pass()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake", offer_tool=True)),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    outcome = workflow.run(result)
    assert outcome.generator == "llm"
    feedback = client.messages.requests[1]["messages"][-1]["content"][-1]["text"]
    assert "JSON arrays" in feedback and "only fix the encoding" in feedback


def test_the_tool_is_not_offered_by_default():
    """Offering it cost a retry on live runs and buys nothing it cannot force."""
    assert ExplainerConfig().offer_tool is False


def test_no_tools_key_is_sent_when_the_tool_is_off(result, pack):
    client = FakeClient([text_response(_payload()), critic_pass()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    workflow.run(result)
    assert "tools" not in client.messages.requests[0]


def test_profile_sample_size_is_a_citable_fact(sheet):
    """Proposal 3.2's sample-size caveat needs a fact, or it costs a retry."""
    fact = sheet.get("profiles.batteries_per_type")
    assert fact is not None
    assert fact.display == "28 to 31"
    assert "13%" in (fact.note or "")


def test_debug_shows_every_finding_not_just_a_summary(result, pack):
    """report() says an attempt failed; debug() must say why."""
    bad = _payload(headline="Porosity is 34% so this is type A.")
    client = FakeClient([text_response(bad), text_response(_payload()), critic_pass()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    text = workflow.run(result).debug()
    assert "VALIDATOR rejected it" in text
    assert "NUM001" in text and "TYPE002" in text
    assert "fix:" in text                      # the hint is carried through
    assert "CRITIC passed" in text
    assert "delivered" in text and "facts used" in text


def test_debug_reports_a_critic_rejection(result, pack):
    client = FakeClient([
        text_response(_payload()), critic_fail(),
        text_response(_payload()), critic_pass(),
    ])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    text = workflow.run(result).debug()
    assert "CRITIC rejected it" in text
    assert "CRIT_CONF" in text


# --------------------------------------------------------------------------- #
# Regression: the critic must be shown the sources it is asked to judge
# --------------------------------------------------------------------------- #
def test_critic_receives_the_cited_sources(result, pack, good_draft):
    """Judging quote relevance without the source produced confident nonsense."""
    from sentinel.critic import cited_sources

    client = FakeClient([critic_pass()])
    ClaudeCritic(client, CriticConfig(model="fake")).review(
        build_fact_sheet(result), good_draft, pack
    )
    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    assert "Sources this explanation cites" in sent
    for sid in cited_sources(good_draft, pack):
        assert f"## [{sid}]" in sent
        assert pack.sources[sid].body[:60] in sent


def test_only_cited_sources_are_sent(result, pack, good_draft):
    from sentinel.critic import cited_sources

    cited = cited_sources(good_draft, pack)
    assert cited                                  # the fixture cites something
    assert set(cited) < set(pack.ids())           # but not the whole pack


def test_critic_is_told_placeholders_are_already_verified(result, pack, good_draft):
    client = FakeClient([critic_pass()])
    ClaudeCritic(client, CriticConfig(model="fake")).review(
        build_fact_sheet(result), good_draft, pack
    )
    system = client.messages.requests[0]["system"]
    assert "already been checked to exist" in system
    assert "does not prove the type" in system


def test_critic_without_a_pack_is_told_check_c_does_not_apply(result, pack):
    draft = Explanation(
        headline="x", why_not_runner_up="y",
        evidence=[EvidenceItem(kind="statistic", claim="a", facts=["cls.fused.B"])],
    )
    client = FakeClient([critic_pass()])
    ClaudeCritic(client, CriticConfig(model="fake")).review(
        build_fact_sheet(result), draft, None
    )
    sent = client.messages.requests[0]["messages"][0]["content"][0]["text"]
    assert "check C does not apply" in sent


def test_fact_coverage_counts_placeholders_in_prose_too(result, pack):
    """Counting only evidence[].facts undercounted coverage by about half."""
    draft = _payload(
        headline="Type {pred_type} at {cls.fused.B}.",
        why_not_runner_up="Against {cls.fused.A}, with {stats.BSE.d50.z_A} on d50.",
    )
    client = FakeClient([text_response(draft), critic_pass()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    used = workflow.run(result).audit["facts_used"]
    # Referenced only in prose, never declared in an evidence item:
    assert "cls.fused.B" in used
    assert "cls.fused.A" in used
    assert "stats.BSE.d50.z_A" in used


def test_declared_but_unreferenced_facts_are_reported(result, pack):
    """Listing a fact you never use is a way to look grounded without being it."""
    draft = _payload()
    draft["evidence"][0]["facts"].append("stats.BSE.solidity.value")   # never cited
    client = FakeClient([text_response(draft), critic_pass()])
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(),
        explainer=ClaudeExplainer(client, ExplainerConfig(model="fake")),
        critic=ClaudeCritic(client, CriticConfig(model="fake")),
    )
    audit = workflow.run(result).audit
    assert "stats.BSE.solidity.value" in audit["facts_declared_unused"]
