import json
from pathlib import Path

import pytest
from PIL import Image

from sentinel import KnowledgePack
from sentinel.explainer import ExplainerError
from .conftest import FakeClient, text_response


@pytest.fixture
def project_knowledge(tmp_path):
    reference = {
        "meta": {"title": "Si/C reference", "verification_method": {"source": "title/abstract level; no full text read"}},
        "features": [{"id": "D1", "name": "Porosity", "linked_properties_and_mechanism": "Pore structure affects transport.",
                      "favorable_state": "An illustrative range of 20–40%.",
                      "verification": {"status": "partially_confirmed", "supporting_works": ["W123"]}}],
        "imaging_modes": [{"id": "bse", "name": "BSE", "cautions": "Grey levels need calibration.",
                           "verification": {"status": "method_guidance_retained", "supporting_works": []}}],
        "references": [{"id": "W123", "title": "A study of electrodes", "doi": "10.example/test"}],
        "deleted_items": [{"id": "F1", "claim": "DELETED CLAIM MUST NOT BE USED"}],
    }
    comparison = {
        "batches": ["Batch_1", "Batch_2"], "caveats": ["Phase labels are unvalidated."],
        "excluded_images": [{"image": "img_x_SE.tif", "reason": "SE excluded"}],
        "detectors": {"bse": {
            "target_pixel_nm": 25.0, "caveats": [], "missing_batches": [],
            "batches": {
                "Batch_1": {"phase_labels_validated": False, "images": [], "phases": {"pore": {"phi": 0.30, "n_images": 7}}},
                "Batch_2": {"phase_labels_validated": False, "images": [], "phases": {"pore": {"phi": 0.27, "n_images": 6}}},
            },
            "comparisons": [{"lot_a": "Batch_1", "lot_b": "Batch_2", "flagged_images": ["img_x_BSE.tif"],
                             "phases": [{"phase": "pore", "delta": -0.03, "delta_rel": -0.10,
                                         "interval95": [-0.04, -0.02], "caveats": ["Imaging may explain part of the difference."]}]}],
        }},
    }
    (tmp_path / "SiC_SEM_reference_verified.json").write_text(json.dumps(reference))
    (tmp_path / "all_batch_comparisons.json").write_text(json.dumps(comparison))
    return KnowledgePack.from_project(tmp_path)


def test_project_jsons_become_separate_traceable_sources(project_knowledge):
    pack = project_knowledge
    text = pack.to_prompt_block()
    assert "partially_confirmed" in text
    assert "method_guidance_retained" in text
    assert "no full text read" in text
    assert "Phase labels are unvalidated" in text
    assert "A study of electrodes" in text
    assert "DELETED CLAIM MUST NOT BE USED" not in text
    assert any("#/features/0" in source.metadata["origin"] for source in pack.sources.values())
    delta = next(fact for fid, fact in pack.context_facts.items() if fid.endswith("phases.0.delta"))
    assert delta.display == "-3.00 percentage points"
    assert "all_batch_comparisons.json" in delta.note
    assert delta.value == -0.03


def test_project_selection_keeps_scope_and_batch_caveats(project_knowledge):
    selected = project_knowledge.select("BSE porosity", limit=1)
    assert "title/abstract" in selected.to_prompt_block()
    assert "Imaging may explain" in selected.to_prompt_block()
    assert len(selected.context_facts) > 0
    assert all(fid.split(".")[1] in selected.sources for fid in selected.context_facts)


def test_sample_preserves_native_schema_and_acquisition_warning(project_knowledge):
    pack = project_knowledge.with_sample({
        "sample_id": "img_71vgq3fw", "predicted_batch": "Batch_3",
        "probabilities": {"fused_F": {"Batch_3": 0.707}},
        "confidence": {"tier": "High"},
        "acquisition": {"warning": "Acquisition alone predicts batch."},
        "features_S": {"siox_ecd_aw": {"value": 3.8472}},
    })
    text = pack.to_prompt_block()
    assert "Acquisition alone predicts batch." in text
    assert "0.707" in text
    fact = next(f for fid, f in pack.context_facts.items() if fid.endswith("siox_ecd_aw.value"))
    assert "um" not in fact.display
    assert "µm" not in fact.display
    assert not any(s.metadata.get("kind") == "sample" for s in project_knowledge.sources.values())


def _grounded_payload(pack):
    fact = next(f for fid, f in pack.context_facts.items() if fid.endswith("phases.0.delta"))
    return {"answerable": True, "paragraphs": [{
        "text": "The reported pore-fraction difference is {" + fact.id + "}.",
        "facts": [fact.id], "citations": [], "images": [],
    }]}


def test_answer_uses_checked_facts_and_plain_text(project_knowledge):
    from sentinel.questions import answer_question
    from main import format_answer

    client = FakeClient([text_response(_grounded_payload(project_knowledge))])
    outcome = answer_question("Compare BSE porosity.", project_knowledge, client=client)
    assert format_answer(outcome.answer.model_dump_json()) == "The reported pore-fraction difference is -3.00 percentage points."
    request = client.messages.requests[0]
    assert "tool_choice" not in request and "temperature" not in request
    assert outcome.audit["validator_passed"] is True
    assert outcome.audit["semantic_critic_ran"] is False
    assert "Acquisition" in request["system"]


@pytest.mark.parametrize("bad_text", ["Porosity differs by 99%.", "Porosity is {invented.value}."])
def test_answer_retries_invented_numbers(project_knowledge, bad_text):
    from sentinel.questions import answer_question

    payload = _grounded_payload(project_knowledge)
    payload["paragraphs"][0]["text"] = bad_text
    client = FakeClient([text_response(payload), text_response(_grounded_payload(project_knowledge))])
    outcome = answer_question("Compare porosity", project_knowledge, client=client)
    assert outcome.audit["attempts"] == 2
    assert len(client.messages.requests) == 2


def test_answer_rejects_fabricated_reference_quote(project_knowledge):
    from sentinel.questions import answer_question

    payload = _grounded_payload(project_knowledge)
    payload["paragraphs"][0]["citations"] = [{"source": "S2", "quote": "Made-up scientific statement."}]
    client = FakeClient([text_response(payload)] * 3)
    with pytest.raises(ExplainerError, match="grounding"):
        answer_question("Compare porosity", project_knowledge, client=client)


def test_answer_rejects_visual_claim_without_attachment(project_knowledge):
    from sentinel.questions import answer_question

    payload = _grounded_payload(project_knowledge)
    payload["paragraphs"][0]["images"] = ["BSE"]
    with pytest.raises(ExplainerError, match="grounding"):
        answer_question("Describe image", project_knowledge, client=FakeClient([text_response(payload)] * 3))


def test_raw_tiff_is_resized_and_labelled_without_fake_evidence_map(tmp_path):
    import base64
    import io
    from sentinel.prompts import ImageAsset

    path = tmp_path / "img_cell_BSE.tif"
    Image.new("RGB", (2000, 1000), "grey").save(path)
    asset = ImageAsset(detector="BSE", kind="raw", path=path)
    block = asset.to_block()
    assert block["source"]["media_type"] == "image/png"
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as preview:
        assert max(preview.size) <= 1568
    assert "raw" in asset.describe()
    assert "red towards" not in asset.describe()


def test_workflow_receives_batch_facts_before_validation(result, project_knowledge):
    from sentinel import ExplainerWorkflow, WorkflowConfig
    from sentinel.explainer import ClaudeExplainer
    from .test_pipeline import _payload

    fact = next(f for fid, f in project_knowledge.context_facts.items() if fid.endswith("phases.0.delta"))
    payload = _payload()
    payload["evidence"][0].update(
        claim="External batch pore-fraction difference is {" + fact.id + "}; this is not a measurement of this sample.",
        facts=[fact.id], citations=[{"source": "S2", "quote": "Pore structure affects transport."}],
    )
    payload["evidence"][1]["citations"] = []
    workflow = ExplainerWorkflow(project_knowledge, WorkflowConfig(use_critic=False),
                                 explainer=ClaudeExplainer(FakeClient([text_response(payload)])))
    outcome = workflow.run(result)
    assert outcome.generator == "llm" and outcome.validator_passed
    assert "-3.00 percentage points" in outcome.explanation.evidence[0].claim
    assert set(project_knowledge.context_facts) <= set(outcome.sheet.facts)
    assert "External context" in outcome.sheet.to_prompt_block()


def test_question_selection_does_not_mix_detectors(project_knowledge):
    project_knowledge._add_record("ETD example", {"batch": "Batch_1", "phi": 0.9},
                                  "batch", "all_batch_comparisons.json#/detectors/etd/batches/Batch_1")
    selected = project_knowledge.select("BSE: compare Batch_1 with Batch_2")
    assert all(source.metadata.get("detector") == "bse" for source in selected.sources.values()
               if source.metadata.get("kind") in {"batch", "comparison"})
    assert all(fid.split(".")[1] in selected.sources for fid in selected.context_facts)


def test_reference_json_changes_change_the_audit_hash(project_knowledge):
    path = project_knowledge.path / "SiC_SEM_reference_verified.json"
    data = json.loads(path.read_text())
    data["features"][0]["verification"]["status"] = "confirmed"
    path.write_text(json.dumps(data))
    assert KnowledgePack.from_project(project_knowledge.path).version != project_knowledge.version


def test_answer_can_explain_missing_information(project_knowledge):
    from sentinel.questions import answer_question

    payload = {"answerable": False, "paragraphs": [{"text": "The supplied records do not include cycling measurements."}]}
    answer = answer_question("What is the cycle life?", project_knowledge,
                             client=FakeClient([text_response(payload)])).answer
    assert not answer.answerable
    assert "do not include" in answer.paragraphs[0].text


def test_a_real_reference_quote_is_accepted(project_knowledge):
    from sentinel.questions import answer_question

    payload = {"answerable": True, "paragraphs": [{
        "text": "Pore structure affects transport.",
        "citations": [{"source": "S2", "quote": "Pore structure affects transport."}],
    }]}
    result = answer_question("What does porosity affect?", project_knowledge,
                             client=FakeClient([text_response(payload)]))
    assert result.audit["sources_used"]["S2"]["origin"].endswith("#/features/0")


def test_json_input_path_is_not_automatically_attached(project_knowledge):
    from sentinel.questions import answer_question

    pack = project_knowledge.with_sample({"sample_id": "img_cell", "inputs": {"BSE": "missing.tif"}})
    client = FakeClient([text_response(_grounded_payload(pack))])
    outcome = answer_question("Review the sample", pack, client=client)
    assert outcome.audit["images_attached"] == []
    assert not any(block["type"] == "image" for block in client.messages.requests[0]["messages"][0]["content"])


def test_invalid_sample_and_empty_question_fail_before_api(project_knowledge):
    from sentinel.questions import answer_question

    with pytest.raises(ValueError, match="non-empty object"):
        project_knowledge.with_sample([])
    with pytest.raises(ValueError, match="empty"):
        answer_question(" ", project_knowledge)
    with pytest.raises(ValueError):
        project_knowledge.with_sample({"value": float("nan")})


def test_main_question_does_not_load_mock_classification(project_knowledge, tmp_path, monkeypatch, capsys):
    import main
    from sentinel.questions import answer_question

    monkeypatch.setattr(main.KnowledgePack, "from_project", lambda root: project_knowledge)
    client = FakeClient([text_response(_grounded_payload(project_knowledge))])
    monkeypatch.setattr(main, "answer_question", lambda question, pack, images: answer_question(question, pack, images, client=client))
    monkeypatch.setattr(main, "explain_battery", lambda *a: pytest.fail("Q&A must not use the mock classifier"))
    monkeypatch.chdir(tmp_path)
    main.main(question="Compare BSE porosity")
    output = capsys.readouterr().out
    assert "-3.00 percentage points" in output
    assert "BAT-07" not in output and "paragraphs" not in output


def test_main_sample_defaults_to_review_question(project_knowledge, tmp_path, monkeypatch, capsys):
    import main
    from sentinel.questions import answer_question

    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"sample_id": "img_71vgq3fw", "predicted_batch": "Batch_3"}))
    monkeypatch.setattr(main.KnowledgePack, "from_project", lambda root: project_knowledge)

    def answer(question, pack, images):
        assert "review" in question.lower()
        assert "img_71vgq3fw" in pack.to_prompt_block()
        return answer_question(question, pack, images, client=FakeClient([text_response(_grounded_payload(pack))]))

    monkeypatch.setattr(main, "answer_question", answer)
    main.main(sample=sample)
    assert "-3.00 percentage points" in capsys.readouterr().out
