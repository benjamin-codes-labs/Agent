"""Speed and density: smaller prompt, cached prefix, lower effort, point form.

None of this may cost information: every number the model is told about must
still be citable, and nothing hidden from the display may become invalid.
"""

import json
import re
from pathlib import Path

import pytest

from sentinel.baselines import load_default_baselines
from sentinel.explainer import ExplainerConfig
from sentinel.knowledge import KnowledgePack
from sentinel.pipeline_facts import (
    STYLE_RULES,
    _baseline_table,
    _displayed_facts,
    explain_facts_document,
    is_model_score_fact,
    parse_facts_document,
    prepare_facts_context,
    validate_facts_response,
    FactsResponse,
)
from .conftest import FakeClient, comparison_payload, critic_pass, text_response

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "examples" / "facts.json"
V3 = ROOT / "lucas-sem-analysis-v3" / "outputs" / "batchid" / "71vgq3fw.json"
needs_reference = pytest.mark.skipif(not V3.exists(), reason="reference outputs not present")


def _run(path, config=None, critic_steps=1):
    doc = Path(path).read_text()
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(doc), KnowledgePack.empty(), baselines=baselines)
    client = FakeClient([text_response(comparison_payload(context)), critic_pass()])
    explain_facts_document(doc, KnowledgePack.empty(), client=client, baselines=baselines, config=config)
    return context, client.messages.requests[0]


# --------------------------------------------------------------------------- #
# Compact baseline table: same information, IDs stated once
# --------------------------------------------------------------------------- #
@needs_reference
@pytest.mark.parametrize("path", [LEGACY, V3])
def test_every_id_the_table_describes_resolves(path):
    """The model builds IDs from the stated pattern; every one must exist."""
    context, _ = _run(path)
    sample = context.samples[0].sample_id
    table = _baseline_table(context, sample)
    base = f"context.{context.baseline_sources[sample]}.measurements"
    comparison = context.baselines[sample]
    batches = sorted({k for e in comparison["measurements"].values() for k in e if k.startswith("Batch_")})
    for key in comparison["measurements"]:
        assert key in table
        for field in ["this_image", "verdict"] + [f"{b}.{s}" for b in batches for s in ("median", "q1", "q3")]:
            assert f"{base}.{key}.{field}" in context.pack.context_facts, f"{key}.{field}"
    example = re.search(r"Example: \{([^}]+)\}", table).group(1)
    assert example in context.pack.context_facts


@needs_reference
def test_table_shows_the_same_numbers_a_placeholder_renders():
    context, _ = _run(V3)
    sample = context.samples[0].sample_id
    lines = _baseline_table(context, sample).splitlines()
    base = f"context.{context.baseline_sources[sample]}.measurements.bse_noise"
    facts = context.pack.context_facts
    head = lines.index(next(l for l in lines if l.startswith("### bse_noise ")))
    assert facts[f"{base}.this_image"].display in lines[head]
    batch3 = next(l for l in lines[head + 1:head + 4] if l.strip().startswith("Batch_3:"))
    for field in ("median", "q1", "q3", "min", "max", "lower_n", "n"):
        assert facts[f"{base}.Batch_3.{field}"].display in batch3, field


# --------------------------------------------------------------------------- #
# Fact table: shorter, but nothing hidden becomes invalid
# --------------------------------------------------------------------------- #
@needs_reference
def test_model_scores_and_duplicates_are_not_displayed():
    context, request = _run(V3)
    shown = {fid for fid, _ in _displayed_facts(context)}
    assert not any(is_model_score_fact(fid) for fid in shown)
    own = context.sample_sources[context.samples[0].sample_id]
    assert f"context.{own}.acquisition.probes.bse_noise_sigma" not in shown          # in the baseline table
    assert f"context.{own}.phases.three_class_pct.pore" not in shown                 # in the baseline table
    assert not any(fid.startswith(f"context.{context.baseline_sources[context.samples[0].sample_id]}.")
                   for fid in shown)                                                  # shown as a table instead
    assert not any(isinstance(context.pack.context_facts[fid].value, str) for fid in shown)  # in the raw document


@needs_reference
def test_a_hidden_fact_is_still_valid_if_cited():
    """Hiding is display only; it must never turn a correct citation into an error."""
    context, _ = _run(V3)
    own = context.sample_sources[context.samples[0].sample_id]
    hidden = f"context.{own}.phases.three_class_pct.pore"
    assert hidden in context.pack.context_facts
    payload = comparison_payload(context)
    payload["explanations"][0]["points"].insert(2, {
        "kind": "limitation", "text": "Pore area here is {" + hidden + "}.", "facts": [hidden]})
    errors = validate_facts_response(FactsResponse.model_validate(payload), context)
    assert not any("unknown fact" in e for e in errors), errors


def test_dino_mode_still_shows_its_scores():
    record = json.loads(LEGACY.read_text())
    context = prepare_facts_context(parse_facts_document(record, decision_mode="dino"), KnowledgePack.empty())
    assert any("dino_head" in fid for fid, _ in _displayed_facts(context))


# --------------------------------------------------------------------------- #
# Effort and caching
# --------------------------------------------------------------------------- #
@needs_reference
def test_medium_effort_is_sent_by_default():
    """Omitted, Sonnet 5.5 thinks at 'high' before writing anything."""
    _, request = _run(LEGACY)
    assert request["output_config"] == {"effort": "medium"}


@needs_reference
def test_effort_is_configurable_and_can_be_left_to_the_api():
    _, low = _run(LEGACY, config=ExplainerConfig(effort="low"))
    assert low["output_config"] == {"effort": "low"}
    _, unset = _run(LEGACY, config=ExplainerConfig(effort=None))
    assert "output_config" not in unset


@needs_reference
def test_stable_prefix_is_cached_and_identical_across_documents():
    _, first = _run(LEGACY)
    _, second = _run(V3)
    a, b = first["messages"][0]["content"], second["messages"][0]["content"]
    assert all(block.get("cache_control") == {"type": "ephemeral"} for block in a)
    assert first["system"] == second["system"]          # cache prefix starts with the system prompt
    assert a[0]["text"] == b[0]["text"]                  # ...and the shared block: one cache entry serves both
    assert a[1]["text"] != b[1]["text"]


@needs_reference
def test_a_repair_attempt_reuses_the_cached_first_message():
    doc = LEGACY.read_text()
    baselines = load_default_baselines(ROOT)
    context = prepare_facts_context(parse_facts_document(doc), KnowledgePack.empty(), baselines=baselines)
    bad = comparison_payload(context)
    bad["explanations"][0]["points"][0]["facts"] = []      # fails validation -> repair
    client = FakeClient([text_response(bad), text_response(comparison_payload(context)), critic_pass()])
    explain_facts_document(doc, KnowledgePack.empty(), client=client, baselines=baselines)
    first, repair = client.messages.requests[0], client.messages.requests[1]
    assert repair["messages"][0] == first["messages"][0]   # byte-identical prefix -> cache hit
    assert repair["system"] == first["system"]


# --------------------------------------------------------------------------- #
# Point form
# --------------------------------------------------------------------------- #
@needs_reference
def test_point_form_style_reaches_the_prompt():
    _, request = _run(LEGACY)
    assert STYLE_RULES.splitlines()[0] in request["system"]
    assert "never B1/B2" in request["system"]
    assert "SHORTEST exact span" in request["system"]


def test_dino_prompt_is_not_given_the_pipeline_style_rules():
    record = LEGACY.read_text()
    context = prepare_facts_context(parse_facts_document(record, decision_mode="dino"), KnowledgePack.empty())
    from .test_dino_only import payload
    client = FakeClient([text_response(payload(context)), critic_pass()])
    explain_facts_document(record, KnowledgePack.empty(), client=client, decision_mode="dino")
    assert STYLE_RULES.splitlines()[0] not in client.messages.requests[0]["system"]


def test_abbreviated_batch_names_are_still_caught():
    """'Batch_1/2' leaves a bare digit; the style rule exists because of this check."""
    from sentinel.textrules import scan_field
    _, hits = scan_field("Batch_1/2 medians are higher")
    assert any(h.kind == "digit" for h in hits)
