"""Fixtures and a fake Claude client, so the whole loop is testable offline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from sentinel import BatteryResult, KnowledgePack, build_fact_sheet

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fake SDK objects
# --------------------------------------------------------------------------- #
@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUse:
    name: str
    input: dict[str, Any]
    id: str = "toolu_fake"
    type: str = "tool_use"


@dataclass
class FakeUsage:
    input_tokens: int = 1000
    output_tokens: int = 200
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: list[Any]
    model: str = "fake-model"
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self, script: list[Callable[[dict[str, Any]], FakeResponse]]):
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError("the fake client ran out of scripted responses")
        step = self.script.pop(0)
        return step(kwargs)


class FakeClient:
    """Stands in for ``anthropic.Anthropic()``."""

    def __init__(self, script: list[Callable[[dict[str, Any]], FakeResponse]]):
        self.messages = FakeMessages(script)


def text_response(payload: dict[str, Any]) -> Callable[[dict[str, Any]], FakeResponse]:
    def step(_: dict[str, Any]) -> FakeResponse:
        return FakeResponse(content=[FakeTextBlock(json.dumps(payload))])
    return step


def tool_response(name: str, payload: dict[str, Any]):
    def step(_: dict[str, Any]) -> FakeResponse:
        return FakeResponse(
            content=[FakeToolUse(name=name, input=payload)], stop_reason="tool_use"
        )
    return step


def fenced_response(payload: dict[str, Any]):
    def step(_: dict[str, Any]) -> FakeResponse:
        body = "Here you go:\n```json\n" + json.dumps(payload) + "\n```\n"
        return FakeResponse(content=[FakeTextBlock(body)])
    return step


def critic_pass():
    return tool_response("report_grounding", {"passed": True, "problems": []})


def critic_fail(kind: str = "B", why: str = "too certain for a medium label"):
    return tool_response("report_grounding", {
        "passed": False,
        "problems": [{"kind": kind, "loc": "headline", "quote": "x", "why": why}],
    })


# --------------------------------------------------------------------------- #
# Data fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def result() -> BatteryResult:
    specs = [
        ("BSE", "porosity", "porosity", 0.271, "fraction", [(0.352, 0.021, -3.857), (0.264, 0.018, 0.389), (0.308, 0.034, -1.088)], -0.88),
        ("BSE", "d50", "median cross-sectional particle diameter", 6.42, "um", [(9.81, 1.12, -3.027), (6.15, 0.74, 0.365), (7.92, 1.41, -1.064)], -0.81),
        ("InLens", "cbd_fraction", "carbon-binder domain fraction", 0.186, "fraction", [(0.131, 0.015, 3.667), (0.179, 0.014, 0.5), (0.152, 0.022, 1.545)], 0.79),
        ("BSE", "solidity", "particle solidity", 0.934, "dimensionless", [(0.961, 0.011, -2.455), (0.929, 0.013, 0.385), (0.944, 0.019, -0.526)], -0.64),
        ("ETD", "crack_density", "crack density", 0.042, "um^-1", [(0.018, 0.009, 2.667), (0.039, 0.012, 0.25), (0.026, 0.014, 1.143)], 0.58),
    ]
    statistics = []
    for rank, (detector, name, label, value, unit, profiles, effect) in enumerate(specs, 1):
        statistics.append({"id": f"stats.{detector}.{name}", "detector": detector, "name": name,
            "label": label, "value": value, "unit": unit,
            "profiles": {kind: {"mean": mean, "sd": sd, "n": count}
                         for kind, (mean, sd, _), count in zip("ABC", profiles, (29, 31, 28))},
            "z": {kind: z for kind, (_, _, z) in zip("ABC", profiles)},
            "separates_pred_vs_runner_up_rank": rank, "cliffs_delta": effect, "primary_detector": detector})
    for kind, median in zip("ABC", (0.349, 0.266, 0.305)):
        statistics[0]["profiles"][kind]["median"] = median
    region_data = {
        "BSE": [(1, [12.4, 30.1, 28.9, 44.6], 239.2, 0.31, (0.214, 0.201, 0.585), (-0.057, 0.019, 0.038)),
                (2, [55.0, 8.2, 68.1, 21.7], 176.8, 0.18, (0.238, 0.192, 0.570), (-0.033, 0.010, 0.023))],
        "InLens": [(1, [40.2, 51.3, 52.8, 63.0], 147.4, 0.27, (0.201, 0.243, 0.556), (-0.014, 0.057, -0.043))],
    }
    return BatteryResult.model_validate({
        "battery_id": "BAT-07", "run_id": "2026-10-03T18:02:11Z", "scope": "v1",
        "versions": {"git": "0" * 40, "segmenters": {"BSE": "xgb_bse_v1", "ETD": "xgb_etd_v1", "InLens": "xgb_inlens_v1"},
                     "backbone": "dinov3_vitb16", "heads": "lr_v1", "llm": "claude-sonnet-5-5"},
        "images": [{"detector": detector, "file": f"BAT-07_{detector}.tif", "pixel_size_um": 0.0488,
                    "pixel_size_source": "fei_metadata", "kv": kv, "quality_flags": flags,
                    "entropy_flag": False, "masked_area_frac": masked}
                   for detector, kv, flags, masked in [("BSE", 5.0, [], 0.021), ("ETD", 5.0, ["charging_streaks"], 0.068), ("InLens", 2.0, [], 0.014)]],
        "classification": {"per_detector": {"BSE": {"A": 0.12, "B": 0.81, "C": 0.07},
                            "ETD": {"A": 0.28, "B": 0.63, "C": 0.09}, "InLens": {"A": 0.19, "B": 0.74, "C": 0.07}},
                           "fused": {"A": 0.197, "B": 0.727, "C": 0.076}, "predicted": "B", "runner_up": "A",
                           "detector_agreement": 3, "stats_classifier": {"predicted": "B", "probs": {"A": 0.24, "B": 0.69, "C": 0.07}},
                           "unlike_any_type": False, "confidence": "medium",
                           "confidence_reasons": ["a charging-streak quality flag was raised on the ETD image"]},
        "statistics": statistics,
        "maps": {detector: {"overview_png": f"runs/R1/BAT-07/{detector}_overview.png",
                            "crops_png": f"runs/R1/BAT-07/{detector}_crops.png",
                            "regions": [{"id": rid, "box_um": box, "area_um2": area, "share_of_evidence": share,
                                         "phase_fractions": dict(zip(("pore", "cbd", "am"), fractions)),
                                         "minus_whole_image": dict(zip(("pore", "cbd", "am"), deltas))}
                                        for rid, box, area, share, fractions, deltas in rows]}
                 for detector, rows in region_data.items()},
    })


@pytest.fixture(scope="session")
def pack() -> KnowledgePack:
    return KnowledgePack.from_path(ROOT / "knowledge" / "pack.md")


@pytest.fixture
def sheet(result):
    return build_fact_sheet(result)


@pytest.fixture
def good_draft(result, pack):
    """A valid explanation (template-derived, with a real citation)."""
    from sentinel.selftest import build_base

    draft, _ = build_base(result, pack)
    return draft


@pytest.fixture(autouse=True)
def _no_accidental_api_calls(monkeypatch, request):
    """Fail loudly instead of billing a real call.

    Every test here is meant to run against the fake client. Once a real key is
    configured, any test that forgets to inject one would silently hit the API
    and spend credits, so the real SDK constructor is blocked outright. A test
    that genuinely wants the network can opt out with
    @pytest.mark.allow_network.
    """
    if request.node.get_closest_marker("allow_network"):
        return

    import anthropic

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a test tried to construct a real anthropic client; inject a "
            "FakeClient, or mark the test with @pytest.mark.allow_network"
        )

    monkeypatch.setattr(anthropic, "Anthropic", refuse)


def evidence_facts(context, sample_id, prefer=None):
    """One displayed, numeric record fact per kind of evidence the record offers.

    The breadth rule requires an answer to draw on several kinds of evidence,
    so the fixture answers must too. Preferences keep values distinct between
    the two samples of the multi-sample fixtures.
    """
    from sentinel.pipeline_facts import EVIDENCE_KINDS, _displayed_facts
    own = f"context.{context.sample_sources[sample_id]}."
    shown = [fid for fid, fact in _displayed_facts(context) if fid.startswith(own)
             and isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool)]
    prefer = prefer or {"composition": ".pore", "measured features": "siox_frac.value",
                        "evidence location": "graphite"}
    out = {}
    for kind, prefixes in EVIDENCE_KINDS.items():
        candidates = [fid for fid in shown if fid[len(own):].startswith(prefixes)]
        if candidates:
            out[kind] = next((fid for fid in candidates if prefer.get(kind, "") in fid), candidates[0])
    return out


def comparison_payload(context):
    """A valid explanation for every sample: real image-versus-batch comparisons.

    Each basis/alternatives point sets this image's value against a batch value
    for the same measurement; two different measurements are compared; an
    opposing measurement is included when one exists. Samples without a
    baseline get a plain record-based answer.
    """
    out = []
    for sample in context.samples:
        sid = sample.sample_id
        comparison = context.baselines.get(sid, {})
        record = evidence_facts(context, sid)
        if comparison.get("status") != "compared":
            quote = {"source": context.guide_source, "quote": "Do not invent alternative-batch morphology."}
            facts = list(record.values())
            first, rest = facts[:2], facts[2:]
            out.append({"sample_id": sid, "reported_answer": sample.answer, "points": [
                {"kind": "basis", "text": "Record figures " + ", ".join("{" + f + "}" for f in first) + ".",
                 "facts": first} if first else
                {"kind": "basis", "text": "The record reports its own measurements.", "citations": [quote]},
                {"kind": "alternatives", "text": "Further figure " + ", ".join("{" + f + "}" for f in rest) + "."
                 if rest else "No batch comparison was available.", "facts": rest,
                 **({} if rest else {"citations": [quote]})},
                {"kind": "limitation", "text": "Nothing here excludes the other categories.", "citations": [quote]},
            ]})
            continue
        b = context.baseline_sources[sid]
        measurements = list(comparison["measurements"])
        opposing = comparison["summary"]["favours_an_alternative"]
        second = opposing[0] if opposing else measurements[-1]
        first = next((m for m in measurements if m != second), second)
        reported = comparison["reported_batches"][0]
        other = next(x for x in ("Batch_1", "Batch_2", "Batch_3") if x not in comparison["reported_batches"])
        m1, m2 = f"context.{b}.measurements.{first}", f"context.{b}.measurements.{second}"
        # Two more kinds of evidence from the record itself, for the breadth rule.
        extra = [record[k] for k in ("composition", "evidence location", "measured features") if k in record][:2]
        out.append({"sample_id": sid, "reported_answer": sample.answer, "points": [
            {"kind": "basis",
             "text": "This image measures {" + m1 + ".this_image}, against a median of {" + m1 + f".{reported}.median}} "
                     "in the reported category" + "".join("; record figure {" + f + "}" for f in extra[:1]) + ".",
             "facts": [f"{m1}.this_image", f"{m1}.{reported}.median", *extra[:1]]},
            {"kind": "alternatives",
             "text": "Here it measures {" + m2 + ".this_image}, against a median of {" + m2 + f".{other}.median}} "
                     "in another category" + "".join("; record figure {" + f + "}" for f in extra[1:2]) + ".",
             "facts": [f"{m2}.this_image", f"{m2}.{other}.median", *extra[1:2]]},
            {"kind": "limitation", "text": "Each profile rests on a handful of images.",
             "citations": [{"source": b, "quote": "Each batch profile rests on only a handful of images"}]},
        ]})
    return {"explanations": out}


def prompt_text(request) -> str:
    """The first user message as one string, whatever its block layout.

    The prompt is split into a cached stable block and a per-document block, so
    tests that look for content must not depend on which block holds it.
    """
    content = request["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return "\n\n".join(block.get("text", "") for block in content)
