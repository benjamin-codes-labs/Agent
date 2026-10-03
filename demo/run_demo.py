"""End-to-end run over two mock batteries, printing every stage.

    python demo/run_demo.py            # scripted model turns, no API key needed
    python demo/run_demo.py --live     # real Sonnet 5.5 + Haiku 4.5 calls

IMPORTANT about --live vs the default. By default the two model turns are
**scripted stand-ins written by hand**, not output from Claude. Everything else
-- the fact sheet, the validation findings, the retry feedback, the rendering,
the audit trail -- is produced by the real code. Scripting the model lets the
demo reproduce specific failure modes on demand, which a live call cannot.
Any transcript taken from the default mode must say so.

With --live and ANTHROPIC_API_KEY set, the same workflow runs against the real
models and the printed drafts are genuinely theirs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.mock_data import bat_21, bat_42                      # noqa: E402
from sentinel import (                                          # noqa: E402
    BatteryResult,
    ExplainerWorkflow,
    KnowledgePack,
    WorkflowConfig,
    build_fact_sheet,
)
from sentinel.critic import ClaudeCritic, CriticConfig          # noqa: E402
from sentinel.explainer import ClaudeExplainer, ExplainerConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RULE = "=" * 78
THIN = "-" * 78


# --------------------------------------------------------------------------- #
# Minimal stand-ins for the SDK response objects
# --------------------------------------------------------------------------- #
@dataclass
class _Text:
    text: str
    type: str = "text"


@dataclass
class _ToolUse:
    name: str
    input: dict[str, Any]
    id: str = "toolu_demo"
    type: str = "tool_use"


@dataclass
class _Usage:
    input_tokens: int = 7400
    output_tokens: int = 520
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 3153


@dataclass
class _Response:
    content: list[Any]
    model: str = "claude-sonnet-5-5"
    stop_reason: str = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class ScriptedMessages:
    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            raise RuntimeError("the scripted client ran out of responses")
        return self.script.pop(0)(kwargs)


class ScriptedClient:
    def __init__(self, script):
        self.messages = ScriptedMessages(script)


def says(payload: dict[str, Any], *, cached: bool = False):
    """A scripted explainer turn returning JSON as text."""
    def step(_):
        usage = _Usage(cache_read_input_tokens=3153, cache_creation_input_tokens=0) \
            if cached else _Usage()
        return _Response(content=[_Text(json.dumps(payload, ensure_ascii=False))], usage=usage)
    return step


def critic(passed: bool, problems: list[dict[str, str]] | None = None):
    def step(_):
        return _Response(
            content=[_ToolUse("report_grounding",
                              {"passed": passed, "problems": problems or []})],
            model="claude-haiku-4-5-20251001", stop_reason="tool_use",
        )
    return step


# --------------------------------------------------------------------------- #
# BAT-21: a flawed first draft, then a corrected one
# --------------------------------------------------------------------------- #
# Four mistakes a model genuinely makes: a typed number, a bare type letter,
# an invented fact ID, and a paraphrased quote.
BAT21_ATTEMPT_1 = {
    "headline": "Classified as type C with a fused probability of 76%, "
                "which is a clear match.",
    "why_not_runner_up": "The particle size distribution is much broader than type A, "
                         "with a span roughly one and a half times the type A mean.",
    "evidence": [
        {
            "kind": "statistic", "detector": "BSE",
            "claim": "The median cross-sectional particle diameter is "
                     "{stats.BSE.d50.value}, almost exactly the type C mean, whereas "
                     "type A particles are about 2 um larger.",
            "facts": ["stats.BSE.d50.value", "stats.BSE.d50.z_C"],
            "regions": [],
            "citations": [],
        },
        {
            "kind": "both", "detector": "BSE",
            "claim": "The attention map concentrates on the coarse pore network in "
                     "BSE region 1, where the pore fraction reaches "
                     "{stats.BSE.pore_network.value}.",
            "facts": ["stats.BSE.pore_network.value"],
            "regions": ["BSE region 1"],
            "citations": [{
                "source": "S3",
                "quote": "Calendering reduces the porosity and flattens the pores.",
            }],
        },
    ],
    "agreement": "All three detectors and the statistics branch agree on type C.",
    "caveats": ["The ETD image was slightly blurred."],
}

BAT21_ATTEMPT_2 = {
    "headline": "Classified as type {pred_type} with a fused probability of "
                "{cls.fused.C}, at {cls.confidence} confidence.",
    "why_not_runner_up": "Particle size is what separates type {pred_type} from type "
                         "{runner_up} here: the median cross-sectional diameter is "
                         "{stats.BSE.d50.value}, sitting {stats.BSE.d50.z_C} from the "
                         "type {pred_type} profile but {stats.BSE.d50.z_A} from type "
                         "{runner_up}, a difference of {stats.BSE.d50.delta_vs_A} "
                         "against the type {runner_up} mean.",
    "evidence": [
        {
            "kind": "statistic", "detector": "BSE",
            "claim": "The median cross-sectional particle diameter is "
                     "{stats.BSE.d50.value}, against a type {pred_type} mean of "
                     "{stats.BSE.d50.mean_C} and a type {runner_up} mean of "
                     "{stats.BSE.d50.mean_A}.",
            "facts": ["stats.BSE.d50.value", "stats.BSE.d50.mean_C",
                      "stats.BSE.d50.mean_A", "stats.BSE.d50.z_C"],
            "regions": [],
            "citations": [{
                "source": "S5",
                "quote": "Percentiles derived from cross-sections are therefore "
                         "useful for comparing samples prepared and imaged the same "
                         "way, and are not comparable with a laser diffraction "
                         "distribution supplied by a material vendor.",
            }],
        },
        {
            "kind": "statistic", "detector": "BSE",
            "claim": "The particle size span is {stats.BSE.span.value}, which is "
                     "{stats.BSE.span.z_C} from the type {pred_type} profile and "
                     "{stats.BSE.span.z_A} from type {runner_up}: a notably broader "
                     "distribution than type {runner_up} shows.",
            "facts": ["stats.BSE.span.value", "stats.BSE.span.z_C",
                      "stats.BSE.span.z_A"],
            "regions": [],
            "citations": [],
        },
        {
            "kind": "both", "detector": "BSE",
            "claim": "BSE region 1 carries {region.BSE.1.share_of_evidence} of the "
                     "positive evidence, and its local pore fraction of "
                     "{region.BSE.1.pore} is {region.BSE.1.pore_delta} above the "
                     "whole-image value, so the coarse open pore network is part of "
                     "what distinguishes this sample.",
            "facts": ["region.BSE.1.share_of_evidence", "region.BSE.1.pore",
                      "region.BSE.1.pore_delta"],
            "regions": ["BSE region 1"],
            "citations": [{
                "source": "S1",
                "quote": "Because the signal depends on composition rather than on "
                         "surface shape, a backscattered electron image is the "
                         "preferred basis for measuring the active material fraction, "
                         "particle size and particle shape.",
            }],
        },
        {
            "kind": "statistic", "detector": "InLens",
            "claim": "The carbon-binder domain fraction is "
                     "{stats.InLens.cbd_fraction.value}, close to the type "
                     "{pred_type} profile at {stats.InLens.cbd_fraction.z_C}.",
            "facts": ["stats.InLens.cbd_fraction.value",
                      "stats.InLens.cbd_fraction.z_C"],
            "regions": ["InLens region 1"],
            "citations": [{
                "source": "S2",
                "quote": "In practice the in-column image is the better basis for "
                         "judging binder coverage and binder texture, while the "
                         "chamber detector is the better basis for pore edges and "
                         "for cross-checking cracks.",
            }],
        },
    ],
    "agreement": "Detectors predicting the fused type: {cls.detector_agreement}. The "
                 "statistics branch independently reaches the same type, which is "
                 "what supports the {cls.confidence} label rather than a lower one.",
    "caveats": [
        "The confidence label is {cls.confidence} because the ETD image carries a "
        "sharpness flag, so its contribution to the fusion is down-weighted.",
        "Particle size percentiles here are cross-sectional and carry stereological "
        "bias, so they are comparable between these samples but not with a vendor's "
        "laser-diffraction figures.",
        "The evidence map shows where the class scores separate, not a causal "
        "mechanism; the processing route behind this microstructure is not measured.",
    ],
}

# --------------------------------------------------------------------------- #
# BAT-42: three drafts that each hide something the rules force into the open
# --------------------------------------------------------------------------- #
BAT42_BASE = {
    "headline": "Classified as type {pred_type} with a fused probability of "
                "{cls.fused.C}.",
    "why_not_runner_up": "Porosity is {stats.BSE.porosity.value}, "
                         "{stats.BSE.porosity.delta_vs_A} above the type {runner_up} "
                         "mean.",
    "evidence": [
        {
            "kind": "statistic", "detector": "BSE",
            "claim": "Porosity is {stats.BSE.porosity.value}, which is "
                     "{stats.BSE.porosity.z_C} from the type {pred_type} profile.",
            "facts": ["stats.BSE.porosity.value", "stats.BSE.porosity.z_C"],
            "regions": [], "citations": [{
                "source": "S3",
                "quote": "Calendering compresses a dried electrode coating to a "
                         "target thickness, which reduces total porosity and tends "
                         "to flatten and align the remaining pore space.",
            }],
        },
        {
            "kind": "statistic", "detector": "BSE",
            "claim": "The median cross-sectional particle diameter is "
                     "{stats.BSE.d50.value}, {stats.BSE.d50.delta_vs_A} above the "
                     "type {runner_up} mean.",
            "facts": ["stats.BSE.d50.value", "stats.BSE.d50.delta_vs_A"],
            "regions": [], "citations": [],
        },
    ],
    "agreement": "The available detectors both favour type {pred_type}.",
    "caveats": ["Only two detectors were available for this sample."],
}


def bat42_attempt(n: int) -> dict[str, Any]:
    """Three drafts, each still concealing something different."""
    draft = json.loads(json.dumps(BAT42_BASE))
    if n == 1:
        # Omits: unlike-any-type, the branch disagreement, the low-confidence hedge.
        draft["headline"] = ("This sample is confidently type {pred_type} at "
                             "{cls.fused.C}.")
    elif n == 2:
        # Discloses the detector gap, still hides the disagreement and the flag.
        draft["caveats"].append(
            "The InLens segmentation entropy is above the training range."
        )
    else:
        # Mentions the disagreement but still never says "unlike any type",
        # and still will not hedge.
        draft["agreement"] = ("The statistics branch differs from the classification "
                              "branch on this sample.")
        draft["caveats"].append("Results should be reviewed.")
    return draft


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_explanation(exp, indent: str = "  ") -> None:
    print(f"{indent}HEADLINE   {exp.headline}")
    print(f"{indent}WHY NOT    {exp.why_not_runner_up}")
    for i, item in enumerate(exp.evidence):
        tag = f"{item.kind}/{item.detector or '-'}"
        print(f"{indent}EVIDENCE {i} [{tag}]")
        print(f"{indent}   {item.claim}")
        if item.facts:
            print(f"{indent}   facts:   {', '.join(item.facts)}")
        if item.regions:
            print(f"{indent}   regions: {', '.join(item.regions)}")
        for c in item.citations:
            print(f"{indent}   cites:   [{c.source}] \"{c.quote[:88]}...\"")
    print(f"{indent}AGREEMENT  {exp.agreement}")
    for c in exp.caveats:
        print(f"{indent}CAVEAT     {c}")


def run_one(
    result: BatteryResult,
    pack: KnowledgePack,
    script,
    *,
    live: bool,
    note: str,
) -> None:
    banner(f"BATTERY {result.battery_id}   --   {note}")

    sheet = build_fact_sheet(result)
    cls = result.classification
    print(f"  Step 4 says:  predicted {cls.predicted}, runner-up {cls.runner_up}, "
          f"fused {cls.fused.top()[1]:.1%}, confidence {cls.confidence}")
    print(f"  detectors:    {', '.join(result.available_detectors())}"
          + (f"   MISSING: {', '.join(result.missing_detectors())}"
             if result.missing_detectors() else ""))
    print(f"  second branch: {cls.stats_classifier.predicted if cls.stats_classifier else '-'}"
          f"   agrees={cls.stats_branch_agrees}")
    print(f"  unlike any training type: {cls.unlike_any_type}")
    print(f"  fact sheet:   {len(sheet.facts)} facts, "
          f"{len(sheet.region_ids)} evidence regions")

    if live:
        explainer = ClaudeExplainer(config=ExplainerConfig())
        crit = ClaudeCritic(config=CriticConfig())
    else:
        client = ScriptedClient(script)
        explainer = ClaudeExplainer(client, ExplainerConfig(model="claude-sonnet-5-5"))
        crit = ClaudeCritic(client, CriticConfig(model="claude-haiku-4-5-20251001"))

    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(max_retries=2), explainer=explainer, critic=crit
    )
    outcome = workflow.run(result)

    print(f"\n{THIN}\n  THE LOOP\n{THIN}")
    for att in outcome.attempts:
        print(f"  {att.summary()}")
        if att.findings:
            for f in att.findings[:8]:
                sev = "ERROR  " if f.severity == "error" else "warning"
                print(f"     {sev} [{f.code}] {f.loc}")
                print(f"             {f.message}")
                if f.observed:
                    print(f"             observed: {f.observed[:92]}")
        if att.critic and att.critic.findings:
            for f in att.critic.findings:
                print(f"     CRITIC  [{f.code}] {f.loc}: {f.message}")

    print(f"\n{THIN}\n  DELIVERED  (generator={outcome.generator}, "
          f"validator_passed={outcome.validator_passed}, "
          f"critic_passed={outcome.critic_passed})\n{THIN}")
    show_explanation(outcome.explanation)

    print(f"\n{THIN}\n  AUDIT TRAIL\n{THIN}")
    for key in ("git", "knowledge_pack", "prompt", "explainer_model", "critic_model",
                "critic_declared_passed", "attempts", "fact_count", "fact_coverage",
                "citations_used", "validator_errors"):
        print(f"  {key:<24} {outcome.audit.get(key)}")

    out = ROOT / "demo" / "out" / f"{result.battery_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(outcome.to_json(), encoding="utf-8")
    print(f"\n  written to demo/out/{result.battery_id}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="call the real Sonnet 5.5 and Haiku 4.5 (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    pack = KnowledgePack.from_path(ROOT / "knowledge" / "pack.md")
    print(f"knowledge pack: {len(pack)} sources {pack.ids()}  version {pack.version}")
    if not args.live:
        print("\n*** The two model turns below are SCRIPTED STAND-INS, hand-written to")
        print("*** reproduce specific failure modes. They are NOT Claude's output.")
        print("*** Everything else is produced by the real code. Use --live with an")
        print("*** API key for genuine model drafts.")

    run_one(
        bat_21(), pack,
        [says(BAT21_ATTEMPT_1), says(BAT21_ATTEMPT_2, cached=True), critic(True)],
        live=args.live,
        note="clean case: a flawed first draft, repaired on retry",
    )
    run_one(
        bat_42(), pack,
        [says(bat42_attempt(1)), says(bat42_attempt(2)), says(bat42_attempt(3))],
        live=args.live,
        note="hard case: missing detector, branches disagree, unlike any type",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
