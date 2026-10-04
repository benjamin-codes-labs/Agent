"""Sentinel v3 -- Step 6, the explainer agent.

The workflow: evidence builder (code) -> Claude Sonnet 5.5 explainer (vision)
-> code validator -> Claude Haiku 4.5 grounding critic -> renderer, with up to
two retries on any failed check and a Jinja2 template fallback.

Quick start::

    from sentinel import KnowledgePack, BatteryResult, explain_battery, collect_images

    result = BatteryResult.model_validate_json(open("BAT-07.json").read())
    pack = KnowledgePack.from_path("knowledge/pack.md")
    outcome = explain_battery(result, pack, collect_images(result, root="runs"))
    print(outcome.report())
    print(outcome.explanation.headline)
"""

from .contract import (
    BatteryResult,
    Citation,
    Classification,
    DetectorMaps,
    EvidenceItem,
    Explanation,
    ImageRecord,
    MapRegion,
    PhaseFractions,
    Profile,
    SignOff,
    StatisticRecord,
    StatsClassifier,
    TypeProbs,
    Versions,
)
from .critic import ClaudeCritic, CriticConfig, CriticVerdict, NullCritic
from .explainer import ClaudeExplainer, ExplainerConfig, ExplainerError
from .facts import EvidenceBuilder, Fact, FactSheet, build_fact_sheet
from .knowledge import KnowledgePack, Source
from .prompts import ImageAsset, collect_images
from .questions import GroundedAnswer, QuestionOutcome, answer_question
from .pipeline_facts import FactsDocument, FactsOutcome, FactsResponse, PipelineSample, explain_facts_document, parse_facts_document
from .renderer import render
from .template import TemplateExplainer, build_template_explanation
from .textrules import TextRuleConfig
from .tools import ReadOnlyTools, run_tool_loop
from .validator import Finding, ValidationReport, Validator, ValidatorConfig, validate
from .workflow import (
    AttemptRecord,
    ExplainOutcome,
    ExplainerWorkflow,
    WorkflowConfig,
    explain_battery,
)

__all__ = [
    "AttemptRecord",
    "BatteryResult",
    "Citation",
    "ClaudeCritic",
    "ClaudeExplainer",
    "Classification",
    "CriticConfig",
    "CriticVerdict",
    "DetectorMaps",
    "EvidenceBuilder",
    "EvidenceItem",
    "ExplainOutcome",
    "ExplainerConfig",
    "ExplainerError",
    "ExplainerWorkflow",
    "Explanation",
    "Fact",
    "FactSheet",
    "FactsDocument",
    "FactsOutcome",
    "FactsResponse",
    "PipelineSample",
    "Finding",
    "GroundedAnswer",
    "ImageAsset",
    "ImageRecord",
    "KnowledgePack",
    "MapRegion",
    "NullCritic",
    "PhaseFractions",
    "Profile",
    "QuestionOutcome",
    "ReadOnlyTools",
    "SignOff",
    "Source",
    "StatisticRecord",
    "StatsClassifier",
    "TemplateExplainer",
    "TextRuleConfig",
    "TypeProbs",
    "ValidationReport",
    "Validator",
    "ValidatorConfig",
    "Versions",
    "WorkflowConfig",
    "answer_question",
    "build_fact_sheet",
    "build_template_explanation",
    "collect_images",
    "explain_battery",
    "explain_facts_document",
    "parse_facts_document",
    "render",
    "run_tool_loop",
    "validate",
]

__version__ = "0.1.0"
