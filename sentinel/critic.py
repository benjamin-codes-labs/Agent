"""Grounding critic (proposal section 6.4), on ``claude-haiku-4-5-20251001``.

The critic catches what code cannot see: the attention map used as evidence,
certainty beyond the confidence label, and a materials claim its quote does not
support.

One caution, carried over from the wider literature and worth stating in the
pitch: entailment judgement on scientific text is weak -- published checkers
score near chance on scientific claim verification. So the critic is treated as
a one-way gate. It can **fail** an explanation; it can never **pass** one the
code validator failed, and its verdict is recorded separately from the
validator's so the two numbers can be reported honestly rather than as one
"checked" figure. ``blocking=False`` keeps its findings as warnings, which is
the right setting if it turns out to be noisy on the day.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .contract import Explanation
from .env import ensure_api_key
from .explainer import ModelCall, extract_json_object
from .facts import FactSheet
from .knowledge import KnowledgePack
from .prompts import CRITIC_ROLE, CRITIC_TOOL, build_critic_message
from .validator import Finding

def cited_sources(
    explanation: Explanation, pack: KnowledgePack | None
) -> dict[str, str]:
    """Full text of just the sources this explanation cites."""
    if pack is None:
        return {}
    wanted = {c.source for item in explanation.evidence for c in item.citations}
    return {
        sid: pack.sources[sid].body
        for sid in sorted(wanted)
        if sid in pack.sources
    }


KIND_LABELS = {
    "A": "the attention map is offered as class evidence",
    "B": "certainty stronger than the confidence label",
    "C": "a materials claim its quote does not support",
    "D": "an unlicensed causal or mechanistic story",
    "E": "a statement contradicting the fact sheet",
}
KIND_CODES = {
    "A": "CRIT_ATTN",
    "B": "CRIT_CONF",
    "C": "CRIT_QUOTE",
    "D": "CRIT_CAUSAL",
    "E": "CRIT_CONTRA",
}


@dataclass
class CriticVerdict:
    #: Whether the critic lets the explanation through. With
    #: ``blocking=False`` this is always True even when the critic objected.
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    call: ModelCall | None = None
    skipped_reason: str | None = None
    #: The critic's own opinion, independent of whether it was allowed to
    #: block. This is what the JSON contract records and the platform shows, so
    #: a non-blocking critic's objection is still visible rather than buried.
    declared_passed: bool | None = None

    @property
    def ran(self) -> bool:
        return self.skipped_reason is None

    def feedback(self) -> str:
        if self.passed or not self.findings:
            return ""
        lines = ["A grounding critic flagged these problems. Fix each one:"]
        lines += [f.render() for f in self.findings]
        return "\n".join(lines)


@dataclass
class CriticConfig:
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1200
    timeout_seconds: float = 90.0
    max_transport_retries: int = 0
    #: Left None deliberately. A live run returned
    #: "Messages.create() got an unexpected keyword argument 'temperature'",
    #: so the non-default-sampling restriction the proposal notes for Sonnet 5.5
    #: applies here too. Sending it silently disabled the critic on every
    #: battery, which the audit reported only as critic_skipped_reason.
    temperature: float | None = None
    #: When True a critic failure blocks delivery and triggers a retry.
    blocking: bool = True
    #: Kinds the critic may block on. Narrow this if it proves noisy.
    blocking_kinds: tuple[str, ...] = ("A", "B", "C", "D", "E")

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("request timeout must be positive and finite")
        if self.max_transport_retries < 0:
            raise ValueError("transport retries must be non-negative")


class ClaudeCritic:
    def __init__(self, client: Any | None = None, config: CriticConfig | None = None):
        self.config = config or CriticConfig()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            ensure_api_key()
            self._client = anthropic.Anthropic(timeout=self.config.timeout_seconds,
                                               max_retries=self.config.max_transport_retries)
        return self._client

    def review(
        self,
        sheet: FactSheet,
        explanation: Explanation,
        pack: KnowledgePack | None = None,
    ) -> CriticVerdict:
        payload = explanation.model_dump(
            exclude={"generator", "validator_passed", "critic_passed"}
        )
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
            "system": CRITIC_ROLE,
            "tools": [CRITIC_TOOL],
            "tool_choice": {"type": "tool", "name": CRITIC_TOOL["name"]},
            "messages": [
                build_critic_message(
                    sheet,
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    cited_sources(explanation, pack),
                )
            ],
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        try:
            response = self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - the critic must never break a run
            return CriticVerdict(
                passed=True,
                skipped_reason=f"the critic call failed ({type(exc).__name__}: {exc})",
            )

        usage = getattr(response, "usage", None)
        call = ModelCall(
            model=getattr(response, "model", self.config.model),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
        verdict = self._read(response, call)
        return verdict

    def _read(self, response: Any, call: ModelCall) -> CriticVerdict:
        data: dict[str, Any] | None = None
        texts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                call.used_tool = True
                candidate = getattr(block, "input", None)
                if isinstance(candidate, dict):
                    data = candidate
                    break
            elif getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", "") or "")
        if data is None and texts:
            try:
                data = extract_json_object("\n".join(texts))
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            return CriticVerdict(
                passed=True, call=call,
                skipped_reason="the critic returned no verdict object",
            )

        problems = data.get("problems") or []
        findings: list[Finding] = []
        for problem in problems:
            if not isinstance(problem, dict):
                continue
            kind = str(problem.get("kind", "")).upper()[:1]
            if kind not in KIND_CODES:
                continue
            severity = "error" if (
                self.config.blocking and kind in self.config.blocking_kinds
            ) else "warning"
            findings.append(Finding(
                code=KIND_CODES[kind],
                severity=severity,  # type: ignore[arg-type]
                loc=str(problem.get("loc") or "explanation"),
                message=f"{KIND_LABELS[kind]}: {problem.get('why', '').strip()}",
                observed=(problem.get("quote") or None),
                fix_hint={
                    "A": "cite a numbered evidence region instead of the attention map",
                    "B": "soften the wording to match the confidence label",
                    "C": "quote a sentence that actually supports the claim, or drop it",
                    "D": "describe the measured difference without asserting its cause",
                    "E": "restate the claim to match the fact sheet",
                }[kind],
            ))

        declared_pass = bool(data.get("passed", not findings)) and not findings
        blocking = [f for f in findings if f.severity == "error"]
        # Two separate questions, deliberately: did the critic object, and is it
        # allowed to stop delivery? Its own boolean is not trusted over its own
        # findings, so a "passed: true" alongside problems still counts as an
        # objection.
        passed = True if not self.config.blocking else (declared_pass and not blocking)
        return CriticVerdict(
            passed=passed,
            findings=findings,
            call=call,
            declared_passed=declared_pass,
        )


class NullCritic:
    """Used when the critic is cut (it is first in the proposal's cut order)."""

    def review(
        self,
        sheet: FactSheet,
        explanation: Explanation,
        pack: KnowledgePack | None = None,
    ) -> CriticVerdict:
        return CriticVerdict(
            passed=True, skipped_reason="the critic is disabled", declared_passed=None
        )
