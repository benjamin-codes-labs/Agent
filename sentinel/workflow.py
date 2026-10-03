"""The agentic workflow (proposal section 6, figure 2).

    evidence builder (code)
      -> explainer          Claude Sonnet 5.5, vision
      -> validator          code
      -> grounding critic   Claude Haiku 4.5
      -> renderer           code
      -> explanations/{battery}.json

"Any failed check: retry up to 2 times, then the template."

Loop control beyond the proposal
--------------------------------
A bare retry cap is not enough; a repair loop can burn its budget oscillating
between two broken drafts. Three guards are added:

* **Monotone progress.** An attempt must lower ``(errors, warnings)``
  lexicographically. Two consecutive non-improving attempts end the loop.
* **Oscillation detection.** The sorted multiset of ``(code, loc, observed)``
  is hashed per attempt; a repeat means the model is stuck and the loop stops.
* **Conversation replay.** The rejected draft is replayed as the assistant turn
  so the next attempt is a diff rather than a fresh draft.

Stopping early is not a loss: the template fallback is always available, always
passes, and is labelled as a template so nobody mistakes it for a cited
explanation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .contract import BatteryResult, Explanation, utc_now
from .critic import ClaudeCritic, CriticConfig, CriticVerdict, NullCritic
from .explainer import (
    ClaudeExplainer,
    ExplainerConfig,
    ExplainerError,
    MalformedToolCall,
    ModelCall,
    TruncatedResponse,
    assistant_echo,
    report_activity,
)
from .facts import EvidenceBuilder, FactSheet
from .knowledge import KnowledgePack
from .prompts import ImageAsset, prompt_fingerprint
from .textrules import PLACEHOLDER
from .renderer import render
from .template import TemplateExplainer, template_validator_config
from .validator import Finding, ValidationReport, Validator, ValidatorConfig

log = logging.getLogger("sentinel.explainer")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class AttemptRecord:
    """One pass through explainer -> validator -> critic."""

    attempt: int
    source: str                       # "llm" | "template"
    ok: bool = False
    error: str | None = None
    findings: list[Finding] = field(default_factory=list)
    critic: CriticVerdict | None = None
    call: ModelCall | None = None
    progress: tuple[int, int] | None = None

    def summary(self) -> str:
        if self.error:
            return f"attempt {self.attempt} ({self.source}): failed to produce JSON - {self.error}"
        n_err = sum(1 for f in self.findings if f.severity == "error")
        n_warn = len(self.findings) - n_err
        verdict = "passed" if self.ok else "rejected"
        bits = [f"attempt {self.attempt} ({self.source}): {verdict}",
                f"{n_err} error(s), {n_warn} warning(s)"]
        if self.critic and self.critic.ran:
            bits.append("critic " + ("passed" if self.critic.passed else "failed"))
        elif self.critic:
            bits.append(f"critic skipped: {self.critic.skipped_reason}")
        if self.call:
            bits.append(self.call.cost_note())
        return " | ".join(bits)


class GenerationFailed(ExplainerError):
    def __init__(self, attempts: list[AttemptRecord]):
        self.attempts = attempts
        details = []
        for attempt in attempts:
            details.append(attempt.summary())
            details.extend(finding.render() for finding in attempt.findings if finding.severity == "error")
            if attempt.critic and not attempt.critic.passed:
                details.append(attempt.critic.feedback())
        super().__init__("No checked AI review was produced; template fallback is disabled.\n" + "\n".join(details))


@dataclass
class ExplainOutcome:
    """Everything Step 6 produces for one battery."""

    battery_id: str
    run_id: str
    explanation: Explanation          # rendered: placeholders already substituted
    draft: Explanation                # pre-render, with placeholders intact
    sheet: FactSheet
    attempts: list[AttemptRecord] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def generator(self) -> str:
        return self.explanation.generator

    @property
    def validator_passed(self) -> bool:
        return bool(self.explanation.validator_passed)

    @property
    def critic_passed(self) -> bool | None:
        return self.explanation.critic_passed

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            {
                "battery_id": self.battery_id,
                "run_id": self.run_id,
                "explanation": self.explanation.model_dump(),
                "draft": self.draft.model_dump(),
                "audit": self.audit,
                "attempts": [a.summary() for a in self.attempts],
            },
            ensure_ascii=False,
            indent=indent,
        )

    def report(self) -> str:
        lines = [
            f"battery {self.battery_id} ({self.run_id})",
            f"  predicted {self.sheet.predicted} over {self.sheet.runner_up}, "
            f"confidence {self.sheet.confidence}",
            f"  generator: {self.generator}",
        ]
        lines += [f"  {a.summary()}" for a in self.attempts]
        return "\n".join(lines)

    def debug(self, *, raw: bool = False) -> str:
        """Everything needed to understand why a run took the path it did.

        ``report()`` says *that* an attempt was rejected; this says *what* was
        rejected and *why*, which is the only way to tell a real model mistake
        from a rule that is too strict. Set ``raw=True`` to include the model's
        unparsed text, for when the failure is in the response itself.
        """
        out: list[str] = [self.report(), ""]
        for att in self.attempts:
            out.append(f"--- attempt {att.attempt} ({att.source}) " + "-" * 44)
            if att.error:
                out.append(f"  the response was unusable: {att.error}")
            if att.call:
                out.append(f"  tokens: {att.call.cost_note()}  "
                           f"stop_reason={att.call.stop_reason}  "
                           f"used_tool={att.call.used_tool}")
            errors = [f for f in att.findings if f.severity == "error"]
            warnings = [f for f in att.findings if f.severity == "warning"]
            if errors:
                out.append(f"  VALIDATOR rejected it, {len(errors)} error(s):")
                out += [f"    {line}" for f in errors for line in f.render().splitlines()]
            if warnings:
                out.append(f"  validator warnings ({len(warnings)}):")
                out += [f"    {line}" for f in warnings for line in f.render().splitlines()]
            if att.critic is not None:
                if not att.critic.ran:
                    out.append(f"  CRITIC did not run: {att.critic.skipped_reason}")
                elif att.critic.findings:
                    verdict = "rejected it" if not att.critic.passed else "objected (non-blocking)"
                    out.append(f"  CRITIC {verdict}:")
                    out += [
                        f"    {line}"
                        for f in att.critic.findings
                        for line in f.render().splitlines()
                    ]
                else:
                    out.append("  CRITIC passed")
            elif not att.error and not errors:
                out.append("  (no checks recorded)")
            if raw and att.call and att.call.raw_text:
                out.append("  raw response (truncated):")
                out.append("    " + att.call.raw_text[:1200].replace("\n", "\n    "))
            out.append("")

        out.append("--- delivered " + "-" * 54)
        out.append(f"  generator={self.generator}  "
                   f"validator_passed={self.validator_passed}  "
                   f"critic_passed={self.critic_passed}")
        out.append(f"  facts used: {len(self.audit.get('facts_used') or [])} "
                   f"of {self.audit.get('fact_count')}  "
                   f"(coverage {self.audit.get('fact_coverage')})")
        out.append(f"  citations: {self.audit.get('citations_used')}")
        if not self.explanation.evidence:
            out.append("  WARNING: the delivered explanation has no evidence items")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class CriticProtocol(Protocol):
    def review(
        self,
        sheet: FactSheet,
        explanation: Explanation,
        pack: KnowledgePack | None = None,
    ) -> CriticVerdict: ...


@dataclass
class WorkflowConfig:
    #: "retry up to 2 times" -> 3 model attempts in total.
    max_retries: int = 2
    #: Consecutive attempts without lexicographic progress before giving up.
    patience: int = 2
    use_critic: bool = True
    allow_template_fallback: bool = True
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    #: Send the images again on retries. Off by default: the fact sheet and the
    #: feedback carry what the retry needs, and re-sending ~6 PNGs per attempt
    #: is the single biggest cost in the loop.
    images_on_retry: bool = False


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #
class ExplainerWorkflow:
    def __init__(
        self,
        pack: KnowledgePack | None = None,
        config: WorkflowConfig | None = None,
        *,
        explainer: ClaudeExplainer | None = None,
        critic: CriticProtocol | None = None,
        builder: EvidenceBuilder | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.progress = progress
        self.pack = pack or KnowledgePack.empty()
        self.config = config or WorkflowConfig()
        self.builder = builder or EvidenceBuilder()
        self.explainer = explainer
        if critic is not None:
            self.critic: CriticProtocol = critic
        elif self.config.use_critic:
            self.critic = ClaudeCritic(config=self.config.critic)
        else:
            self.critic = NullCritic()

    # ------------------------------------------------------------------ #
    def run(
        self,
        result: BatteryResult,
        images: Iterable[ImageAsset] = (),
    ) -> ExplainOutcome:
        started = time.monotonic()
        sheet = self.builder.build(result)
        sheet.facts.update(self.pack.context_facts)
        if self.progress:
            self.progress(f"Prepared {len(sheet.facts)} facts and {len(self.pack)} selected sources")
        validator = Validator(sheet, self.pack, self.config.validator)
        images = list(images)
        attempts: list[AttemptRecord] = []

        draft, report, critic = self._try_model(sheet, validator, images, attempts)

        if draft is None:
            if not self.config.allow_template_fallback:
                raise GenerationFailed(attempts)
            draft = TemplateExplainer().build(sheet)
            report = Validator(
                sheet, self.pack, template_validator_config(self.config.validator)
            ).validate(draft)
            record = AttemptRecord(
                attempt=len(attempts) + 1,
                source="template",
                ok=report.passed,
                findings=list(report.findings),
                progress=report.progress_key(),
            )
            attempts.append(record)
            if not report.passed:
                # The template is generated from the facts by code, so this is a
                # bug in the template, not a model failure. Surface it loudly
                # rather than shipping an unchecked explanation.
                log.error(
                    "the template explanation failed validation for %s:\n%s",
                    result.battery_id,
                    report.feedback(),
                )
            critic = None

        draft.validator_passed = report.passed if report else False
        # The critic's own opinion, not whether it was allowed to block: a
        # non-blocking critic that objected is recorded as a failure here, and
        # the platform shows "critic flagged" even though the text shipped.
        draft.critic_passed = (
            critic.declared_passed if (critic is not None and critic.ran) else None
        )
        rendered = render(draft, sheet, strict=draft.generator == "llm")

        return ExplainOutcome(
            battery_id=result.battery_id,
            run_id=result.run_id,
            explanation=rendered,
            draft=draft,
            sheet=sheet,
            attempts=attempts,
            audit={**self._audit(result, sheet, draft, attempts, report, critic),
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "request_timeout_seconds": self.config.explainer.timeout_seconds,
                   "max_transport_retries": self.config.explainer.max_transport_retries},
        )

    # ------------------------------------------------------------------ #
    def _try_model(
        self,
        sheet: FactSheet,
        validator: Validator,
        images: list[ImageAsset],
        attempts: list[AttemptRecord],
    ) -> tuple[Explanation | None, ValidationReport | None, CriticVerdict | None]:
        """Run the explainer with feedback retries. Returns (draft, report, critic)."""
        explainer = self.explainer
        if explainer is None:
            try:
                explainer = ClaudeExplainer(config=self.config.explainer)
                _ = explainer.client            # fail fast if the SDK/key is absent
            except Exception as exc:  # noqa: BLE001
                attempts.append(AttemptRecord(
                    attempt=1, source="llm",
                    error=f"explainer unavailable ({type(exc).__name__}: {exc})",
                ))
                return None, None, None

        history: list[dict[str, Any]] = []
        feedback: str | None = None
        best_progress: tuple[int, int] | None = None
        stalled = 0
        seen_signatures: set[tuple] = set()

        for attempt in range(1, self.config.max_retries + 2):
            send_images = images if (attempt == 1 or self.config.images_on_retry) else []
            record = AttemptRecord(attempt=attempt, source="llm")
            try:
                with report_activity(self.progress, f"Generating AI review (attempt {attempt}/{self.config.max_retries + 1})"):
                    draft, call, sent = explainer.generate(
                        sheet, self.pack, send_images, feedback=feedback, history=history
                    )
                record.call = call
            except TruncatedResponse as exc:
                record.error = str(exc)
                attempts.append(record)
                log.warning("attempt %d was cut off at the output limit: %s", attempt, exc)
                # Asking for valid JSON here would be the wrong instruction: the
                # draft was well-formed, just too long to finish.
                feedback = (
                    "Your previous response was cut off before the JSON finished, "
                    "because it was too long for the output budget. Write a SHORTER "
                    "explanation: at most three evidence items, one or two sentences "
                    "per claim, a single-sentence headline, and the shortest quote "
                    "that still supports each claim (it must stay exact). Return "
                    "exactly one complete JSON object."
                )
                continue
            except MalformedToolCall as exc:
                record.error = str(exc)
                attempts.append(record)
                log.warning("attempt %d encoded its arguments wrongly: %s", attempt, exc)
                # Nothing it *said* was wrong, so do not invite a rewrite of the
                # content -- name the encoding mistake precisely instead.
                feedback = (
                    f"Your previous response was not valid JSON ({exc}).\n\n"
                    "Do not use XML or any tag syntax. Reply with a single JSON "
                    "object as plain text. \"evidence\" and \"caveats\" must be JSON "
                    "arrays written with square brackets, like "
                    '"caveats": ["first caveat", "second caveat"], and each evidence '
                    "item must be a JSON object with the keys kind, detector, claim, "
                    "facts, regions and citations. Keep the content of your previous "
                    "answer; only fix the encoding."
                )
                continue
            except ExplainerError as exc:
                record.error = str(exc)
                attempts.append(record)
                log.warning("attempt %d produced no usable object: %s", attempt, exc)
                feedback = (
                    "Your previous response could not be parsed as the required JSON "
                    f"object ({exc}). Return exactly one JSON object and nothing else."
                )
                continue
            except Exception as exc:  # noqa: BLE001 - API/transport failure
                record.error = f"{type(exc).__name__}: {exc}"
                attempts.append(record)
                log.warning("the explainer call failed on attempt %d: %s", attempt, exc)
                break

            if self.progress:
                self.progress("Validating facts, numbers and citations")
            report = validator.validate(draft)
            record.findings = list(report.findings)
            record.progress = report.progress_key()

            critic: CriticVerdict | None = None
            if report.passed:
                with report_activity(self.progress, "Checking the review with the grounding critic"):
                    critic = self.critic.review(sheet, draft, self.pack)
                record.critic = critic
                if critic.passed:
                    if self.progress:
                        self.progress("Grounding critic passed" if critic.ran else f"Critic skipped: {critic.skipped_reason}")
                    record.ok = True
                    attempts.append(record)
                    return draft, report, critic

            attempts.append(record)
            if self.progress:
                self.progress(f"Attempt {attempt} rejected by grounding checks; "
                              + ("preparing a repair" if attempt <= self.config.max_retries else "attempt limit reached"))

            # -- loop guards ------------------------------------------- #
            signature = report.signature() + tuple(
                f.key() for f in (critic.findings if critic else [])
            )
            if signature in seen_signatures:
                log.info("attempt %d repeated an earlier failure set; stopping", attempt)
                break
            seen_signatures.add(signature)

            progress = report.progress_key()
            if best_progress is not None and progress >= best_progress:
                stalled += 1
                if stalled >= self.config.patience:
                    log.info("no progress after %d attempts; stopping", stalled)
                    break
            else:
                stalled = 0
            if best_progress is None or progress < best_progress:
                best_progress = progress

            if attempt > self.config.max_retries:
                break

            # Carry the whole conversation forward and replay the rejected
            # draft, so the next attempt is a diff and the message list still
            # begins with a user turn.
            history = sent + [assistant_echo(draft)]
            feedback = "\n\n".join(
                part for part in (report.feedback(), critic.feedback() if critic else "")
                if part
            )

        return None, None, None

    # ------------------------------------------------------------------ #
    def _audit(
        self,
        result: BatteryResult,
        sheet: FactSheet,
        draft: Explanation,
        attempts: list[AttemptRecord],
        report: ValidationReport | None,
        critic: CriticVerdict | None,
    ) -> dict[str, Any]:
        # Two different questions, and conflating them undercounted coverage by
        # about half: which facts the evidence items *declare*, and which facts
        # the prose actually *references*. The headline, why_not_runner_up,
        # agreement and caveats carry placeholders but have no facts list, so
        # counting only declarations missed every number in them.
        declared = {fid for item in draft.evidence for fid in item.facts}
        referenced = {
            name
            for _, text in draft.text_fields()
            for name in (m.group(1).strip() for m in PLACEHOLDER.finditer(text))
            if name in sheet
        }
        facts_used = sorted(declared | referenced)
        # A fact listed but never referenced is a way to look grounded without
        # being grounded, so it is reported rather than silently folded in.
        declared_unused = sorted(declared - referenced)
        citations_used = sorted(
            {c.source for item in draft.evidence for c in item.citations}
        )
        return {
            "written_at": utc_now(),
            "git": result.versions.git,
            "scope": result.scope,
            "knowledge_pack": self.pack.version,
            "knowledge_pack_sources": self.pack.ids(),
            "prompt": prompt_fingerprint(self.pack),
            # Read from the live objects, not the config: an injected explainer
            # or critic carries its own model, and the audit must name what
            # actually ran.
            "explainer_model": getattr(
                getattr(self.explainer, "config", None), "model",
                self.config.explainer.model,
            ),
            "critic_model": getattr(
                getattr(self.critic, "config", None), "model", None
            ),
            "critic_blocking": getattr(
                getattr(self.critic, "config", None), "blocking", None
            ),
            "critic_declared_passed": critic.declared_passed if critic else None,
            "critic_ran": bool(critic and critic.ran),
            "critic_skipped_reason": critic.skipped_reason if critic else None,
            "attempts": len(attempts),
            "attempt_log": [a.summary() for a in attempts],
            "validator_errors": [f.code for f in (report.errors if report else [])],
            "validator_warnings": [f.code for f in (report.warnings if report else [])],
            "critic_findings": [f.code for f in (critic.findings if critic else [])],
            "fact_count": len(sheet.facts),
            "facts_used": facts_used,
            "facts_declared_unused": declared_unused,
            "fact_coverage": (
                round(len(facts_used) / len(sheet.facts), 3) if sheet.facts else None
            ),
            "citations_used": citations_used,
        }


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def explain_battery(
    result: BatteryResult,
    pack: KnowledgePack | None = None,
    images: Iterable[ImageAsset] = (),
    *,
    config: WorkflowConfig | None = None,
    explainer: ClaudeExplainer | None = None,
    critic: CriticProtocol | None = None,
    progress: Callable[[str], None] | None = None,
) -> ExplainOutcome:
    """Explain one battery. The entry point Modal and the CLI both call."""
    workflow = ExplainerWorkflow(
        pack, config, explainer=explainer, critic=critic, progress=progress
    )
    return workflow.run(result, images)
