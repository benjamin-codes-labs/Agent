"""Placeholder renderer (proposal section 6.4, last line).

The explainer writes ``{stats.BSE.porosity.value}``; the renderer substitutes
the fact's ``display`` string. Because ``display`` is computed once in the
evidence builder, the number on screen is byte-identical to the number in the
JSON contract -- there is no second rounding step that could make the audit
trail disagree with the text.

Rendering happens *after* validation. An unresolved placeholder at this stage is
a programming error, not a model error, so it raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import Citation, EvidenceItem, Explanation
from .facts import FactSheet
from .textrules import PLACEHOLDER


class UnresolvedPlaceholder(KeyError):
    """A placeholder survived validation but has no fact. Should never happen."""


@dataclass
class Renderer:
    sheet: FactSheet
    strict: bool = True

    def _lookup(self, name: str) -> str:
        if name == "pred_type":
            return self.sheet.predicted
        if name == "runner_up":
            return self.sheet.runner_up
        if name == "confidence":
            return self.sheet.confidence
        fact = self.sheet.get(name)
        if fact is not None:
            return fact.display
        if self.strict:
            raise UnresolvedPlaceholder(
                f"{{{name}}} has no fact on the sheet; validation should have "
                f"caught this (closest: {', '.join(self.sheet.nearest_ids(name)) or 'none'})"
            )
        return f"[{name}?]"

    def text(self, value: str) -> str:
        return PLACEHOLDER.sub(lambda m: self._lookup(m.group(1).strip()), value)

    def explanation(self, explanation: Explanation) -> Explanation:
        return Explanation(
            headline=self.text(explanation.headline),
            why_not_runner_up=self.text(explanation.why_not_runner_up),
            evidence=[
                EvidenceItem(
                    kind=item.kind,
                    detector=item.detector,
                    claim=self.text(item.claim),
                    facts=list(item.facts),
                    regions=list(item.regions),
                    citations=[
                        Citation(source=c.source, quote=c.quote) for c in item.citations
                    ],
                )
                for item in explanation.evidence
            ],
            agreement=self.text(explanation.agreement),
            caveats=[self.text(c) for c in explanation.caveats],
            generator=explanation.generator,
            validator_passed=explanation.validator_passed,
            critic_passed=explanation.critic_passed,
        )


def render(explanation: Explanation, sheet: FactSheet, *, strict: bool = True) -> Explanation:
    return Renderer(sheet, strict=strict).explanation(explanation)
