"""The explainer call (proposal section 6.3).

Model: ``claude-sonnet-5-5``, one call with vision.

Two platform constraints noted in the proposal are respected here and are worth
not forgetting, because violating them is an API error rather than a quiet
degradation: for this model we do **not** force ``tool_choice`` and we do
**not** set ``temperature``. The tool is offered and the model may use it; if it
answers with text instead, the JSON is extracted from the text. Both paths are
handled so a refusal to call the tool never costs a retry.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, Callable, Iterable, Protocol

from pydantic import ValidationError

from .contract import Explanation
from .env import api_key_hint, ensure_api_key
from .facts import FactSheet
from .knowledge import KnowledgePack
from .prompts import (
    EXPLAINER_TOOL,
    ImageAsset,
    build_system_prompt,
    build_user_message,
)

FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@contextmanager
def report_activity(progress: Callable[[str], None] | None, message: str, *, interval: float = 10.0):
    if progress is None:
        yield
        return
    progress(message)
    stop = Event()

    def heartbeat():
        while not stop.wait(interval):
            progress(message + " — still waiting for the provider")

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


class ExplainerError(RuntimeError):
    """The model produced something that is not a usable explanation object."""


class MalformedToolCall(ExplainerError):
    """A tool call arrived with its arguments encoded as pseudo-XML, not JSON.

    Seen live in two different spellings: ``<parameter name="item">...`` and
    ``<item><kind>both</kind>...</caveats>``, both landing in ``evidence`` as a
    single string. Enumerating tag names is a losing game, so detection is
    structural instead: a field the contract declares as a list arriving as a
    string means the arguments were never JSON-encoded.

    Retryable, and worth distinguishing from a content error because the fix is
    purely about encoding -- nothing the model *said* was wrong.
    """


class TruncatedResponse(ExplainerError):
    """The response was cut off at max_tokens, so the JSON is incomplete.

    Kept separate from ExplainerError because the repair advice differs: the
    model did nothing wrong about *format*, it ran out of room. Telling it to
    "return valid JSON" wastes a retry; the fix is a shorter explanation or a
    bigger budget.
    """


class MessagesClient(Protocol):
    """The slice of the Anthropic SDK this module uses."""

    @property
    def messages(self) -> Any: ...


@dataclass
class ModelCall:
    """One request/response pair, kept for the audit trail."""

    model: str
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    used_tool: bool = False
    raw_text: str | None = None

    def cost_note(self) -> str:
        bits = [self.model]
        if self.input_tokens is not None:
            bits.append(f"in={self.input_tokens}")
        if self.cache_read_tokens:
            bits.append(f"cached={self.cache_read_tokens}")
        if self.output_tokens is not None:
            bits.append(f"out={self.output_tokens}")
        return " ".join(bits)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull one JSON object out of model text, fenced or bare."""
    candidates: list[str] = []
    fenced = FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text.strip())
    # Last resort: the outermost brace-balanced span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ExplainerError(
        "no JSON object found in the model response; first 300 characters: "
        + text[:300].replace("\n", " ")
    )


def parse_explanation(payload: dict[str, Any]) -> Explanation:
    """Validate the model's object against the contract, tolerating extras."""
    # Tool-call plumbing sometimes lands inside the structured payload itself,
    # e.g. ``"evidence": "\\n<invoke name=\\"x\\">\\n</invoke>\\n"``. Pydantic
    # then reports a confusing type error about a list, so catch it here and
    # call it what it is: a malformed call that needs re-emitting, not a
    # content problem the model should reason about.
    for field in ("evidence", "caveats"):
        if isinstance(payload.get(field), str):
            raise MalformedToolCall(
                f"the {field!r} field arrived as a string rather than a list, so the "
                f"tool arguments were not JSON-encoded. First 120 characters: "
                f"{payload[field][:120]!r}"
            )
    if _looks_like_leaked_tool_syntax(json.dumps(payload, default=str)):
        raise MalformedToolCall(
            "the submitted object contains raw tool-call syntax in place of a "
            "field value, which means the call was emitted malformed"
        )

    allowed = set(Explanation.model_fields) - {
        "generator", "validator_passed", "critic_passed"
    }
    cleaned = {k: v for k, v in payload.items() if k in allowed}
    for key in ("headline", "why_not_runner_up", "agreement"):
        value = cleaned.get(key)
        if isinstance(value, list) and value and all(isinstance(part, str) for part in value):
            cleaned[key] = " ".join(part.strip() for part in value)
    for key in ("evidence", "caveats"):
        if cleaned.get(key) is None:
            cleaned[key] = []
    for item in cleaned.get("evidence") or []:
        if isinstance(item, dict):
            item.setdefault("facts", [])
            item.setdefault("regions", [])
            item.setdefault("citations", [])
            if item.get("detector") in {"", "null", "none"}:
                item["detector"] = None
    try:
        return Explanation(**cleaned, generator="llm")
    except ValidationError as exc:
        raise ExplainerError(f"the explanation object does not fit the contract: {exc}") from exc


@dataclass
class ExplainerConfig:
    model: str = "claude-sonnet-5-5"
    #: Five evidence items carrying verbatim citations runs well past 3000
    #: tokens, and a response cut mid-JSON costs a whole retry for nothing.
    #: Output tokens are billed only as used, so the budget is generous.
    max_tokens: int = 8000
    timeout_seconds: float = 90.0
    max_transport_retries: int = 0
    #: Off by default, on the evidence of live runs. Because ``tool_choice``
    #: cannot be forced on this model, offering the tool is only a suggestion --
    #: and when the model took it, its arguments twice came back as pseudo-XML
    #: with ``evidence`` as a string, costing a retry each time. The
    #: JSON-in-text path has parsed first time on every live attempt, the
    #: output schema is already spelled out in the prompt, and the contract is
    #: enforced by pydantic either way, so the tool buys nothing here and
    #: introduces a failure mode. Turn it back on to experiment.
    offer_tool: bool = False
    #: Left False deliberately: this model rejects forced tool use.
    force_tool: bool = False
    #: Left None deliberately: this model rejects non-default sampling params.
    temperature: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("request timeout must be positive and finite")
        if self.max_transport_retries < 0:
            raise ValueError("transport retries must be non-negative")


class ClaudeExplainer:
    """Wraps one Sonnet call. Inject a fake client in tests."""

    def __init__(
        self,
        client: Any | None = None,
        config: ExplainerConfig | None = None,
    ):
        self.config = config or ExplainerConfig()
        self._client = client
        self.calls: list[ModelCall] = []

    # -- client ----------------------------------------------------------- #
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ExplainerError(
                    "the anthropic SDK is not installed; run the template path instead"
                ) from exc
            # Pick the key up from a .env beside the project, so nobody has to
            # export it and nobody is tempted to paste it into source.
            if not ensure_api_key():
                raise ExplainerError(api_key_hint())
            self._client = anthropic.Anthropic(timeout=self.config.timeout_seconds,
                                               max_retries=self.config.max_transport_retries)
        return self._client

    # -- generation ------------------------------------------------------- #
    def generate(
        self,
        sheet: FactSheet,
        pack: KnowledgePack,
        images: Iterable[ImageAsset] = (),
        *,
        feedback: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[Explanation, ModelCall, list[dict[str, Any]]]:
        """Returns (explanation, call, the full message list that was sent).

        The caller keeps the returned list and appends the assistant turn to it,
        so the conversation stays well-formed: it must begin with a user turn,
        and an assistant-first message list is an API error. Keeping the first
        turn also keeps the images in context on retries without re-uploading
        them.
        """
        messages = list(history or [])
        messages.append(build_user_message(sheet, images, feedback=feedback, include_facts=not bool(history)))

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
            "system": build_system_prompt(pack),
            "messages": messages,
        }
        if self.config.offer_tool:
            kwargs["tools"] = [EXPLAINER_TOOL]
            if self.config.force_tool:
                kwargs["tool_choice"] = {"type": "tool", "name": EXPLAINER_TOOL["name"]}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        response = self.client.messages.create(**kwargs)
        call, payload = self._read_response(response)
        self.calls.append(call)
        return parse_explanation(payload), call, messages

    def _read_response(self, response: Any) -> tuple[ModelCall, dict[str, Any]]:
        usage = getattr(response, "usage", None)
        call = ModelCall(
            model=getattr(response, "model", self.config.model),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None),
        )
        texts: list[str] = []
        for block in getattr(response, "content", []) or []:
            kind = getattr(block, "type", None)
            if kind == "tool_use" and getattr(block, "name", "") == EXPLAINER_TOOL["name"]:
                call.used_tool = True
                payload = getattr(block, "input", None)
                if isinstance(payload, dict):
                    return call, payload
            elif kind == "text":
                texts.append(getattr(block, "text", "") or "")
        joined = "\n".join(texts).strip()
        call.raw_text = joined[:4000] or None

        # Truncation first: it is the most common failure and it masquerades as
        # a parse error. A response cut at max_tokens leaves JSON broken
        # mid-token, and a cut tool call can leak its raw parameter syntax into
        # the text, so "no JSON object found" would send the model chasing a
        # formatting problem it does not have.
        if call.stop_reason == "max_tokens":
            raise TruncatedResponse(
                f"the response hit the {self.config.max_tokens}-token output limit and "
                f"was cut off mid-JSON (stop_reason='max_tokens'). Either raise "
                f"ExplainerConfig.max_tokens or ask for a shorter explanation."
            )
        if not joined:
            raise ExplainerError(
                f"the model returned no usable content (stop_reason={call.stop_reason!r})"
            )
        if _looks_like_leaked_tool_syntax(joined):
            raise TruncatedResponse(
                "the response contains raw tool-call syntax rather than JSON, which "
                "happens when a tool call is cut short. Treating it as truncation."
            )
        return call, extract_json_object(joined)


#: Fragments of tool-call plumbing that only appear when a call is cut short.
_TOOL_SYNTAX_MARKERS = (
    "<parameter name=",
    "<invoke name=",
    "<function_calls>",
    "<parameter",
)


def _looks_like_leaked_tool_syntax(text: str) -> bool:
    return any(marker in text for marker in _TOOL_SYNTAX_MARKERS)


def assistant_echo(explanation: Explanation) -> dict[str, Any]:
    """The assistant turn to replay when sending validator feedback.

    Keeping the rejected attempt in the conversation is what lets the model
    produce a *diff* rather than a fresh draft, which is what makes the loop
    converge.
    """
    payload = explanation.model_dump(
        exclude={"generator", "validator_passed", "critic_passed"}
    )
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
    }
