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
    return BatteryResult.model_validate_json(
        (ROOT / "examples" / "BAT-07.json").read_text(encoding="utf-8")
    )


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
