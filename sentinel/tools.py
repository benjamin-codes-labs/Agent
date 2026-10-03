"""v2: a genuine agent loop with read-only tools (proposal section 6.6).

``zoom_region``, ``get_stat_distribution`` and ``search_knowledge``, with a step
limit. No tool can change a prediction -- enforced structurally, since the tool
registry has no write path and every handler takes the fact sheet as read-only
input.

This is the one place where the fixed workflow becomes a real agent loop, and
the v1 path does not import it. Keep it that way: if the loop misbehaves on the
day, the fallback is simply not passing ``tools=`` to the explainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .contract import BatteryResult
from .facts import FactSheet
from .knowledge import KnowledgePack

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

ZOOM_REGION: dict[str, Any] = {
    "name": "zoom_region",
    "description": (
        "Return the measured facts for one numbered evidence region, and the path "
        "to its untinted crop. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "detector": {"type": "string", "enum": ["BSE", "ETD", "InLens"]},
            "id": {"type": "integer", "description": "the region number, e.g. 1"},
        },
        "required": ["detector", "id"],
    },
}

GET_STAT_DISTRIBUTION: dict[str, Any] = {
    "name": "get_stat_distribution",
    "description": (
        "Return each type's profile for one statistic together with this battery's "
        "value and z-scores, so you can see where this battery sits. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "stat": {
                "type": "string",
                "description": "a statistic base ID, e.g. stats.BSE.porosity",
            }
        },
        "required": ["stat"],
    },
}

SEARCH_KNOWLEDGE: dict[str, Any] = {
    "name": "search_knowledge",
    "description": (
        "Search the knowledge pack and return verbatim snippets with their source "
        "IDs. Quote only what this returns. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
    },
}


@dataclass
class ReadOnlyTools:
    """Tool handlers bound to one battery. Nothing here can mutate state."""

    sheet: FactSheet
    result: BatteryResult
    pack: KnowledgePack
    asset_root: Path = Path(".")
    max_calls: int = 6
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    # -- registry --------------------------------------------------------- #
    def definitions(self) -> list[dict[str, Any]]:
        return [ZOOM_REGION, GET_STAT_DISTRIBUTION, SEARCH_KNOWLEDGE]

    def handlers(self) -> dict[str, ToolHandler]:
        return {
            "zoom_region": self.zoom_region,
            "get_stat_distribution": self.get_stat_distribution,
            "search_knowledge": self.search_knowledge,
        }

    def dispatch(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if len(self.calls) >= self.max_calls:
            return {
                "error": f"tool budget of {self.max_calls} calls is spent; "
                         "write the explanation with what you have"
            }
        handler = self.handlers().get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        self.calls.append((name, payload))
        try:
            return handler(payload)
        except Exception as exc:  # noqa: BLE001 - a tool error must not kill the run
            return {"error": f"{type(exc).__name__}: {exc}"}

    @property
    def budget_left(self) -> int:
        return max(0, self.max_calls - len(self.calls))

    # -- handlers --------------------------------------------------------- #
    def zoom_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        detector = payload.get("detector")
        rid = payload.get("id")
        name = f"{detector} region {rid}"
        found = self.result.region(name)
        if found is None:
            return {
                "error": f"{name} does not exist",
                "available": self.sheet.region_ids,
            }
        _, region = found
        prefix = f"region.{detector}.{rid}"
        facts = {
            fact_id: self.sheet.facts[fact_id].display
            for fact_id in self.sheet.ids()
            if fact_id.startswith(prefix)
        }
        maps = self.result.maps.get(detector)  # type: ignore[arg-type]
        return {
            "region": name,
            "box_um": region.box_um,
            "facts": facts,
            "crop_png": (str(self.asset_root / maps.crops_png) if maps and maps.crops_png else None),
            "note": "quote these facts by their IDs; do not retype the numbers",
        }

    def get_stat_distribution(self, payload: dict[str, Any]) -> dict[str, Any]:
        stat_id = str(payload.get("stat", "")).strip()
        record = self.result.statistic(stat_id)
        if record is None:
            candidates = sorted(
                {
                    fid[: -len(".value")]
                    for fid in self.sheet.ids()
                    if fid.startswith("stats.") and fid.endswith(".value")
                }
            )
            return {"error": f"{stat_id!r} is not a statistic", "available": candidates}
        facts = {
            fact_id: self.sheet.facts[fact_id].display
            for fact_id in self.sheet.ids()
            if fact_id.startswith(stat_id + ".")
        }
        return {
            "stat": stat_id,
            "detector": record.detector,
            "unit": record.unit,
            "facts": facts,
            "separating_rank": record.separates_pred_vs_runner_up_rank,
            "note": "an SD computed from about 30 batteries is itself uncertain "
                    "by roughly 13 per cent",
        }

    def search_knowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query", ""))
        limit = int(payload.get("limit") or 3)
        hits = self.pack.search(query, limit=max(1, min(limit, 5)))
        return {
            "query": query,
            "hits": hits,
            "note": "quote a snippet character for character, or do not cite it",
        }


def run_tool_loop(
    client: Any,
    *,
    model: str,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: ReadOnlyTools,
    extra_tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 8000,
    max_steps: int = 8,
) -> Any:
    """Drive the read-only tool loop until the model stops asking for tools.

    Returns the final response. The step limit is hard: the loop exits even if
    the model would keep calling tools, which is what keeps a v2 demo bounded.
    """
    definitions = tools.definitions() + list(extra_tools or [])
    response = None
    for _ in range(max_steps):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=definitions,
            messages=messages,
        )
        tool_uses = [
            block for block in (getattr(response, "content", []) or [])
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", "") in tools.handlers()
        ]
        if not tool_uses:
            return response
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            payload = getattr(block, "input", {}) or {}
            output = tools.dispatch(block.name, payload)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(output, ensure_ascii=False),
            })
        messages.append({"role": "user", "content": results})
    return response
