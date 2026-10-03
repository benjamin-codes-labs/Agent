"""Modal deployment of Step 6 (proposal section 14).

    modal deploy sentinel/modal_app.py
    modal run sentinel/modal_app.py::explain_one --battery-id BAT-07 --run-id R1

``explain_battery`` is one CPU function per battery, spawned after
classification; it calls the Claude API and writes
``explanations/{battery}.json``.

API names are the post-1.0 ones (``min_containers``, ``Image.add_local_*``,
``@modal.fastapi_endpoint``): several were renamed in Modal 1.0 and the old
spellings are now errors.

Volume discipline from the proposal: a write is always a new file, so concurrent
last-write-wins cannot corrupt a result and the files form the audit trail.
Call ``volume.commit()`` after writing and ``volume.reload()`` before reading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

APP_NAME = "sentinel-explainer"
VOLUME_NAME = "sentinel-data"
DATA_ROOT = Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "anthropic~=0.69",
        "pydantic~=2.9",
        "jinja2~=3.1",
        "pillow~=11.0",
    )
    # Explicit in 1.0: implicit automounting is gone, and its absence is the
    # usual cause of "works locally, ModuleNotFoundError on Modal".
    .add_local_python_source("sentinel")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
anthropic_secret = modal.Secret.from_name("")


# --------------------------------------------------------------------------- #
# Paths on the Volume
# --------------------------------------------------------------------------- #
def battery_path(run_id: str, battery_id: str) -> Path:
    return DATA_ROOT / "runs" / run_id / "batteries" / f"{battery_id}.json"


def explanation_path(run_id: str, battery_id: str) -> Path:
    return DATA_ROOT / "runs" / run_id / "explanations" / f"{battery_id}.json"


def knowledge_path() -> Path:
    return DATA_ROOT / "knowledge"


# --------------------------------------------------------------------------- #
# The explainer function
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={str(DATA_ROOT): volume},
    secrets=[anthropic_secret],
    timeout=600,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
    # Warm one container so the on-stage run does not pay a cold start.
    min_containers=int(os.environ.get("SENTINEL_WARM", "0") or 0),
)
def explain_battery(run_id: str, battery_id: str, *, use_critic: bool = True) -> dict[str, Any]:
    """Explain one battery and write ``explanations/{battery}.json``."""
    from sentinel import BatteryResult, KnowledgePack, WorkflowConfig, collect_images
    from sentinel.critic import CriticConfig
    from sentinel.workflow import ExplainerWorkflow

    volume.reload()

    source = battery_path(run_id, battery_id)
    if not source.is_file():
        raise FileNotFoundError(f"no result document at {source}")
    result = BatteryResult.model_validate_json(source.read_text(encoding="utf-8"))

    pack_dir = knowledge_path()
    pack = KnowledgePack.from_path(pack_dir) if pack_dir.exists() else KnowledgePack.empty()

    images = collect_images(result, root=DATA_ROOT)

    workflow = ExplainerWorkflow(
        pack,
        WorkflowConfig(use_critic=use_critic, critic=CriticConfig(blocking=True)),
    )
    outcome = workflow.run(result, images)

    # Write the explanation document, then the updated battery document. Both
    # are new files; nothing is modified in place.
    target = explanation_path(run_id, battery_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(outcome.to_json(), encoding="utf-8")

    merged = result.with_explanation(outcome.explanation)
    merged_path = source.with_name(f"{battery_id}.explained.json")
    merged_path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")

    volume.commit()

    return {
        "battery_id": battery_id,
        "run_id": run_id,
        "generator": outcome.generator,
        "validator_passed": outcome.validator_passed,
        "critic_passed": outcome.critic_passed,
        "attempts": len(outcome.attempts),
        "explanation_path": str(target.relative_to(DATA_ROOT)),
        "headline": outcome.explanation.headline,
    }


@app.function(image=image, volumes={str(DATA_ROOT): volume}, timeout=900)
def explain_run(run_id: str, *, use_critic: bool = True) -> list[dict[str, Any]]:
    """Fan out over every battery in a run.

    ``return_exceptions=True`` matters on the day: one battery whose Claude call
    fails must not take the whole batch down with it.
    """
    volume.reload()
    folder = DATA_ROOT / "runs" / run_id / "batteries"
    battery_ids = sorted(p.stem for p in folder.glob("*.json") if ".explained" not in p.name)
    if not battery_ids:
        return []

    results: list[dict[str, Any]] = []
    for battery_id, outcome in zip(
        battery_ids,
        explain_battery.map(
            [run_id] * len(battery_ids),
            battery_ids,
            kwargs={"use_critic": use_critic},
            return_exceptions=True,
        ),
    ):
        if isinstance(outcome, Exception):
            results.append({
                "battery_id": battery_id,
                "run_id": run_id,
                "error": f"{type(outcome).__name__}: {outcome}",
            })
        else:
            results.append(outcome)
    return results


# --------------------------------------------------------------------------- #
# Validator-only endpoint: the deterministic half, no API key needed.
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={str(DATA_ROOT): volume}, timeout=300)
def revalidate(run_id: str, battery_id: str) -> dict[str, Any]:
    """Re-run the code validator over a stored explanation.

    Useful twice: as a cheap regression check after a rule change, and as the
    "what does the checker actually catch" demo, since it needs no model call.
    """
    from sentinel import BatteryResult, Explanation, KnowledgePack, Validator, build_fact_sheet

    volume.reload()
    result = BatteryResult.model_validate_json(
        battery_path(run_id, battery_id).read_text(encoding="utf-8")
    )
    stored = json.loads(explanation_path(run_id, battery_id).read_text(encoding="utf-8"))
    draft = Explanation(**stored["draft"])

    pack_dir = knowledge_path()
    pack = KnowledgePack.from_path(pack_dir) if pack_dir.exists() else KnowledgePack.empty()
    report = Validator(build_fact_sheet(result), pack).validate(draft)
    return {
        "battery_id": battery_id,
        "passed": report.passed,
        "errors": [f.render() for f in report.errors],
        "warnings": [f.render() for f in report.warnings],
    }


# --------------------------------------------------------------------------- #
# Local entrypoints
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def explain_one(battery_id: str = "BAT-07", run_id: str = "R1", critic: bool = True):
    out = explain_battery.remote(run_id, battery_id, use_critic=critic)
    print(json.dumps(out, indent=2))


@app.local_entrypoint()
def explain_all(run_id: str = "R1", critic: bool = True):
    rows = explain_run.remote(run_id, use_critic=critic)
    ok = sum(1 for r in rows if not r.get("error"))
    llm = sum(1 for r in rows if r.get("generator") == "llm")
    print(json.dumps(rows, indent=2))
    print(f"\n{ok}/{len(rows)} explained, {llm} by the model, {ok - llm} by template")
