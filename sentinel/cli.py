"""Command line for Step 6.

    python -m sentinel.cli explain examples/BAT-07.json --pack knowledge/pack.md
    python -m sentinel.cli explain examples/BAT-07.json --template-only
    python -m sentinel.cli validate examples/BAT-07.json --explanation draft.json
    python -m sentinel.cli facts examples/BAT-07.json
    python -m sentinel.cli selftest examples/BAT-07.json

``--template-only`` is the offline path: no API key, no network, deterministic
output. It is what the demo falls back to and what the platform's offline mode
reads.

``selftest`` runs the seeded-defect suite: it injects known faults into a valid
explanation and reports whether the checker catches each one. That doubles as
the "what does your validator actually do" answer on stage, and as a
precision/recall table if the team extends the defect list.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .contract import BatteryResult, Explanation
from .facts import build_fact_sheet
from .knowledge import KnowledgePack
from .prompts import collect_images
from .renderer import render
from .selftest import DEFECTS, run_selftest
from .template import TemplateExplainer, template_validator_config
from .validator import Validator
from .workflow import ExplainerWorkflow, WorkflowConfig


def _load(path: str) -> BatteryResult:
    return BatteryResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _pack(path: str | None) -> KnowledgePack:
    if not path:
        return KnowledgePack.empty()
    return KnowledgePack.from_path(path)


def cmd_explain(args: argparse.Namespace) -> int:
    result = _load(args.result)
    pack = _pack(args.pack)

    if args.template_only:
        sheet = build_fact_sheet(result)
        draft = TemplateExplainer().build(sheet)
        report = Validator(sheet, pack, template_validator_config()).validate(draft)
        draft.validator_passed = report.passed
        rendered = render(draft, sheet, strict=False)
        print(json.dumps(rendered.model_dump(), indent=2, ensure_ascii=False))
        if not report.passed:
            print(report.feedback(), file=sys.stderr)
            return 1
        return 0

    images = collect_images(result, root=args.image_root) if args.images else []
    workflow = ExplainerWorkflow(
        pack, WorkflowConfig(use_critic=not args.no_critic, max_retries=args.max_retries)
    )
    outcome = workflow.run(result, images)

    print(outcome.debug(raw=args.raw) if args.debug else outcome.report(),
          file=sys.stderr)
    if args.out:
        Path(args.out).write_text(outcome.to_json(), encoding="utf-8")
        print(f"written to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(outcome.explanation.model_dump(), indent=2, ensure_ascii=False))
    return 0 if outcome.validator_passed else 1


def cmd_validate(args: argparse.Namespace) -> int:
    result = _load(args.result)
    sheet = build_fact_sheet(result)
    payload = json.loads(Path(args.explanation).read_text(encoding="utf-8"))
    payload = payload.get("draft", payload)
    draft = Explanation(**payload)
    report = Validator(sheet, _pack(args.pack)).validate(draft)
    if report.passed and not report.warnings:
        print("validator: pass")
        return 0
    print(report.feedback())
    return 0 if report.passed else 1


def cmd_facts(args: argparse.Namespace) -> int:
    sheet = build_fact_sheet(_load(args.result))
    if args.json:
        print(sheet.model_dump_json(indent=2))
    else:
        print(sheet.to_prompt_block())
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    rows, caught = run_selftest(_load(args.result), _pack(args.pack))
    width = max(len(name) for name, _, _, _ in rows)
    for name, expected_code, got, ok in rows:
        mark = "caught " if ok else "MISSED "
        print(f"{mark} {name:<{width}}  expect {expected_code:<12} got {got or '-'}")
    print(f"\n{caught}/{len(rows)} seeded defects caught")
    return 0 if caught == len(rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("explain", help="run the full workflow on one battery")
    p.add_argument("result", help="path to a battery result JSON document")
    p.add_argument("--pack", help="knowledge pack file or directory")
    p.add_argument("--out", help="write the full outcome here")
    p.add_argument("--images", action="store_true", help="send the Step 5 PNGs")
    p.add_argument("--image-root", default=".", help="root for the PNG paths")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--template-only", action="store_true", help="offline, no API call")
    p.add_argument("--debug", action="store_true",
                   help="print every finding from every attempt, not just a summary")
    p.add_argument("--raw", action="store_true",
                   help="with --debug, also print the model's unparsed response")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("validate", help="validate a stored explanation")
    p.add_argument("result")
    p.add_argument("--explanation", required=True)
    p.add_argument("--pack")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("facts", help="print the fact sheet the model would see")
    p.add_argument("result")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_facts)

    p = sub.add_parser("selftest", help="inject known defects and score the checker")
    p.add_argument("result")
    p.add_argument("--pack")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
