import argparse
import json
import re
import sys
import time
from pathlib import Path

from sentinel import BatteryResult, Explanation, KnowledgePack, ValidatorConfig, WorkflowConfig, explain_battery, collect_images
from sentinel.env import load_dotenv
from sentinel.explainer import ExplainerConfig, ExplainerError
from sentinel.critic import CriticConfig
from sentinel.prompts import ImageAsset
from sentinel.questions import GroundedAnswer, answer_question

ROOT = Path(__file__).resolve().parent


def format_points(points) -> str:
    bullets = []
    for point in points:
        if not point.strip():
            continue
        lines = point.strip().splitlines()
        first = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", lines[0])
        bullets.append("- " + first + "".join("\n  " + line.strip() for line in lines[1:]))
    return "\n\n".join(bullets)


def format_review(explanation_json: str) -> str:
    explanation = Explanation.model_validate_json(explanation_json)
    summary = " ".join(text.strip() for text in (explanation.headline, explanation.agreement) if text.strip())
    return format_points([
        summary,
        *(item.claim for item in explanation.evidence),
        explanation.why_not_runner_up,
        *explanation.caveats,
    ])


def format_answer(answer_json: str) -> str:
    answer = GroundedAnswer.model_validate_json(answer_json)
    return format_points(paragraph.text for paragraph in answer.paragraphs)


def question_images(specs: list[str]) -> list[ImageAsset]:
    detectors = {"bse": "BSE", "etd": "ETD", "inlens": "InLens"}
    images = []
    for spec in specs:
        detector, separator, name = spec.partition("=")
        if not separator or detector.lower() not in detectors or not name.strip():
            raise ValueError("--image must be BSE=path, ETD=path or Inlens=path")
        path = Path(name).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise ValueError(f"image does not exist: {path}")
        if path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"unsupported image format: {path.suffix}")
        images.append(ImageAsset(detector=detectors[detector.lower()], kind="raw", path=path))
    return images


def main(*, debug: bool = False, question: str | None = None, sample: str | Path | None = None,
         images: list[str] | None = None, quiet: bool = False, full_context: bool = False,
         timeout: float = 90.0, max_attempts: int = 2, max_tokens: int = 8000) -> None:
    if max_attempts < 1 or max_tokens < 1:
        raise ValueError("max-attempts and max-output-tokens must be positive")
    config = ExplainerConfig(timeout_seconds=timeout, max_tokens=max_tokens)
    started = time.monotonic()

    def report(message):
        print(f"[Sentinel +{time.monotonic() - started:.1f}s] {message}", file=sys.stderr, flush=True)

    progress = None if quiet else report
    if progress:
        progress("Loading knowledge and selecting relevant context")
    load_dotenv(start=ROOT)
    pack = KnowledgePack.from_project(ROOT)
    if question is not None or sample is not None or images:
        if sample is not None:
            path = Path(sample).expanduser()
            path = path if path.is_absolute() else ROOT / path
            pack = pack.with_sample(json.loads(path.read_text(encoding="utf-8")), origin=str(path))
        if question is None:
            question = ("Write a concise classification review, not a statistics inventory. In about five or six "
                        "short points, explain the strongest observed or measured characteristics that fit the "
                        "predicted category and why those characteristics fit the runner-up or other categories "
                        "less well. Use plain-language contrasts and very few numbers. Include material opposing "
                        "evidence and the main confidence limitation. Compare batches only when feature definitions "
                        "and imaging conditions are compatible; state when another category lacks a comparable "
                        "profile instead of inventing its characteristics. If no classifier output is supplied, "
                        "do not assign a battery type.")
        outcome = answer_question(question, pack, question_images(images or []), config=config,
                                  max_retries=max_attempts - 1, full_context=full_context, progress=progress,
                                  max_words=280, max_points=6)
        if progress:
            progress("Answer ready")
        print(format_answer(outcome.answer.model_dump_json()))
        if debug:
            print(json.dumps(outcome.audit, ensure_ascii=False, indent=2), file=sys.stderr)
        return
    result = BatteryResult.model_validate_json(
        (ROOT / "examples" / "BAT-07.json").read_text(encoding="utf-8")
    )
    query = "SEM classification review BSE ETD InLens " + " ".join(stat.label or stat.name for stat in result.statistics)
    query += " " + " ".join(result.quality_flagged()).replace("_", " ")
    pack = pack.select(query, limit=12 if full_context else 6, compact=not full_context,
                       batch_statistics=full_context)
    outcome = explain_battery(result, pack, collect_images(result, root=ROOT / "runs"),
                              config=WorkflowConfig(allow_template_fallback=False, max_retries=max_attempts - 1,
                                                    validator=ValidatorConfig(max_words=280, max_evidence_items=3, max_caveats=2),
                                                    explainer=config, critic=CriticConfig(timeout_seconds=timeout)),
                              progress=progress)
    if outcome.generator != "llm":
        raise ExplainerError("No AI-generated review was returned; refusing to display a fixed template.")
    if progress:
        progress("Review ready")

    print(format_review(outcome.explanation.model_dump_json()))      # rendered, real numbers — this goes on the battery page
    if debug:
        print(outcome.generator, file=sys.stderr)        # "llm" or "template" — show this in the UI
        print(outcome.critic_passed, file=sys.stderr)    # False means the critic objected even if it shipped
        print(outcome.audit, file=sys.stderr)            # pack hash, prompt hash, models, attempts, facts used


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI-written bullet-point reviews or answers using the local SEM knowledge JSONs (requires API credentials).")
    parser.add_argument("--question", help="Ask a question about the batches, SEM reference or supplied sample.")
    parser.add_argument("--sample", type=Path, help="Optional sample-result JSON, including the teammate's native schema.")
    parser.add_argument("--image", action="append", default=[], metavar="DETECTOR=PATH",
                        help="Attach an actual SEM image, e.g. BSE=Batches/Batch_3/img_71vgq3fw_BSE.tif.")
    parser.add_argument("--debug", action="store_true", help="Print diagnostics and source provenance to stderr.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress updates; print only the checked answer.")
    parser.add_argument("--full-context", action="store_true", help="Include detailed selected records and bibliography instead of compact context.")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP operation inactivity timeout in seconds (default 90; not a total-run deadline).")
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum generation attempts including repairs (default 2).")
    parser.add_argument("--max-output-tokens", type=int, default=8000, help="Output ceiling including JSON and citations (default 8000; tokens are used only as needed).")
    args = parser.parse_args()
    try:
        main(debug=args.debug, question=args.question, sample=args.sample, images=args.image,
             quiet=args.quiet, full_context=args.full_context, timeout=args.timeout,
             max_attempts=args.max_attempts, max_tokens=args.max_output_tokens)
    except (OSError, ValueError, ExplainerError) as exc:
        parser.exit(1, f"Unable to generate a grounded response: {exc}\n")
