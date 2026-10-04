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
from sentinel.pipeline_facts import FactsResponse, PipelineSample, explain_facts_document
from sentinel.baselines import BatchBaselines, load_default_baselines

ROOT = Path(__file__).resolve().parent
DEFAULT_RESPONSE_CACHE = ROOT / "explanations" / ".cache"


def format_points(points, *, separator: str = "\n\n") -> str:
    bullets = []
    for point in points:
        if not point.strip():
            continue
        lines = point.strip().splitlines()
        first = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", lines[0])
        bullets.append("- " + first + "".join("\n  " + line.strip() for line in lines[1:]))
    return separator.join(bullets)


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


def _one_point(text: str) -> str:
    """One reader-facing point: no bullet marker, no embedded line breaks."""
    text = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


#: How points are joined into the one string per image: a bullet per line.
POINT_PREFIX = "- "
POINT_SEPARATOR = "\n"


def format_facts_response(response_json: str, samples: list[PipelineSample]) -> dict[str, str]:
    """``{sample_id: "- point\\n- point\\n..."}`` in input order, one entry per image.

    The value is the explanation only, as one string with a bullet per line.
    The category and its probabilities are shown elsewhere on the page, so
    neither the heading nor any score is repeated here; the key identifies the
    image.
    """
    response = FactsResponse.model_validate_json(response_json)
    by_id = {item.sample_id: item for item in response.explanations}
    expected = {sample.sample_id for sample in samples}
    if len(response.explanations) != len(samples) or set(by_id) != expected:
        raise ValueError("the response must contain exactly one explanation per input sample")
    result: dict[str, str] = {}
    for sample in samples:
        item = by_id[sample.sample_id]
        if item.reported_answer != sample.answer:
            raise ValueError(f"the response changed the decision for {sample.sample_id}")
        points = [p for p in (_one_point(point.text) for point in item.points) if p]
        result[sample.sample_id] = POINT_SEPARATOR.join(POINT_PREFIX + p for p in points)
    return result


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
         images: list[str] | None = None, quiet: bool = True, full_context: bool = False,
         timeout: float = 90.0, max_attempts: int = 3, max_tokens: int = 8000,
         facts: str | Path | None = None, legacy_result: BatteryResult | None = None,
         dino_only: bool = False, baselines: str | Path | None = None,
         use_baselines: bool = True, effort: str | None = "low",
         critic: str = "advisory", model: str | None = None,
         use_cache: bool = True, cache_dir: str | Path | None = None,
         refresh_cache: bool = False) -> dict[str, str] | None:
    if facts is not None and sample is not None:
        raise ValueError("supply facts or sample, not both")
    if legacy_result is not None and (facts is not None or sample is not None or question is not None or images):
        raise ValueError("legacy_result cannot be combined with facts, sample, question or image inputs")
    if max_attempts < 1 or max_tokens < 1:
        raise ValueError("max-attempts and max-output-tokens must be positive")
    if refresh_cache and not use_cache:
        raise ValueError("--refresh-cache cannot be combined with --no-cache")
    selected_model = model or ExplainerConfig().model
    config = ExplainerConfig(model=selected_model, timeout_seconds=timeout, max_tokens=max_tokens,
                             effort=None if selected_model.startswith("claude-haiku-") else effort)
    started = time.monotonic()

    def report(message):
        print(f"[Sentinel +{time.monotonic() - started:.1f}s] {message}", file=sys.stderr, flush=True)

    progress = None if quiet else report
    if progress:
        progress("Loading knowledge and selecting relevant context")
    load_dotenv(start=ROOT)
    # DINO-only mode withholds every non-DINO input by design, so it loads nothing else.
    pack = KnowledgePack.empty() if dino_only else KnowledgePack.from_project(ROOT)
    batch_baselines = None
    if not dino_only and use_baselines:
        batch_baselines = (BatchBaselines.from_reference_dir(Path(baselines).expanduser())
                           if baselines is not None else load_default_baselines(ROOT))
        if progress:
            progress(f"Batch baselines: {len(batch_baselines.records)} reference images from {batch_baselines.origin}"
                     if batch_baselines else "Batch baselines: none found; explaining from each record alone")
    facts_path = facts if facts is not None else sample
    if legacy_result is None and (facts_path is not None or (question is None and not images)):
        if images:
            raise ValueError("facts-file narration is text-only; use reported metrics and map_text, not --image attachments")
        path = Path(facts_path).expanduser() if facts_path is not None else ROOT / "examples" / "facts.json"
        path = path if path.is_absolute() else ROOT / path
        document = path.read_bytes().decode("utf-8-sig")
        outcome = explain_facts_document(document, pack, config=config,
                                         max_retries=max_attempts - 1, full_context=full_context,
                                         question=question, progress=progress,
                                         decision_mode="dino" if dino_only else "pipeline",
                                         baselines=batch_baselines, critic_mode=critic, short_ids=True,
                                         cache_dir=(Path(cache_dir).expanduser() if cache_dir is not None
                                                    else DEFAULT_RESPONSE_CACHE) if use_cache else None,
                                         refresh_cache=refresh_cache)
        explanations = format_facts_response(outcome.response.model_dump_json(), outcome.samples)
        print(json.dumps(explanations, ensure_ascii=False, indent=2))
        if debug:
            print(json.dumps(outcome.audit, ensure_ascii=False, indent=2), file=sys.stderr)
        return explanations
    if question is not None or images:
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
    result = legacy_result
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
    parser = argparse.ArgumentParser(description="Explain a whole facts.json as a JSON object mapping each image id to its list of reasoning points; defaults to examples/facts.json.")
    parser.add_argument("--question", help="Ask a knowledge question, or focus each image explanation when --facts is supplied.")
    parser.add_argument("--facts", "--sample", dest="facts", type=Path,
                        help="Pipeline JSON, read unchanged as one whole document. --sample is a compatibility alias.")
    parser.add_argument("--image", action="append", default=[], metavar="DETECTOR=PATH",
                        help="Attach an actual SEM image, e.g. BSE=Batches/Batch_3/img_71vgq3fw_BSE.tif.")
    parser.add_argument("--debug", action="store_true", help="Print diagnostics and source provenance to stderr.")
    parser.add_argument("--verbose", action="store_true", help="Show progress updates on stderr. By default only the result is printed.")
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)  # the default now; kept so old commands still run
    parser.add_argument("--full-context", action="store_true", help="Include detailed selected records and bibliography instead of compact context.")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP operation inactivity timeout in seconds (default 90; not a total-run deadline).")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum generation attempts including repairs (default 3).")
    parser.add_argument("--max-output-tokens", type=int, default=8000, help="Output ceiling including JSON and citations (default 8000; tokens are used only as needed).")
    parser.add_argument("--dino-only", action="store_true", help="Narrate only the DINO score ranking, withholding every other input (the previous default).")
    parser.add_argument("--baselines", type=Path, help="Folder of reference facts JSONs with true_batch (default: lucas-sem-analysis-v3/outputs/batchid).")
    parser.add_argument("--no-baselines", action="store_true", help="Do not compare with batch baselines.")
    parser.add_argument("--critic", choices=["advisory", "blocking", "off"], default="advisory",
                        help="Grounding critic: advisory records objections in the audit (default); blocking lets them reject the answer; off skips the call (fastest).")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max", "default"], default="low",
                        help="Thinking effort (default low; medium restores the previous setting; ignored for Haiku; 'default' sends nothing).")
    parser.add_argument("--model", help="Override the generation model (default: Sonnet; smaller models may need more repairs).")
    parser.add_argument("--no-cache", action="store_true", help="Generate afresh without reading or writing the checked-response cache.")
    parser.add_argument("--refresh-cache", action="store_true", help="Generate afresh and replace the cache entry only after checks pass.")
    parser.add_argument("--cache-dir", type=Path, help="Private response-cache directory (default explanations/.cache; entries expire after 24 hours).")
    args = parser.parse_args()
    try:
        main(debug=args.debug, question=args.question, facts=args.facts, images=args.image,
             quiet=not args.verbose, full_context=args.full_context, timeout=args.timeout,
             max_attempts=args.max_attempts, max_tokens=args.max_output_tokens,
             dino_only=args.dino_only, baselines=args.baselines, use_baselines=not args.no_baselines,
             effort=None if args.effort == "default" else args.effort, critic=args.critic,
             model=args.model, use_cache=not args.no_cache, cache_dir=args.cache_dir,
             refresh_cache=args.refresh_cache)
    except (OSError, ValueError, ExplainerError) as exc:
        parser.exit(1, f"Unable to generate a grounded response: {exc}\n")
