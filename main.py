import argparse
import json
import sys
from pathlib import Path

from sentinel import BatteryResult, Explanation, KnowledgePack, explain_battery, collect_images
from sentinel.explainer import ExplainerError
from sentinel.prompts import ImageAsset
from sentinel.questions import GroundedAnswer, answer_question


ROOT = Path(__file__).resolve().parent


def format_review(explanation_json: str) -> str:
    explanation = Explanation.model_validate_json(explanation_json)
    paragraphs = [
        explanation.headline,
        *(item.claim for item in explanation.evidence),
        explanation.why_not_runner_up,
        explanation.agreement,
        " ".join(caveat.strip() for caveat in explanation.caveats if caveat.strip()),
    ]
    return "\n\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())


def format_answer(answer_json: str) -> str:
    answer = GroundedAnswer.model_validate_json(answer_json)
    return "\n\n".join(paragraph.text for paragraph in answer.paragraphs)


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
         images: list[str] | None = None) -> None:
    pack = KnowledgePack.from_project(ROOT)
    if question is not None or sample is not None or images:
        if sample is not None:
            path = Path(sample).expanduser()
            path = path if path.is_absolute() else ROOT / path
            pack = pack.with_sample(json.loads(path.read_text(encoding="utf-8")), origin=str(path))
        if question is None:
            question = ("Write a full professional review of the supplied sample output and any attached images. "
                        "Explain supporting and opposing evidence for the reported classification, compare the "
                        "runner-up, and discuss acquisition confounding and uncertainty. Use the batch statistics "
                        "and SEM reference only where compatible. If no classifier output is supplied, do not "
                        "assign a battery type.")
        outcome = answer_question(question, pack, question_images(images or []))
        print(format_answer(outcome.answer.model_dump_json()))
        if debug:
            print(json.dumps(outcome.audit, ensure_ascii=False, indent=2), file=sys.stderr)
        return
    result = BatteryResult.model_validate_json(
        (ROOT / "examples" / "BAT-07.json").read_text(encoding="utf-8")
    )
    pack = pack.select("SEM classification review BSE ETD InLens " + " ".join(stat.name for stat in result.statistics))
    outcome = explain_battery(result, pack, collect_images(result, root=ROOT / "runs"))

    print(format_review(outcome.explanation.model_dump_json()))      # rendered, real numbers — this goes on the battery page
    if debug:
        print(outcome.generator, file=sys.stderr)        # "llm" or "template" — show this in the UI
        print(outcome.critic_passed, file=sys.stderr)    # False means the critic objected even if it shipped
        print(outcome.audit, file=sys.stderr)            # pack hash, prompt hash, models, attempts, facts used


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review a battery or ask questions using the local SEM knowledge JSONs.")
    parser.add_argument("--question", help="Ask a question about the batches, SEM reference or supplied sample.")
    parser.add_argument("--sample", type=Path, help="Optional sample-result JSON, including the teammate's native schema.")
    parser.add_argument("--image", action="append", default=[], metavar="DETECTOR=PATH",
                        help="Attach an actual SEM image, e.g. BSE=Batches/Batch_3/img_71vgq3fw_BSE.tif.")
    parser.add_argument("--debug", action="store_true", help="Print diagnostics and source provenance to stderr.")
    args = parser.parse_args()
    try:
        main(debug=args.debug, question=args.question, sample=args.sample, images=args.image)
    except (OSError, ValueError, ExplainerError) as exc:
        parser.exit(1, f"Unable to generate a grounded response: {exc}\n")
