"""Run the whole pipeline, or a range of stages, in order.

  python run_pipeline.py                    # everything except the paid AI-annotation stages
  python run_pipeline.py --from teacher     # resume from a stage
  python run_pipeline.py --only metrics validate

The AI-annotation stages (src.ai_labels, src.ai_validate, src.triangle_test) call the Anthropic API and cost money; run them
explicitly (see README). Their results are committed in outputs/metrics/, so the pipeline can be rebuilt
without re-running them as long as the superpixels are regenerated with the same settings (src.ai_labels prepare).
"""
import argparse
import subprocess
import sys

STAGES = [
    ("manifest", ["src.manifest"]),
    ("preprocess", ["src.preprocess"]),
    ("seeds", ["src.seeds"]),
    ("features", ["src.features"]),
    ("superpixels", ["src.ai_labels", "prepare"]),
    ("ai_apply", ["src.ai_labels", "apply"]),
    ("teacher", ["src.teacher", "train"]),
    ("ablation", ["src.teacher", "ablation"]),
    ("stability", ["src.teacher", "stability"]),
    ("student", ["src.student", "train"]),
    ("student_predict", ["src.student", "predict"]),
    ("export_onnx", ["src.student", "export"]),
    ("finalize", ["src.finalize", "--source", "student"]),
    ("instances", ["src.instances"]),
    ("metrics", ["src.metrics"]),
    ("validate", ["src.validate"]),
    ("figures", ["src.figures"]),
    ("nnunet", ["src.export_nnunet"]),
    # batch identification (docs/route_to_80 synthesis, trimmed)
    ("dino", ["src.dino"]),
    ("bid_features", ["src.bid_features", "--labels", "final"]),
    ("bid_cache", ["src.bid_model", "cache", "--labels", "final"]),
    ("bid_score", ["src.bid_model", "score"]),
    ("bid_null", ["src.bid_model", "null", "--n", "200"]),
    ("bid_fit", ["src.bid_model", "fit"]),
    # v2: acquisition-fingerprint model + agreement-rule decision layer
    ("fingerprint", ["src.fingerprint"]),
    ("fingerprint_model", ["src.fingerprint_model", "fit"]),
    ("decision", ["src.decision", "evaluate"]),
    # v3: the teammate's additions (texture model, matcher, GET4 range rule, calibration, nested threshold)
    ("texture_features", ["src.texture_model", "features"]),
    ("known_spot_index", ["src.known_spot", "index"]),
    ("known_spot_eval", ["src.known_spot", "evaluate"]),
    ("v3_prepare", ["src.v3", "prepare"]),
    ("v3_evaluate", ["src.v3", "evaluate"]),
    ("v3_fit", ["src.v3", "fit"]),
    ("bid_explain", ["src.bid_explain", "samples"]),
    # exploratory analyses (logged in outputs/metrics/look_ledger.csv)
    ("bid_compare", ["src.bid_compare", "--n", "200"]),
    ("kpi", ["src.kpi"]),
    ("kpi_topo", ["src.kpi_topo"]),
    ("b12", ["src.b12", "--n", "2000"]),
    ("roughness_check", ["src.roughness_check"]),
    ("same_batch", ["src.same_batch"]),
    ("fingerprint_b12", ["src.fingerprint_b12"]),
    ("head_ablation", ["src.head_ablation"]),
    ("cascade_test", ["src.cascade_test"]),
    ("final_model", ["src.final_model"]),
    ("compare_versions", ["src.compare_versions"]),
]


def main():
    names = [s for s, _ in STAGES]
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=names)
    ap.add_argument("--only", nargs="*", choices=names)
    args = ap.parse_args()
    todo = STAGES
    if args.start:
        todo = STAGES[names.index(args.start):]
    if args.only:
        todo = [s for s in STAGES if s[0] in args.only]
    for name, cmd in todo:
        print(f"\n=== {name}: python -m {' '.join(cmd)}", flush=True)
        if subprocess.call([sys.executable, "-m", *cmd]) != 0:
            sys.exit(f"stage {name} failed")


if __name__ == "__main__":
    main()
