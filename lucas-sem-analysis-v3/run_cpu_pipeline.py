"""Rebuild the label maps and overlays on a CPU-only machine, from the raw TIFFs.

The delivered segmentation comes from the U-Net student, which needs a CUDA GPU and the v1.0 model weights.
Without them, this runs the deterministic stages plus the LightGBM teacher (CPU) and finalises the teacher's
label maps into outputs/segmentation/. It does not touch the committed student-based metrics in
outputs/metrics/ (phase_fractions, batch_stats, particles, ...). The teacher's own fractions go to
outputs/metrics/teacher_cpu_fractions.csv.

  python run_cpu_pipeline.py                 # all CPU stages
  python run_cpu_pipeline.py --from teacher  # resume
  SEM_WORKERS=3 python run_cpu_pipeline.py   # fewer parallel processes on machines with < 16 GB RAM

Raw data is read from SEM_RAW_DIR, else the hackathon repo's data/raw/batch_N (see src/config.py).
"""
import argparse
import subprocess
import sys

STAGES = [
    ("preprocess", ["src.preprocess"]),
    ("seeds", ["src.seeds", "--overlay"]),           # no overlay figures: keep the committed report figures
    ("features", ["src.features"]),
    ("superpixels", ["src.ai_labels", "prepare"]),
    ("ai_apply", ["src.ai_labels", "apply"]),         # the committed, paid AI labels; no API calls
    ("teacher", ["src.teacher", "train", "--tag", "_cpu"]),
    ("finalize", ["src.finalize", "--source", "teacher_cpu"]),
]


def main():
    names = [s for s, _ in STAGES]
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=names)
    args = ap.parse_args()
    todo = STAGES[names.index(args.start):] if args.start else STAGES
    for name, cmd in todo:
        print(f"\n=== {name}: python -m {' '.join(cmd)}", flush=True)
        if subprocess.call([sys.executable, "-m", *cmd]) != 0:
            sys.exit(f"stage {name} failed")


if __name__ == "__main__":
    main()
