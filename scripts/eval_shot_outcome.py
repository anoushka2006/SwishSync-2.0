#!/usr/bin/env python3
"""Make/miss agreement eval: full pipeline with auto hoop lock vs ground truth.

Runs every Testing clip with the hoop model (automatic lock), then compares
the predicted `outcome.verdict` of the primary shot against the user-confirmed
labels in run_full_eval_rerun.OUTCOME_GROUND_TRUTH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swishsync_cv.config import DetectionConfig, PipelineConfig
from swishsync_cv.pipeline import run_pipeline
from scripts.evaluate_testing_clips import CLIP_LABELS, slugify
from scripts.run_full_eval_rerun import OUTCOME_GROUND_TRUTH

TESTING_DIR = ROOT / "videos" / "Testing"
DEFAULT_OUT = ROOT / "outputs" / "temp" / "outcome_check"


def main() -> None:
    parser = argparse.ArgumentParser(description="Make/miss agreement eval")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--hoop-model",
        default=str(ROOT / "models" / "hoop_ball_yolov8.pt"),
    )
    parser.add_argument(
        "--ball-model",
        default="yolov8n.pt",
        help="Ball detector weights (default stock COCO yolov8n).",
    )
    parser.add_argument("--only", nargs="*", help="Clip labels to run (e.g. A K R)")
    args = parser.parse_args()

    rows = []
    for filename, label in sorted(CLIP_LABELS.items(), key=lambda kv: kv[1]):
        if args.only and label not in args.only:
            continue
        video = TESTING_DIR / filename
        if not video.exists():
            continue
        out_dir = args.out_dir / slugify(filename)
        run_pipeline(
            PipelineConfig(
                input_video=video,
                output_dir=out_dir,
                detection=DetectionConfig(
                    model_path=args.ball_model,
                    hoop_model_path=args.hoop_model,
                ),
            )
        )
        shots = json.loads((out_dir / "shots.json").read_text())
        predicted = "n/a"
        if shots:
            primary = max(shots, key=lambda s: len(s.get("candidate_points", [])))
            predicted = (primary.get("outcome") or {}).get("verdict") or "unknown"
        truth = OUTCOME_GROUND_TRUTH.get(label, "n/a")
        detected_truth = truth if shots or truth == "n/a" else f"{truth} (undetected)"
        rows.append((label, predicted, detected_truth, predicted == truth))

    print(f"\n{'clip':4} {'predicted':>9} {'truth':>18} agree")
    agree = total = 0
    for label, predicted, truth, ok in rows:
        scored = not truth.startswith("n/a") and predicted != "n/a"
        if scored:
            total += 1
            agree += ok
        print(f"{label:4} {predicted:>9} {truth:>18} {'✓' if ok else '✗' if scored else '—'}")
    print(f"\nagreement: {agree}/{total} (shots the pipeline never detects excluded)")


if __name__ == "__main__":
    main()
