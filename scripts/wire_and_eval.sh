#!/usr/bin/env bash
# Wire the trained detector + run the comparison and full eval.
# Runs LOCALLY (needs the Testing videos + laptop on). Safe to schedule/cron
# once the trained weights are downloaded.
#
# Prereq: download the trained yolo11n weights from Roboflow (UI: model ->
# Deploy -> Download Weights, format yolov11/pytorch) to:
#   models/hoop_ball_yolo11n.pt
set -uo pipefail
cd "$(dirname "$0")/.."

WEIGHTS="models/hoop_ball_yolo11n.pt"
if [ ! -f "$WEIGHTS" ]; then
  echo "Missing $WEIGHTS."
  echo "Download the trained model from Roboflow (basketball-strategy/"
  echo "cv-cnfd4-eaond-1-yolo11n-t1) as ultralytics .pt into models/, then rerun."
  exit 1
fi

echo "== detector comparison (baseline vs trained) =="
python scripts/compare_detectors.py \
  --candidate-ball "$WEIGHTS" \
  --candidate-hoop "$WEIGHTS" \
  --out outputs/temp/detector_cmp/summary.json

echo "== full make/miss eval with trained weights =="
python scripts/eval_shot_outcome.py --out-dir outputs/temp/outcome_trained \
  --ball-model "$WEIGHTS" --hoop-model "$WEIGHTS" \
  | grep -E "^[A-S] |agreement"
echo "Done. Compare agreement/arcs vs the yolov8 baseline before adopting."
