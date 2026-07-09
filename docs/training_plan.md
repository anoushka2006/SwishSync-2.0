# Own hoop/ball weights — training plan

Goal: replace the unlicensed `models/hoop_ball_yolov8.pt` with weights we own,
with rim-only boxes and far-court coverage (fixes D/E/G unknowns).

## Data sources (in order)

1. **Own clips** — `python scripts/extract_training_frames.py` writes
   annotation frames to `datasets/own_clips/images/` (rim-interaction biased).
   Annotate classes: `basketball`, `rim` (ring only — NOT the net/backboard).
2. **CC-BY 4.0 Roboflow Universe datasets** (attribution kept here):
   - **FORKED** → `basketball-strategy/cv-cnfd4-eaond` (from `cv-8scak/cv-cnfd4`,
     3666 imgs, classes basketball/people/rim, CC BY 4.0, most-downloaded).
     Next: `versions_generate` → `versions_export` yolov8.
   - Alt clean 2-class: `wennn/basketball-6hbv9` (1341 imgs, basketball/rim).
   - Alt large: `basketball-hoop-tsdku/basketball-and-rim` (6270 imgs).
   All via Roboflow MCP (workspace basketball-strategy).
3. **Going forward:** user-collected footage (highest value: matches courts,
   lighting, angles).

## Not allowed

- Distilling / pseudo-labeling from the unlicensed avishah3 weights.
- NBA/broadcast or scraped social-media footage.

## Training

- Base: `yolov8n` fine-tune (note Ultralytics is AGPL — fine locally; for a
  commercial app either buy an Ultralytics license or retrain the same dataset
  on an Apache-licensed detector, e.g. RT-DETR/PP-YOLOE family).
- Acceptance: `scripts/eval_auto_hoop.py` 19/19 locks; full-pipeline CORE gate
  vs current model; rim-detection at far court (D/E/G) above 0.35 conf near
  the ring during ball arrival.

## Wiring the trained model (2026-07-06)

Training finished: `basketball-strategy/cv-cnfd4-eaond-1-yolo11n-t1` (yolo11n,
classes basketball/people/rim). The pipeline detector is pluggable by path —
same `YoloObjectDetector` / `Detector` interface, same `SparseBallDetection`
output — so the trained model slots in without downstream changes.

**Your one manual step (opencv-safe):** in Roboflow, open the trained model ->
Deploy -> Download Weights (ultralytics/yolov11 .pt) -> save to
`models/hoop_ball_yolo11n.pt`. (Not automated: no MCP weight-download tool, and
the roboflow SDK install risks clobbering opencv.)

**Frame uploads are REST, never SDK** (2026-07-09): the uploader script and the
curl path both hit `api.roboflow.com/dataset/<slug>/upload` directly. Key note:
the classic 20-char private key works for this endpoint; the `rf_`-prefixed key
does NOT (401). Batch `swishsync-own-clips` holds the own-court frames.

**Then, laptop on:**
```bash
scripts/wire_and_eval.sh          # comparison + full make/miss eval, trained weights
```

`scripts/compare_detectors.py` reports baseline vs candidate across A-S:
shot-detection rate, measured points/shot (recall proxy), trajectory
completeness, weighted RMSE, confidence, detection count (FP proxy), fps.
True recall / FP need per-frame ball GT (A-S unannotated) — a future annotation
task; the count proxies stand in meanwhile.

## Baseline weights convention (2026-07-06)

Keep a frozen baseline detector to measure every future model against:
- `models/hoop_ball_yolov8_baseline.pt` — the original avishah3 weights
  (classes Basketball / Basketball Hoop). **Never overwrite.** Recreate if lost:
  `curl -sL -o models/hoop_ball_yolov8_baseline.pt \
   https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker/raw/master/best.pt`
- `models/hoop_ball_yolov8.pt` — current candidate/active detector (iterated).

Benchmark a new detector vs the frozen baseline:
`python scripts/run_detector_benchmark.py` — per-clip shot/measured/RMSE/verdict,
plus IMPROVED / REGRESSED / STILL FAILING / CORE-worse summary.
