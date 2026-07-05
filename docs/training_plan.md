# Own hoop/ball weights — training plan

Goal: replace the unlicensed `models/hoop_ball_yolov8.pt` with weights we own,
with rim-only boxes and far-court coverage (fixes D/E/G unknowns).

## Data sources (in order)

1. **Own clips** — `python scripts/extract_training_frames.py` writes
   annotation frames to `datasets/own_clips/images/` (rim-interaction biased).
   Annotate classes: `basketball`, `rim` (ring only — NOT the net/backboard).
2. **CC-BY 4.0 Roboflow Universe datasets** (verify license shown on each
   dataset page at download time; keep attribution in this file):
   - `computer-vision-d5fjh/basketball-detection-dn6fg` — ~4.9k images,
     person/ball/hoop, CC BY 4.0.
   - `loganwork/basketball-rdtyv` — 366 images incl. a `Rim` class, CC BY 4.0.
   Download via Roboflow (free account) in YOLOv8 format.
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
