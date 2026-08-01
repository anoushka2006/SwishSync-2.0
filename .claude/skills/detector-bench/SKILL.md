---
name: detector-bench
description: Evaluate a new/updated detection model against the frozen baseline the right way — per-class isolation, CORE gate, and an adopt/reject read-out. Use when new weights land in models/, when the user asks to benchmark/compare a detector, or after any Roboflow training finishes.
---

# detector-bench — evaluate a detector without fooling yourself

The one methodology rule (learned the expensive way): **never evaluate a new
model as ball+hoop at once.** A model can be an excellent hoop detector and a
terrible ball detector simultaneously — the combined run hides which. The
trained yolo11n looked catastrophic combined (CORE RMSE 0.76→15.4) but was
byte-identical to baseline as hoop-only.

## 0. Preconditions

- Frozen baseline exists: `models/hoop_ball_yolov8_baseline.pt` (immutable —
  recreate command in `docs/training_plan.md` if missing).
- Candidate weights have their OWN filename (never overwrite an existing model
  file). Verify it loads and print classes:
  `python -c "from ultralytics import YOLO; print(YOLO('<path>').names)"`
- Class names must map through `DetectionConfig` aliases (basketball/rim/hoop
  variants are covered; anything else → extend aliases first).

## 1. Isolation runs (order matters — cheapest, most decisive first)

1. **Hoop-only:** `python scripts/check_core_drift.py` after pointing its
   hoop model at the candidate (or run eval with `--hoop-model <candidate>`).
   Gate: every CORE clip |ΔRMSE| ≤ 0.15.
2. **Ball-only (CORE clips first):** candidate as `--ball-model`, baseline
   hoop. If CORE RMSE blows up → ball class is noisy → try confidence sweep
   (0.5 / 0.7 / 0.85) before rejecting.
3. **Combined**, only if both isolations pass.

## 2. Full benchmark

```bash
python scripts/run_detector_benchmark.py   # baseline vs candidate, all 19
```

Read the four summary lines: IMPROVED / REGRESSED / STILL FAILING / CORE-worse.

## 3. Adopt / reject read-out (write it, don't vibe it)

| Question | Evidence required |
|---|---|
| CORE safe? | drift gate output, all 5 clips |
| Recall better? | measured-points deltas on E, H, I, K |
| Verdicts better? | wrong-verdict count vs `OUTCOME_GROUND_TRUTH` (must not increase) |
| F/J recovered? | shot + fit present |
| Hoop consistent? | 19/19 freeze ≤ frame 18, |dx| ≤ 15px |

Decision → `docs/DECISION_LOG.md` entry (adopt OR reject — rejections are
decisions too). Adoption = the candidate becomes the new active model file;
the baseline file NEVER changes.

## Known traps

- Count proxies ≠ recall: more detections without more measured points =
  false positives, not recall.
- A verdict "improvement" that comes with hoop-geometry churn is noise —
  check the freeze diagnostics before crediting the model.
- pip installs for model tooling can clobber opencv/numpy → rerun `pytest` +
  CORE gate after ANY such install.
