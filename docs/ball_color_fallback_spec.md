# Phase 2 Spec: HSV / Circularity Ball Fallback

**Status:** Spec only — do not implement until Phase 1 (floor-band detection gates) is merged and stable.

**Phase 1 prerequisite:** [`detection_cleanup_eval.json`](../outputs/eval/shot_story/detection_cleanup_eval.json) shows CORE clips unchanged by floor-band gating (A/B filter causality check identical RMSE).

---

## Problem

Clips **F** and **J** remain zero-shot after Phase 1:

| Clip | Issue |
|------|-------|
| **F** | Zero YOLO basketball labels during flight void (f13–145) despite dense idle inference |
| **J** | 45-frame flight void (f45–89); brief f40–44 burst blocked by lifecycle rim-reacquisition gate |

Phase 1 removes floor-band false positives but does **not** recover missed flight detections.

---

## Proposed fallback (future PR)

### Trigger conditions (all required)

1. `floor_band_rejection_enabled=True` (Phase 1 gates active)
2. Hoop locked
3. Pre-first-shot idle (`lifecycle_state == idle` and no finalized shots)
4. Full-frame YOLO + Phase 1 gates return **zero** upper-court basketball candidates for the frame
5. Feature flag `color_ball_fallback_enabled=True` (default **False**)

### Search region

- Restrict to existing lifecycle start zone: `center_y < rim_center_y + start_below_rim_margin_px`
- Exclude floor band: `center_y <= rim_center_y + floor_below_rim_margin_px`

Reuse constants from [`ShotCandidateConfig`](../src/swishsync_cv/config.py); do not introduce new geometry.

### Cues

| Cue | Starting point |
|-----|----------------|
| Orange HSV | [`HoopLockConfig`](../src/swishsync_cv/config.py) hue/sat/val ranges (hoop detector baseline) |
| Blob size | Min/max diameter in pixels tuned from CORE clip ball bbox stats |
| Circularity | `4π·area / perimeter²` on largest contour; reject below threshold (~0.6–0.75) |

### Output

- Synthetic [`SparseBallDetection`](../src/swishsync_cv/data.py) with low confidence tag (e.g. 0.20)
- Must still pass existing collection start gates in [`shot_candidate.py`](../src/swishsync_cv/tracking/shot_candidate.py)
- Register via [`SparseBallDetectionBuffer.register_detection`](../src/swishsync_cv/tracking/sparse_detection.py)

### Wiring (future)

```
pipeline.py
  after filter_basketball_detections + extract_ball_detection miss
  before try_roi_ball_detection (ROI still requires active track)
  call try_color_ball_fallback(...)  # new module, flag-gated
```

**Do not combine** with Phase 1 gate changes in the same PR.

---

## CORE safety

- Default flag off; benchmark-only enable for F/J/D stress runs
- Acceptance: CORE A/C/P/R/S RMSE drift ≤ 0.05px vs Phase 1 frozen baseline **with flag off**
- With flag on: measure F/J recovery separately; reject if CORE candidate counts or RMSE shift

---

## Clip-specific expectations

| Clip | Fallback may help? | Blocker if fallback alone |
|------|-------------------|---------------------------|
| **F** | Yes, if ball is visible orange/round during void | COCO blindness may persist for non-orange motion blur |
| **J** | Unlikely in void; burst already has YOLO hits | Lifecycle `_looks_like_rim_reacquisition` — separate PR |

---

## Out of scope for Phase 2 implementation

- Trajectory fitting, rendering, hoop lock changes
- Lower YOLO confidence thresholds
- ROI threshold changes
- Custom YOLO fine-tuning (separate milestone if fallback insufficient)
- Kalman / temporal trackers

---

## Phase 2 acceptance pre-work

Before implementing fallback:

1. Phase 1 merged and stable ≥ 1 eval cycle
2. CORE RMSE on A/C/P/R/S within 0.05px of Phase 1 post-cleanup baseline (filter on/off identical per clip)
3. No new lifecycle regressions on D/H stress clips
4. This spec reviewed and flag defaults confirmed
