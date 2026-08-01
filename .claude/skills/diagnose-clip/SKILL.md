---
name: diagnose-clip
description: Visually debug a Testing clip — extract rim-zone or full-frame montages around the moment in question and READ them before concluding anything. Use when a clip's verdict/arc/lock looks wrong, when deriving or checking make/miss evidence, or whenever a rendering claim needs eyes on pixels.
---

# diagnose-clip — look at the pixels before theorizing

Rule this skill exists to enforce: **no claim about what a clip shows without
extracting frames and reading them.** Trajectory JSON tells you where the
tracker THINKS the ball was; only frames tell you where it WAS.

## 1. Locate the moment

From the clip's `shots.json` (in the eval output dir, slug via
`scripts/evaluate_testing_clips.slugify`):
- flight end: max `frame_index` in `candidate_points`
- release: `story.release_frame`
- crossing: `outcome.crossing_frame`
- finalize reason: pipeline log (`floor_idle` / `idle` / `end_of_video`) —
  reason `end_of_video` on a mid-clip shot is itself the bug (zombie candidate).

## 2. Extract a montage (scratchpad, not the repo)

Pattern — rim-zone crops, N tiles, frame numbers burned in:

```python
import cv2, numpy as np
cap = cv2.VideoCapture("videos/Testing/<file>")
frames = [start + i*stride for i in range(8)]        # window around the moment
rim = (x1, y1, x2, y2)                                # hoop-model box or lock box
w = x2 - x1
crop_box = (int(cx - 1.6*w), int(y1 - 0.8*w), int(cx + 1.6*w), int(y2 + 2.2*w))
tiles = []
for fi in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
    t = f[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]].copy()
    cv2.putText(t, str(fi), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    tiles.append(t)
cv2.imwrite("<scratchpad>/montage.jpg", np.hstack(tiles))
```

Then **Read the image**. Single-row montages of ≤ 8 tiles stay legible; if the
window was wrong, re-extract with a shifted window — do not squint at the
wrong frames.

## 3. Interpretation guide (rim zone)

- Ball emerges below ring INSIDE span, deep (≥ 0.35 rim-width below) → make.
- Barely below ring inside span → usually the ball passing IN FRONT of the
  rim on arrival (2D depth trap) — ambiguous, not make evidence.
- Below ring outside span → miss. Rises above ring after contact → miss.
- Ball wedged/lingering on the rim for many frames → widen the window until
  it falls somewhere.
- Two identical detections at identical coords across frames = static clutter,
  not the ball.

## 4. Processed-output checks

Grab the LAST frames of `processed.mp4` for arc/verdict-color claims (the arc
appears only after finalize; mid-flight frames legitimately show dots only).
Compare against the raw clip when deciding whether tracker or renderer is at
fault. Verdict labels are provisional until the USER confirms them —
`OUTCOME_GROUND_TRUTH` only changes on user confirmation.
