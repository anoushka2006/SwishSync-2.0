# Filming spec — new testing & training clips

What to film so every clip serves both the benchmark and detector training.

## Camera
- Tripod / fully static (the pipeline's hoop-freeze and homography assume it).
- Whole flight visible: shooter, full arc, rim, and 2–3 s after the ball lands.
- Angles: primary set = side-ish view like the current A–S clips (the make/miss
  geometry assumes it). ALSO film some new angles (45°, closer, farther,
  slightly behind shooter) — those enter as a new STRESS category and train
  detection robustness; they are NOT expected to pass make/miss yet.

## Content
- **Different shooters: encouraged.** Any number of people across clips; keep
  ONE active shooter in frame at a time (pose picks the top person).
- **Makes above all**: target ≈ 50/50 make/miss overall (current set is
  6 make / 13 miss — make-starved).
- **Rim rattles, both outcomes**: ≥ 8 clips where the ball dances on the rim
  (the C/L/N/Q class of problem needs examples to tune against).
- **Multi-shot workout clips: the majority.** 3–10 shots per clip, natural
  rebound/dribble between shots. Plus a few single-shot clips for clean
  labeling. Multi-shot clips feed the workout-pipeline milestones (M8/M9).
- Variety: different courts, lighting (evening/indoor/overcast), ball colors.

## Per-clip label (jot while filming — memory fades)
- Ordered outcomes per shot, e.g. `clip_T: make, miss, make, make, miss`
- Rough shot count is enough for order matching; exact frames not needed.

## Volume target
- 20–30 clips. Every clip double-dips: benchmark entry + frames for
  detector training (rim-interaction frames auto-extracted).
