# Decision Log

Records non-obvious architectural decisions and the reasoning behind them.
Add an entry whenever a choice was made that future contributors might question.

---

## 2026-06-14 — trusted_flight_points Phase 0

**Decision:** Compute and store `trusted_flight_points` in Phase 0 without wiring
it into the fitter.

**Why:** The filter chain (measured-only + tighter contiguous gap + floor bounce)
differs enough from the current `fit_points` path that wiring it immediately
could change CORE RMSE baselines. Storing first lets us measure divergence across
the full 19-clip benchmark before any fit impact.

**Alternatives rejected:**
- Wire into fitter in same PR: too many unknowns; CORE regression risk.
- Derive from `fit_diagnostics.used_in_fit`: conflates geometric pre-selection
  with the fitter's statistical outlier removal — different concepts, different
  extension points.
- Put in `parabola.py` alongside `select_contiguous_flight_cluster`: correct for
  now but `trusted_flight.py` is the documented home for future geometric stages.

**Phase 2 gate:** Zero divergence on `scripts/eval_trusted_flight.py` across all
19 clips before switching the fitter input.

---

## 2026-06-14 — max_flight_gap_frames = 10 vs reacquisition_gap_frames = 15

**Decision:** Use a separate `TrustedFlightConfig.max_flight_gap_frames = 10`
instead of reusing `ShotCandidateConfig.reacquisition_gap_frames = 15`.

**Why:** The two thresholds serve different roles:
- `reacquisition_gap_frames`: how long the shot collector waits before assuming
  the flight has ended (lifecycle decision, triggers finalization).
- `max_flight_gap_frames`: how tightly the trusted selection defines the primary
  arc (geometric quality filter, affects what we call "trusted").

Keeping them separate lets each be tuned without cross-contaminating lifecycle
and fit-quality heuristics.

---

## 2026-06-14 — motion-validation excluded from Phase 0 definition

**Decision:** Horizontal jump, vertical acceleration, and upward-reversal checks
are **not** part of the v1 `trusted_flight_points` definition.

**Why:** Each filter has independent CORE impact that must be measured in
isolation. Bundling them into Phase 0 would make it impossible to attribute a
future RMSE regression to a specific filter. Each gets its own PR, its own
exclusion reason string, and its own CORE-safe gate.

---

## 2026-07-02 — trusted_flight Phase 0 real-clip gate: PASS at max_flight_gap_frames = 10

**Decision:** Accept `TrustedFlightConfig.max_flight_gap_frames = 10` unchanged.
Phase 0 is verified on real benchmark clips; no threshold tuning required.

**Evidence:** Regenerated CORE (A, C, P, R, S) and STRESS (B, E, G, I, K) via
`scripts/run_full_eval_rerun.py`, then ran `scripts/eval_trusted_flight.py`
plus a frame-level trusted-vs-fit comparison:

- CORE A/C/P/R/S: trusted frame set identical to the current fit input
  (21/14/27/17/17 frames respectively), zero post-gap drops, zero divergence.
- B and K exercise the long-gap path: B excludes frame 31 as `post_cluster`
  (24-frame gap), K excludes frames 100 and 104 (41-frame gap). In both cases
  the existing fit path had also excluded those frames, so divergence is zero
  even on gap clips.
- E/G/I show no post-gap drops — their intra-flight gaps are ≤3 frames.
  They stress sparsity/perspective, not gap membership.

**Consequence:** The tighter 10-frame gap never split a real flight in this
suite. Phase 2 wiring remains gated on extending the zero-divergence result to
the remaining 9 benchmark clips.

---

## 2026-07-03 — trusted_flight Phase 1: story window sourced from trusted selection (render-only)

**Decision:** `compute_shot_story` derives `flight_start_frame` /
`flight_end_frame` from `ShotCandidate.trusted_flight_debug` (new
`flight_start_frame` / `flight_end_frame` properties on
`TrustedFlightSelection`), falling back to the previous
`candidate_points[0]` / fit-window logic when the selection is missing or
empty. The fit input, parabola fitting, and confidence are untouched.

**Verification:**
- `tests/test_trusted_flight_story.py` (5 tests): story window == trusted
  window; long-gap (12-frame, > trusted 10 but <= fit 15) story window stops
  at the gap break while the fit still spans it; fit/diagnostics/confidence
  byte-identical whether the story uses trusted or fallback; fallback path;
  empty-selection properties. Full suite 121 passed.
- 10-clip real eval (CORE A/C/P/R/S + STRESS B/E/G/I/K): `parabola_fit`,
  `fit_diagnostics`, `confidence` JSON blocks byte-identical before/after on
  every clip; story windows unchanged on all 10.

**Why windows didn't move on real clips:** every real gap in the suite is
either <= 3 frames (below both thresholds) or >= 24 frames (above both), so
the fit path already excluded the post-gap points. Phase 1 makes the story
window structurally tied to the trusted selection instead of incidentally
matching the fit; the 11-15-frame band where they differ is covered by tests.

**Phase 2 remains gated:** fit-affecting, requires explicit approval plus
pinned golden RMSE/confidence.

---

## 2026-07-05 — Automatic hoop lock via dedicated hoop YOLO weights

**Decision:** add optional `DetectionConfig.hoop_model_path` (CLI
`--hoop-model`) wiring a second, hoop-only `YoloObjectDetector`
(`models/hoop_ball_yolov8.pt`, classes Basketball / Basketball Hoop, sourced
from avishah3/AI-Basketball-Shot-Detection-Tracker — **no license published;
local experimentation only, do not redistribute**). Its hoop-labeled records
merge into the existing `HybridHoopDetector` scoring path. The hoop model runs
only pre-lock and during revalidation; ball detection stays on stock COCO
weights, so the CORE ball path is untouched. Manual bbox / interactive
selection remain as overrides.

**Verification:**
- `scripts/eval_auto_hoop.py`: 19/19 clips lock automatically by frame 3
  (conf ≈ 0.77); |dx_center| ≤ 24 px except B (−68) and C (+59) where the
  lock merges an orange color-region candidate.
- Full-pipeline CORE gate (auto vs manual lock): RMSE improves on **all
  five** CORE clips — A 1.18→0.76, C 0.64→0.56, P 1.48→1.19, R 0.92→0.73,
  S 1.60→1.10 (P/R/S land on the original pinned baselines). All shots still
  detected.

**Note:** the model's tight rim box sits 22–106 px above the hand-drawn
bbox bottoms (which included backboard/pole), shifting rim-relative
thresholds. The CORE gate above is the acceptance evidence.

---

## 2026-07-05 — Make/miss classification (render/observability-only)

**Decision:** new pure `tracking/shot_outcome.py::classify_shot_outcome`
computed in `finalize_shot()` after fit and confidence; stored as
`ShotCandidate.outcome`, serialized as `outcome` block; never read by fit,
confidence, story, or lifecycle. Verdict from the last downward crossing of
the ring line inside the rim x-span; two methods: `measured` (interpolated
straddling pair) then `fit` (descending parabola branch at the ring line,
only when the ball was tracked to within 160 px of it); otherwise honest
`unknown`. Ring line = hoop bbox **top** for squat (tight rim) boxes,
legacy bbox bottom for tall hand-drawn boxes.

**Verification:** `tests/test_shot_outcome.py` (10 tests; full suite 131
passed) + 19-clip agreement run against visually derived labels (see
CURRENT_STATUS; labels pending user confirmation).
