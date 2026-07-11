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

---

## 2026-07-05 — Rim-zone re-scan + outcome ground truth + arc coloring

**Decision:** make/miss verdicts are now refined by direct observation of the
rim zone (`refine_outcome_with_rim_zone`, still render-only). The pipeline
keeps a rolling 72-frame buffer of rim crops; when a shot finalizes — even at
end-of-video, where candidates that linger waiting for reacquisition close —
ball detection runs retroactively over buffered crops past the flight end,
then live for 48 more frames. Evidence rules: first below-ring emergence
decides; an inside-span make must emerge ≥ 0.35×rim-width below the ring
(shallow inside points are the ball passing in FRONT of the rim); rattle
exception (inside+deeper shortly after a near-rim outside point); sustained
rise above the ring after rim contact = miss (user's bounce heuristic);
static near-identical detections are dropped as rim clutter; min detection
confidence 0.35 (real balls score 0.85+). Finalized arcs render green/red by
verdict (`arc_drawing.py`), falling back to the legacy color for unknown.

**Ground truth (user-confirmed, in `run_full_eval_rerun.OUTCOME_GROUND_TRUTH`):**
makes A C F L N Q; misses B D E G H I J K M O P R S.

**Verification:** `scripts/eval_shot_outcome.py` — agreement 14/17 with ZERO
wrong verdicts: 14 correct, 3 honest unknowns (D, E, G: far-court clips where
the ball at the rim never clears 0.35 confidence — fix belongs to own-weights
training). F, J remain undetected shots (FAILURE clips). Full suite 135
passed. CORE untouched: all rescan state is render-only.

**Baselines re-pinned** in `run_full_eval_rerun.BASELINE_RMSE` to the
2026-07-05 manual-lock rerun (A 1.18, C 0.64, P 1.48, R 0.92, S 1.60).

---

## 2026-07-05 — Rim-only ring geometry + entry-angle metric

**Decision:** `HoopLock.rim_bbox_xyxy` — an orange-ring-only sub-box refined
from the HSV color mask inside the hoop bbox (`refine_rim_bbox`,
hoop_detector.py), computed opportunistically in `HoopLockTracker.update`
(auto locks only; manual locks keep legacy geometry). `shot_outcome` prefers
it for ring line + x-span via `_rim_geometry`. New render-only metric
`ShotOutcome.entry_angle_deg` — arc angle vs horizontal at the ring crossing
from the parabola slope (Noah/Pillar research says ~45° is optimal; the
metric is licensing-safe, their patents cover real-time feedback systems).
Shown in the eval browser.

**Verification:** 138 tests passing; outcome agreement unchanged at 14/17
(zero wrong verdicts) with rim-only geometry active. Entry angles physically
plausible: makes cluster 38–45°, misses scatter 19–55°.

---

## 2026-07-05 — floor_idle finalize (fixes mid-video arc rendering)

**Decision:** a sustained streak of excluded floor-bounce points
(>= max_idle_frames, default 8) finalizes the active candidate with new
reason `floor_idle`. Previously a ball bouncing on the floor after the shot
kept the candidate alive to end-of-video (every measured point reset the
activity clock even when excluded as a floor bounce), so several clips never
rendered a fitted arc or verdict color mid-clip. Fit inputs are unchanged —
candidate_points membership is identical; only finalize timing moves earlier.
Also: finalized arc thickness 3→5 (extension 2→3) so the arc reads thicker
than the 4px-radius dots.

**Verification:** outcome agreement unchanged at 14/17; CORE RMSE identical
(A 0.76, C 0.56, P 1.19, R 0.73, S 1.10 on auto-lock); fits now exist
mid-video on all clips with sufficient points; 138 tests pass.

---

## 2026-07-06 — Hoop lock freeze (static camera) + full solid arc

**Decision:** `HoopLockConfig.freeze_when_locked` (default True). Once the lock
CONVERGES (>= 6 consecutive locked frames with center drift <= 2.5px), it
freezes: detector no longer runs, no smoothing/revalidation, position fixed.
Fixes reported bug — hoop anchor drifting when the ball passes through the rim
(ball+rim merge scored high, dragging the smoothed lock). Static-camera
assumption makes this safe. Freeze only AFTER convergence: freezing at first
confident lock (frame ~3) used a coarse box and flipped borderline verdicts.

**Arc render:** `draw_finalized_arc` now draws one continuous smooth solid arc
across the whole render span (release → rim) in the make/miss color with a dark
underlay outline, replacing the stubby observed-segment + dotted-grey extension.
Thickness 5 (outline 8), reads thicker than 4px dots. Both panes use it.

**Verification:** 140 tests pass. CORE RMSE identical (A .76 C .57 P 1.19
R .73 S 1.10) — freeze doesn't touch the fit. Outcome 13/17 (was 14): freeze
flips B/C because their prior "correct" verdicts rode the anchor drift we
removed; recovers G. Borderline geometry — own rim-only weights (task #2)
resolve it. Net: correct behavior over a borderline metric artifact.

---

## 2026-07-06 — Mid-video arc rendering + live verdict colour

**Two bugs behind "arc not rendering / wrong colour":**
1. Immortal candidate: after the ball is lost, the pipeline emits interpolated
   gap-fill points every frame; these reset the idle timer and are never added
   or floor-excluded, so the candidate never finalized until end-of-video and
   the fitted arc was computed post-loop (never drawn). Fix: cap consecutive
   interpolated points at `reacquisition_gap_frames` (15) → finalize (idle).
2. Stale arc colour: make/miss verdict is settled by the rim re-scan, which
   only re-refined at window end (finalize+48). A make decided mid-window
   showed red until then. Fix: re-refine every frame as rim-zone points arrive
   (refine is a cheap list scan) + settle once from buffered crops at finalize.

**Result:** finalized arc now renders mid-clip (L: from ~frame 85, was never),
full solid smooth arc, correct green/miss-red the moment the crossing lands.
Outcome agreement 14/17 unchanged; 140 tests pass; candidate_points membership
identical (interpolated never entered the fit) so fits unchanged.

---

## 2026-07-06 — Render-only ball trail (end-to-end tracking, task #3)

**Decision:** a rolling ~2.5s ball-dot trail on the left panel traces the ball
across the whole clip (dribbles + pre/post-shot), matching the reference look.
Captured from the RAW detections BEFORE the floor gate (so low dribbles show)
and stored in a separate pipeline deque that never feeds the shot fit. The
fitted arc still renders only release→rim.

**Context / alternatives:** (a) separate render-only trail [chosen], (b) open
the main detection gate to always-on and feed the shot manager — rejected:
would shift shot boundaries and move CORE baselines for no correctness gain,
and the floor gate exists precisely to keep low bounces out of fits.

**Trade-off:** the trail reuses detections that already run (~every 3rd frame
post-shot), so near-zero extra CPU; it is purely cosmetic and carries no
analytics meaning yet. `_draw_ball_trail` in overlay.py; deque + `_append_ball_trail`
in pipeline.py.

---

## 2026-07-06 — Early median-snap hoop freeze (kills during-shot drift)

**Decision:** freeze now fires early — accumulate locked boxes through
locked<->revalidation flicker (drop the phase=="locked" gate), and freeze after
EITHER 6 quiet frames OR a 12-frame cap, snapping the lock to the MEDIAN of the
accumulated boxes. Before, flaky far-court clips (IMG_2029*) flickered
locked/revalidation, reset the accumulator, and only froze ~frame 45 — after
the shot, so the anchor visibly drifted while the ball hit the rim.

**Verification:** all 19 clips now freeze by frame <=18 (was up to 57). Tests
140 pass. CORE RMSE unchanged (freeze never touches the fit). Outcome 13/17
(was 14): the median box shifts borderline make/miss geometry (C/L/N/Q makes
now read miss; D/E/G recover). This churn is the symptom the user flagged —
the unlicensed model boxes the hoop inconsistently (full hoop+net vs rim) and
orange rim-refine fails on ~half the clips. **Fix is task #2: train rim-only
weights on the forked CC-BY dataset for one consistent box → stable geometry.**

---

## 2026-07-06 — Posture metrics via YOLO-pose (mediapipe pivoted out)

**Decision:** shooting-form angles (elbow shoulder-elbow-wrist, knee
hip-knee-ankle, back-bend torso-vs-vertical) from `pose/posture.py` (pure,
pose-source-agnostic, unit-tested) fed by `pose/pose_estimator.py`.

**Context / alternatives:** task asked for mediapipe (chosen for no-GPU). But
mediapipe 0.10.35 **segfaults on Python 3.13** (exit 139 on init) and its wheel
dragged in opencv-contrib 5.0, clobbering opencv-python. Pivoted to ultralytics
YOLO-pose: already a core dep, CPU-only (same no-GPU reason), no crash, all 6
joints detected at >=0.97 conf. COCO keypoints remapped to MediaPipe indices so
`posture.py` is unchanged.

**Trade-off:** YOLO-pose is a heavier model than mediapipe's lite pose, but it
runs once per shot at the release frame (standalone `scripts/eval_posture.py`),
NOT per frame in the core pipeline — zero detection/fit impact. Full per-frame
overlay in the main pipeline is a later, flag-gated step. mediapipe left OUT of
deps intentionally; revisit only on a Python where it is stable.

**Note:** opencv-python is now 5.0 / numpy 2.5 (from the mediapipe install
churn); CORE RMSE verified identical, so the bump is safe.

---

## 2026-07-06 — Shot charts with confirm-step court calibration (task #5)

**Decision:** `court/homography.py` (image->court-plane projection from >=4
clicked point pairs) + `court/shot_chart.py` (half-court render, green makes /
red misses). Shooter court position = ankle midpoint from YOLO-pose at the
release frame, projected via the calibration. Calibration is a one-time
per-camera CONFIRM-STEP (`scripts/calibrate_court.py`: click known court points,
enter their feet coords), NOT automatic — auto leg-projection court mapping is
NEX Team patented (US 11594029), so a human confirm step is safer and
licensing-clean. Wiring: `scripts/build_shot_chart.py`.

**Trade-off:** requires a manual calibration per court setup (a few clicks
once). Real charts for the Testing clips await that calibration; the math +
renderer are unit-tested (13 tests) and the demo chart renders correctly.

---

## 2026-07-06 — Lower fit threshold to 4 points (arc on short shots)

**Decision:** `min_measured_points_for_fit` 5 -> 4. Clips I and K had 4 clean
flight points, below the old threshold, so they got no parabola fit and no
colored arc. A quadratic needs only 3 points; 4 gives one DOF of robustness
(clip I fits at r2=1.00, rmse 1.4px). Now every clip with a detected shot
renders a make/miss arc; only F/J (no shot at all) have none.

**Verification:** 156 tests pass (updated the boundary test: 3 pts insufficient,
4 pts fits). Outcome agreement 13/17 unchanged — I/K stay correct miss. CORE
clips have 13-24 points, unaffected by the threshold.

---

## 2026-07-06 — Detector-comparison tooling; trained model wiring staged

**Decision:** trained yolo11n (`cv-cnfd4-eaond-1-yolo11n-t1`) wires in by path
alone — `DetectionConfig.model_path` / `hoop_model_path` already swap models
behind the same `Detector` interface with identical `SparseBallDetection`
output, so no downstream change. Added `scripts/compare_detectors.py`
(baseline vs candidate over A-S: shot rate, measured/shot, completeness, RMSE,
confidence, detection count, fps) and `scripts/wire_and_eval.sh` handoff;
`eval_shot_outcome.py` gained `--ball-model`.

**Context / alternatives:** weight acquisition — (a) roboflow SDK download
[rejected now: risks the opencv breakage mediapipe already caused, no MCP
weight-download tool], (b) local retrain [slower], (c) hosted inference API
[network per frame, detector rewrite]. Chose: user downloads the .pt from the
Roboflow UI (opencv-safe), drops into models/, runs wire_and_eval.sh when
laptop is on.

**Trade-off:** recall / false-positive are count-based proxies until A-S get
per-frame ball ground-truth annotation.

---

## 2026-07-07 — M1 verdict: trained ball class REJECTED at every threshold

**Decision:** the trained yolo11n's ball class is not adoptable at any
confidence. Sweep (candidate ball + frozen baseline hoop, CORE clips):

| conf | A | C | P | R | S |
|---|---|---|---|---|---|
| pins | 0.76 | 0.57 | 1.19 | 0.73 | 1.10 |
| 0.50 | 22.6 | 10.2 | 69.5 | 24.4 | 38.3 |
| 0.70 | 22.5 | 10.2 | 69.0 | 24.4 | 38.3 |
| 0.85 | 5.6 | 10.3 | 66.5 | 24.3 | 42.1 |

RMSE is threshold-INSENSITIVE → the false positives are high-confidence: the
model confidently detects non-balls (or mislocalizes). A threshold cannot fix
confident wrongness; only retraining can (M2: own-court frames, rim-only
labels). Ball detector stays `yolov8n.pt`; hoop stays baseline weights.

**Trade-off:** we forgo the recall win (E 5→63 pts, F/J shots found) until M2.
**Retro:** sweep infra (Opus-built `--ball-conf` isolation knob) is reusable
for every future detector — the experiment cost one script arg. Matches
failure mode #1 (Wholesale Swap) prevention working as designed.

---

## 2026-07-09 — Platform refactor accepted (GPU-optional, CPU-mandatory)

**Decision:** adopt the Phase-A platform architecture on
`feature/platform-refactor`: IR-mediated stages (`src/swishsync/`, coexists
with `swishsync_cv` during migration), plugin registry, and the WHAT/WHERE
split — Detector/Tracker/EventDetector define WHAT runs, an inference backend
(cpu | cuda) defines WHERE. **CPU-only end-to-end stays a hard requirement**;
GPU is a config line (`backend: cuda`), never a rewrite. PRD + North Star land
in docs/PRD.md.

**Amendments accepted with the proposal (Fable review):**
1. *Migration gate fix:* byte-identical CORE output is unreachable by fitting
   raw ball tracks — ~90% of correctness lives in candidate selection
   (gates/lifecycle/floor_idle/rescan), not the fitter. Phase-A step 4 wraps
   the legacy engine WHOLESALE (sparse buffer + ShotCandidateManager +
   finalize) as the first Tracker/EventDetector; stages split later, each
   split re-gated.
2. *No ByteTrack:* multi-object Kalman tracking for one ball adds a dep and
   changes association → breaks byte-identical. First tracker = `single_ball`
   wrapping existing logic; ByteTrack only when multi-player exists.
3. *Backends trimmed to cpu|cuda:* ultralytics takes `device=` directly, so
   CudaBackend hands the device string to the model. triton/http stubs cut
   (YAGNI; re-add when a remote GPU exists).
4. *IR trimmed to shipped scope:* Team/jersey/REFEREE/POSSESSION/REBOUND
   removed (PRD lists them out of scope); SCHEMA_VERSION exists for their
   return.
5. *M2 trains BOTH yolov11n and rf-detr-nano* on the same v2 dataset;
   /detector-bench decides on CORE + CPU fps. RF-DETR is the Apache-2.0
   commercial path but CPU speed is unproven — data decides, not license
   preference.

**Trade-off:** two packages coexist during migration (import-path duality)
until every CORE clip is byte-identical through the platform, then
`swishsync_cv` retires module by module.

---

## 2026-07-09 — MP-B PASS: legacy engine behind platform interfaces, CORE exact

**Result:** all five CORE clips through the platform Pipeline
(`YoloDetector` + `LegacyShotEngine` + `run_platform_clip.py`) match the legacy
runner to **exact float equality** (delta 0.000e+00 on A/C/P/R/S). 167 tests
pass; `git diff src/swishsync_cv` empty; smoke OK. MP-C (stage split + GPU
tier) is unblocked.

**Trade-offs accepted:** engine reads `ShotCandidateManager._cooldown` (private)
for finalize reasons — documented adapter introspection, dissolves at MP-C.
`Tracker.update` ignores its `detections` arg in the wholesale shape (engine
must control stride/ROI detection itself; that's WHY the gate is exact).

**Retro (ritual self-iteration):** the Sonnet build agent stalled mid-sweep —
second stall of this kind — and a rerun died on a stale scratchpad path. Both
now named failure modes (#13 The Stalled Marathon, #14 The Stale Path) with
rules: agents prove ONE clip then stop, main session runs sweeps; always use
the current session's scratchpad, mkdir -p first. Conventions updated:
scoped-gruntwork spawns sanctioned, babysitting banned.

---

## 2026-07-09 — Intake batch labeled; flight-end spec; airball direction

**Ground truth recorded:** 15 clips (T–AI, Y deleted as incomplete), 28
user-confirmed per-shot labels (13 make / 15 miss) in
`run_full_eval_rerun.OUTCOME_GROUND_TRUTH_V2` + `INTAKE_FILES`. Pipeline
detected only 14 of 28 shots — the multi-shot gap is now measured (M8).

**AA hoop-lock diagnosis (frame-verified):** lock chose the correct near hoop
(two hoops in frame) but sits low on the net — dusk lighting defeats the
orange rim refine on this court. Fix routes through M2 training data (these
frames are in the annotation batch), not an ad-hoc threshold tweak.

**User observations feeding M2:** ball detects reliably only against sky/white
background — training-data bias; annotation must prioritize busy-background
(buildings/trees/court) frames. Ball detection weak on all new clips (also
camera quality); several arcs ingested post-rim-bounce points.

**Flight-end spec accepted (reshapes M3):** a shot's flight ENDS at the first
ball-detection ∩ rim-box contact. Arc renders release→contact only. Post-
contact motion is verdict evidence (bounce-up after hoop-box contact ⇒ miss —
already the rescan rule) and render-trail, never fit input. Ball remains
tracked before and after (trail). Fit-affecting boundary change ⇒ CORE gate +
explicit go before default-on.

**Airball direction (user asked; recommendation adopted):** airball stays
classified as `miss` NOW — that output is already correct and honest; no code
change needed. A render-only `miss_subtype` (airball / rim_out / short) lands
AFTER M2 weights make near-rim trajectories trustworthy: geometric test =
fitted descending branch never enters rim x-span AND no rim-zone contact ⇒
airball. Teaching a model this before detection is robust would learn the
detector's blind spots, not basketball.

---

## 2026-07-10 — Verdict evidence rules refined; keys secured; annotation guidance

**Annotation rule (user asked):** annotate EVERY visible instance — both rims,
both people, any ball. Unlabeled instances are label noise (teach the model
"some rims aren't rims"). Rim SELECTION is pipeline logic, never the detector's.

**Verdict rules (user, queued behind annotations/M2):** bounce-up after hoop
contact is LIKELY-miss evidence, not a verdict — batch-2 contains
roll-around-rim-and-in clips that would fool a hard rule. Make-confirmation:
ball falling directly below rim after a predicted make = confirmation weight
(doubles with M5 court projection: court-space landing under the hoop).
M3 spec updated. miss_subtype promoted to milestone M10 (post-M2/M3).

**Keys secured:** audit found NO keys in tracked files or git history (REST
calls kept them in shell only). `.env` created (git-ignored, verified via
`git check-ignore`), `.env.example` committed, uploader auto-reads `.env`
(stdlib loader, no python-dotenv dep). Rule: keys live in `.env` only — never
in repo files, scripts, or docs.

---

## 2026-07-11 — Independent code-reviewer agent (Fable framework, cheaper seats)

**Decision:** Add `.claude/agents/code-reviewer.md` — a review agent that
encodes the Fable-tier review method as an explicit, literal procedure
(scope lanes → invariants gate → intent-vs-implementation → failure-scenario
hunting → evidence grading → quality bars → verdict rules). It runs on Opus
(Sonnet on fallback); the procedure, not the model tier, carries the quality.

**Why:** Fable won't stay free; review quality must survive the downgrade.
Encoding the judgment layer as checkable steps (e.g. "no failure scenario ⇒
not a bug finding", "metric moved without a named mechanism ⇒ finding, not a
win") lets a smaller model reproduce most of the value.

**Independence rule:** the reviewer is never given the coder session's chat,
plan, or rationale — it reviews the diff cold and re-derives intent from diff
+ repo docs. Same-context review inherits the coder's blindspots; the repo's
own history (RMSE "improvement" via truncated arcs, silent baseline re-pins)
is exactly the class of thing a cold reader catches and a warm one rationalizes.

**Alternatives rejected:**
- Reviewing in the coder's session: cheapest, but biased — the reviewer would
  read the coder's own justifications as evidence.
- /code-review skill only: good generic bug hunt, but doesn't enforce repo
  invariants (render/fit boundary, baseline immutability, two-package drift)
  or the block/approve decision rules.

**Wiring:** CLAUDE.md model-lane + production-guardrail sections updated —
reviewer verdict rides with every milestone/PR summary; BLOCK stops the ship
until addressed or user-overridden.

---

## 2026-07-11 — v2probe (yolo11n, partial own-annotations) bench: NOT adopted; training direction VALIDATED

**Decision:** `models/hoop_ball_v2probe.pt` is not wired as the active detector,
but the M2 retrain direction it probes is validated on both classes. Continue
annotating the remaining own-clip frames and proceed to the YOLO26 Nano
full-dataset train.

**Hoop class (isolated): near-adoptable.** 19/19 auto-locks at frame 3
(bar ≤ 18); the two documented dx outliers are fixed (B −68→−5.9, C +59→+9.6).
Tight ring-only boxes confirmed visually (clip C frame 15: ring box vs v8's
hoop+net box). Two consequences block immediate adoption:
1. Convention shift: `rim_center_y = bbox_xyxy[3]` now lands ~50–90px higher
   (ring bottom, not net bottom). Every rim-relative threshold (shot-start,
   floor cutoff rim+100px, rim anchor, outcome zones) was tuned to the old
   convention → CORE start-frames move later and RMSE pins shift (all 5 CORE
   clips drift, in the *lower* direction). Adoption requires a deliberate
   re-pin with this entry as the named cause, not a silent one.
2. Remaining dx misses vs the ≤15px bar: F +40.9, H +24.5, G +16.7 — all
   2027-series clips.

**Ball class (isolated, baseline hoop): major recall win, blocked by
segmentation, not by the model.** Measured points improved on 19/19 clips
(+10…+61; far-court E 5→63, H 7→68, I 4→48, K 4→51 — target was 1.5×, got
8–12×). Both accepted-failure clips now track (F 28 pts, J 22 pts, J fits at
RMSE 6.06). RMSE blowups (15–70) are threshold-insensitive (conf sweep
0.5/0.7/0.85 barely moves them) because the detections are REAL: the model
keeps tracking the ball through rim contact and the rebound (verified on P:
flight f92–117 identical to baseline at conf 0.85–0.9, then rebound f118–149
merges a second arc into the fit). Verdict flips A make→miss, B miss→make ride
that mechanism. The old detector's rim-contact dropout was doing flight-end
segmentation by accident; a strong detector needs the real flight-end logic
(spec already recorded 2026-07-09) BEFORE any strong ball model can be adopted.

**Annotation guidance for the remaining frames (in value order):**
1. 2027-series court/angle frames — the only remaining rim dx failures (F/G/H).
2. Rim-contact and occlusion moments (ball on/inside ring, hands near ball) —
   this is where flight-end logic will need detector reliability.
3. New shooters/angles per filming_spec — detection generalization.
Ball-in-clean-flight frames are the LOWEST priority: recall there is already
saturated. Per 2026-07-10 rule: annotate every visible instance.

**YOLO26 gate:** unchanged from M2 — verify ultralytics loads a YOLO26 .pt
end-to-end (CPU) before committing the full train to it; per-class isolation
bench + this same adoption ceremony apply to the full-dataset weights.
