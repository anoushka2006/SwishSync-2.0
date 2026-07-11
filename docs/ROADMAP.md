# SwishSync roadmap — milestone-based, model-routable

Each milestone: goal → execution steps (with model lane) → success metrics
(checkable, not adjectives) → rollback. A milestone ships only when its metrics
pass AND the standing quality bars in CLAUDE.md pass. Ship ritual: `/milestone`.
Product definition + North Star: [docs/PRD.md](PRD.md).

Status legend: [ ] not started · [~] in progress · [x] done

---

# Platform track (branch `feature/platform-refactor`)

Architecture change under the same product (see DECISION_LOG 2026-07-09):
model-agnostic stages + GPU-optional backends, **CPU-only always runs
end-to-end**. Runs in parallel with the product track below; merges only
through PR + your "go".

## MP-A — Phase A scaffold  [x]

IR schemas, registry, backends (cpu|cuda), interfaces, ShotEventDetector
adapter over the legacy fit, smoke test. Applied from the reviewed patch, then
amended: IR trimmed to shipped scope, backends trimmed to cpu|cuda,
ByteTrack→`single_ball`, migration step 4 rewritten to wholesale-engine wrap.

**Success metrics:** `smoke_platform.py` prints SMOKE OK; platform tests pass
alongside the existing suite; legacy pipeline byte-untouched (CORE gate green).

## MP-B — Legacy engine behind platform interfaces + CORE-clip gate  [x]

(Sonnet builds, Opus judges) `swishsync.vision.detection.yolo` wraps the
existing YoloObjectDetector behind `Detector` (backend supplies device);
`single_ball` Tracker + `legacy_shot` EventDetector wrap SparseBallDetection
buffer + ShotCandidateManager + finalize_shot wholesale. New
`scripts/run_platform_clip.py` routes CORE clips through the platform Pipeline.

**Success metrics (exit gate):**
- Clip C through platform Pipeline: `weighted_residual_rmse` identical to the
  legacy runner to 1e-9 (byte-identical goal; hard fail above 0.15).
- All 5 CORE clips within the same gate before MP-B closes.
- Zero changes inside `swishsync_cv` (adapter-only; `git diff src/swishsync_cv`
  empty).

## MP-C — Stage split + GPU tier  [ ]  ← UNBLOCKED (MP-B passed 2026-07-09)

Split the wholesale wrap into true detect→track→event stages one seam at a
time, re-running the MP-B gate after each split. Then wire `backend: cuda`
through the detector (ultralytics `device=`) and prove the GPU tier on a CUDA
machine (or defer proof until one exists — config lands either way, CPU output
unchanged).

**Success metrics:** every split lands with the CORE gate green; cpu/gpu
configs differ by the backend line only; CPU-only machine runs the full suite.

---

# Product track (branch `feature/ball-tracking`)

---

## M1 — Salvage the trained ball detector (precision sweep)  [x] → REJECT

**Goal:** keep the trained model's recall (63 pts on E vs 5 baseline; F/J shots
found) without the false positives that blew CORE RMSE to 15–70px.

**Why first:** cheapest experiment with the highest information value — decides
whether M2 needs a retrain or just a threshold.

**Steps**
1. (Sonnet) Add `--ball-conf` to `scripts/run_detector_benchmark.py` trained
   config; sweep ball-class confidence 0.50 / 0.70 / 0.85 on CORE clips only.
2. (Opus) Judge sweep: pick lowest conf where CORE passes the drift gate.
3. (Sonnet) Full 19-clip benchmark at the chosen conf.
4. (Opus) Adopt/reject decision → DECISION_LOG entry either way.

**Success metrics**
- CORE: all 5 clips |ΔRMSE| ≤ 0.15 vs `hoop_ball_yolov8_baseline.pt` pins.
- Recall kept: measured points on E, H, I, K ≥ 1.5× their baseline counts
  (baseline: E 5, H 7, I 4, K 4).
- F and J: shot detected with a parabola fit.
- Make/miss: wrong-verdict count ≤ baseline's 4 (C, L, N, Q).

**Rollback:** trained ball rejected → ball stays `yolov8n.pt`; revisit after M2.

**Outcome (2026-07-07): REJECT.** RMSE threshold-insensitive (P ~67-69px at
every conf 0.5–0.85) → high-confidence false positives; retraining is the only
fix. Sweep table + reasoning in DECISION_LOG. The `--ball-conf` isolation knob
in `run_detector_benchmark.py` is reusable for every future candidate.

---

## M2 — Rim-only retrain v2 (own frames + rim-only labels)  [ ]

**Goal:** one consistent tight rim box on all 19 clips (ends the full-hoop-vs-
back-rim inconsistency the user flagged) and a ball class trained on OUR courts.

**Steps**
1. (User) Annotate the 419 frames in `datasets/own_clips/images/` in Roboflow
   (`basketball-strategy/cv-cnfd4-eaond`): classes `basketball`, `rim`
   (ring ONLY — never net/backboard). ~2–3h manual.
   Training-data policy (see docs/filming_spec.md): different shooters and
   different camera angles are WANTED in training data from this batch onward —
   detection must generalize. (Verdict geometry stays side-view; that's M-parked
   "angle-aware geometry", not a training constraint.)
   **Dual-arch (updated 2026-07-10):** full-dataset run trains YOLO26 Nano
   (Ultralytics successor: NMS-free, faster CPU inference — supersedes yolov11
   in-family) and RF-DETR Small (Apache-2.0, "needs less data", non-ultralytics
   so requires an adapter); /detector-bench decides on CORE + CPU fps. The
   2026-07-10 PROBE run stays yolov11n deliberately: same arch as the failed
   v1 model isolates the data variable (failure mode #1). Verify installed
   ultralytics loads YOLO26 .pt before committing M2 to it.
2. (Haiku) Verify class balance + split via MCP `projects_get`.
3. (Opus, money gate — ask user) `versions_generate` v2 → `trainings_create`
   yolov11n. Roboflow credits spent only on explicit "go".
4. (User) Download weights from Roboflow UI → `models/hoop_ball_v2.pt`
   (SDK download stays banned: opencv clobber risk).
5. (Sonnet) `python scripts/run_detector_benchmark.py` vs frozen baseline.
6. (Opus) Adopt/reject + DECISION_LOG.

**Success metrics**
- Hoop: 19/19 auto-lock by frame ≤ 18; |dx_center| ≤ 15px on every clip
  (today: B −68, C +59 are outliers).
- `rim_bbox_xyxy` (orange refine) present on 19/19 locks (today: ~10/19).
- CORE drift gate passes with v2 as hoop.
- Ball at rim: ≥ 1 detection ≥ 0.35 conf inside the rim crop during ball
  arrival on D, E, G (today: 0 — the source of their `unknown`s).
- Busy-background ball detection (2026-07-09 batch): measured points ≥ 8 on
  T/U/AC/AD/AE (today ball tracks mostly against sky only — training bias).
- AA-class hoop lock: dusk/two-hoop clip AA locks ON the ring (frame-verified),
  not the net.

**Rollback:** keep current weights; own frames remain annotated (sunk cost 0).

**Probe outcome (2026-07-11, `hoop_ball_v2probe.pt`, yolov11n on partial
annotations):** direction VALIDATED, weights not adopted — full read-out in
DECISION_LOG 2026-07-11. Hoop: 19/19 locks at frame 3, B/C dx outliers fixed;
remaining dx misses F/G/H (2027 series). Ball: recall up on 19/19 clips
(F and J now track); RMSE blowups are the missing flight-end segmentation, so
flight-end logic is a HARD PREREQUISITE for adopting any strong ball model.
Annotation priority for the remaining frames: 2027-series court/angle, then
rim-contact/occlusion moments, then new shooters/angles; clean-flight ball
frames are saturated.

---

## M3 — Rim-contact flight end + rim-bounce verdicts  [ ]  ← gated on M2

**Spec (user, 2026-07-09; refined 2026-07-10):** flight ENDS at first
ball∩rim-box contact; arc renders release→contact; ball stays tracked
before/after (trail). Post-contact motion is verdict EVIDENCE, weighed not
decisive:
- bounce-up after hoop contact ⇒ LIKELY miss — overridable by later
  below-ring-inside evidence (batch-2 has roll-around-rim-and-in clips that
  would fool a hard rule).
- make-confirmation: after a predicted make, ball tracked falling DIRECTLY
  below the rim ⇒ confirmation signal (already the rescan below-ring rule;
  formalize as confirmation weight). Doubles with M5 court projection later:
  landing position in court space should sit under the hoop.
Fit-affecting boundary change ⇒ CORE gate + explicit go.

**Goal:** rattle-in makes read as makes. Requires the ball to be *seen* during
the rattle — that's why this is gated on a usable ball-at-rim detector.

**Steps**
1. (Sonnet) Feed rim-rescan from the adopted detector at the chosen conf.
2. (Opus) Re-tune rescan rules on the four clips (depth ratio, rattle window)
   against `OUTCOME_GROUND_TRUTH`; every change re-runs the full 19.
3. (Sonnet) Unit tests for each rule change.

**Success metrics**
- Arc never extends past first rim contact on any labeled clip (frame check).
- C, L, N, Q verdict == make.
- Zero previously-correct clips flip wrong (wrong-verdict count strictly
  decreases).
- Agreement ≥ 16/17 (or /19 once F/J detect).

---

## M4 — F/J shot recovery  [ ]  ← gated on M1/M2 adoption

**Goal:** the two FAILURE clips detect their shots (trained model already
proved the balls are findable).

**Success metrics**
- F: shot with fit + verdict == make. J: shot with fit + verdict == miss.
- Zero regression on the other 17 (same fits, same verdicts).
- CORE gate passes.

---

## M5 — Real shot charts  [ ]  ← needs user calibration, one-time

**Steps**
1. (User) `python scripts/calibrate_court.py --input <clip> --out
   calib/<court>.json` once per court (3 courts in Testing set).
2. (Sonnet) `build_shot_chart.py` across clips; add chart PNG to eval browser
   per court.

**Success metrics**
- Chart renders for every calibrated court; each charted shot's court position
  within the court bounds; make/miss colors match verdicts.

---

## M6 — Merge feature/pose-overlay  [ ]  ← awaiting user review

**Steps:** user reviews PR diff → plain-English summary provided → "go" →
merge. (AI never merges itself.)

**Success metrics**
- Flag off: output video byte-identical to pre-merge (checksum one clip).
- Flag on: skeleton + 3 angles render on release frame of 19/19 shot clips.
- 156+ tests pass.

---

## M7 — Trusted-flight Phase 2 (fit input swap)  [ ]  ← long-standing gate

**Goal:** the fitter consumes `trusted_flight` selection instead of the ad-hoc
cluster. Fit-affecting: highest ceremony.

**Preconditions (already proven):** zero divergence on CORE; divergence only on
M/O (by design).

**Steps**
1. (Fable — architecture) Wiring design + migration plan per
   `docs/trusted_flight_points_spec.md`.
2. (Sonnet) Implement behind a config flag, default off.
3. (Opus) Flag-on eval: byte-identical fits on zero-divergence clips; M/O
   changes reviewed shot-by-shot.
4. (User) "go" to flip the default; re-pin baselines with DECISION_LOG entry.

**Success metrics**
- Flag off: all artifacts byte-identical.
- Flag on: CORE fits identical; M/O RMSE improves (both currently > 9px).

---

## M8 — New-clip intake & multi-shot benchmark  [~]  ← labels recorded 2026-07-09

Batch 1 done through intake: 15 clips (T–AI; Y deleted), 28 per-shot labels in
OUTCOME_GROUND_TRUTH_V2, zero crashes, 328 training frames extracted, labeling
page shipped (scripts/build_intake_browser.py). Detected 14/28 shots → add:
**multi-shot detection recovery** — find why shots after the first don't start
(cooldown/idle gating/sparse stride between shots), fix behind the CORE gate,
score with per-shot eval.

**Goal:** absorb the new filming batch (docs/filming_spec.md) into the
benchmark, including multi-shot workout clips with per-shot ground truth.

**Steps**
1. (User) Film per the spec; note ordered outcomes per clip
   (`clip_T: make, miss, make, ...`). Drop files in `videos/`.
2. (Haiku) Inventory new clips; extend `CLIP_LABELS` with new letters/slugs.
3. (Sonnet) Ground-truth schema v2: per-shot ordered labels
   (`OUTCOME_GROUND_TRUTH_V2: dict[clip, list[verdict]]`), eval scores ALL
   finalized shots per clip matched to labels by order — not just the primary.
4. (Sonnet) Frame extraction for training doubles automatically
   (`extract_training_frames.py` over new clips).
5. (Opus) Categorize new clips CORE/STRESS/FAILURE; side-view single/multi-shot
   clean clips are CORE candidates; new angles enter as STRESS only.

**Success metrics**
- Every new clip runs through the pipeline without crash.
- Eval reports per-shot agreement: `sum(correct shots)/sum(labeled shots)`
  across all clips (multi-shot counted shot-by-shot).
- Shot-count accuracy: detected shot count == labeled count on ≥ 80% of
  multi-shot clips (misses of count are their own failure row).
- New-angle clips: tracked + arcs rendered (verdict exempt until angle-aware
  geometry ships).

## M9 — Workout session analytics  [ ]  ← gated on M8

**Goal:** the "full shooting workout" product loop: one multi-shot video in →
session stats out.

**Steps**
1. (Sonnet) Session summary from `finalized_shots`: attempts, makes, FG%,
   per-shot entry angle, streaks; serialize `session.json` next to shots.json.
2. (Sonnet) Eval-browser session view: per-clip shot list with verdicts +
   the session stat line; shot chart per session once M5 calibration exists.
3. (Opus) Judge against user-labeled workout clips from M8.

**Success metrics**
- `session.json` per clip: attempts == labeled shot count, makes == labeled
  makes on clips where per-shot verdicts are all correct.
- Browser shows the session line for every multi-shot clip.
- FG% correct wherever the per-shot verdicts are correct (pure arithmetic —
  any mismatch is a bug, not a model limit).

## M10 — miss_subtype classification  [ ]  ← gated on M2 + M3 (user-approved 2026-07-10)

Render-only `miss_subtype` on missed shots: `airball` (fitted descending branch
never enters rim x-span AND no rim-zone contact), `rim_out` (rim contact then
outside emergence), `short` (arc apex/reach short of rim plane). Pure geometry,
no model training. Gated on M2 weights (trustworthy near-rim trajectories) so
the classifier learns basketball, not the detector's blind spots.

**Success metrics:** every labeled miss in the batches gets a subtype; zero
make/miss verdict changes (evidence-only field); subtype agreement reviewed by
user on the labeled misses before the field leaves "provisional".

## Parked (revisit when triggered)

- **Angle-aware verdict geometry** — make/miss for non-side camera angles
  (rim-ellipse crossing instead of x-span). Trigger: M8 delivers new-angle
  clips and their verdicts matter to the user.
- **Multi-person shooter selection** — pose currently takes the top-confidence
  person; workout clips with bystanders may need shooter tracking. Trigger:
  first M8 clip where the wrong person gets the skeleton.

- **WASB heatmap detection** — only if M1+M2 still miss far-court balls.
- **Per-frame ball ground truth** on A–S — unlocks true recall/FP metrics
  (today: count proxies). Trigger: first detector decision that the proxies
  can't settle.
- **Entry-angle coaching view** (Noah-style arc feedback) — product feature;
  compute exists (`entry_angle_deg`), needs UI design. Patent check first
  (real-time audible feedback is Pillar Vision territory).
- **mediapipe** — banned on py3.13 (segfault). Revisit only on a Python where
  its wheels are stable, and only if YOLO-pose proves insufficient.
