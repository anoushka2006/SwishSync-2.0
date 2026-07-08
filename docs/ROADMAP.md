# SwishSync roadmap — milestone-based, model-routable

Each milestone: goal → execution steps (with model lane) → success metrics
(checkable, not adjectives) → rollback. A milestone ships only when its metrics
pass AND the standing quality bars in CLAUDE.md pass. Ship ritual: `/milestone`.

Status legend: [ ] not started · [~] in progress · [x] done

---

## M1 — Salvage the trained ball detector (precision sweep)  [ ]

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

---

## M2 — Rim-only retrain v2 (own frames + rim-only labels)  [ ]

**Goal:** one consistent tight rim box on all 19 clips (ends the full-hoop-vs-
back-rim inconsistency the user flagged) and a ball class trained on OUR courts.

**Steps**
1. (User) Annotate the 419 frames in `datasets/own_clips/images/` in Roboflow
   (`basketball-strategy/cv-cnfd4-eaond`): classes `basketball`, `rim`
   (ring ONLY — never net/backboard). ~2–3h manual.
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

**Rollback:** keep current weights; own frames remain annotated (sunk cost 0).

---

## M3 — Rim-bounce make/miss (C, L, N, Q)  [ ]  ← gated on M1 or M2

**Goal:** rattle-in makes read as makes. Requires the ball to be *seen* during
the rattle — that's why this is gated on a usable ball-at-rim detector.

**Steps**
1. (Sonnet) Feed rim-rescan from the adopted detector at the chosen conf.
2. (Opus) Re-tune rescan rules on the four clips (depth ratio, rattle window)
   against `OUTCOME_GROUND_TRUTH`; every change re-runs the full 19.
3. (Sonnet) Unit tests for each rule change.

**Success metrics**
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

## Parked (revisit when triggered)

- **WASB heatmap detection** — only if M1+M2 still miss far-court balls.
- **Per-frame ball ground truth** on A–S — unlocks true recall/FP metrics
  (today: count proxies). Trigger: first detector decision that the proxies
  can't settle.
- **Entry-angle coaching view** (Noah-style arc feedback) — product feature;
  compute exists (`entry_angle_deg`), needs UI design. Patent check first
  (real-time audible feedback is Pillar Vision territory).
- **mediapipe** — banned on py3.13 (segfault). Revisit only on a Python where
  its wheels are stable, and only if YOLO-pose proves insufficient.
