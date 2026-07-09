# SwishSync 2.0 — Product Requirements Document & North Star

Status: current product. Scope is deliberately unchanged from the existing
project vision. The platform refactor (`feature/platform-refactor`) is an
*architecture* change underneath this product, not a scope change.

---

## North Star

> The most rigorous, honest, offline analyzer for a single basketball shot's
> flight — turning one shooting-workout video into a trustworthy fitted
> trajectory, confidence, and shot story, with zero misleading output.

Everything is measured against one question: **does the analyzer tell the truth
about the shot?** A clean fit on a clear shot, an honest "no shot" on a clip the
detector can't see, and never a plausible-looking arc that isn't real. Truthful
beats impressive.

**Explicit non-goal / separate future project:** the "Basketball Intelligence
Engine" (multi-player tracking, possessions, events, team analytics, full-game
understanding) is a distinct project with a different architecture, data, and
camera assumptions. It is out of scope for this PRD. The platform refactor keeps
the door open for it without committing to it.

---

## Problem

A shooter or coach films a workout on a phone. They want objective feedback on
shot trajectory — arc shape, apex, consistency — without a GPU rig, a subscription
product, or manual frame-by-frame review. Existing consumer tools (HomeCourt) are
closed and phone-locked; nothing offers an open, offline, inspectable analyzer.

## Users

- **Primary:** the developer/researcher iterating the pipeline (benchmark-driven).
- **Secondary:** a coach or shooter running the CLI on their own clips offline.
- Not yet: real-time users, live-game users, multi-player workflows.

## Product scope (in)

- Single-shot trajectory mapping from an uploaded video.
- Manual and automatic hoop lock.
- Ball detection, shot lifecycle, robust weighted parabola fit.
- Confidence scoring (detection + trajectory).
- Render-only shot story: pickup, continuity, gap bridges, arc, release marker.
- Make/miss verdict (render-only, side-view geometry).
- Dual-pane processed video + JSON/CSV diagnostics + eval browser.
- Benchmark suite (CORE / STRESS / FAILURE, clips A–S).

## Out of scope

- Multi-player detection, tracking, re-ID, team classification.
- Possession / pass / rebound / screen / any multi-object event inference.
- Full-court occupancy, spacing, passing networks, defensive analytics.
- Pose/biomechanics beyond existing release metrics.
- Real-time / streaming. Processing is offline batch.
- True perspective/court-space normalization (right panel stays camera-space
  until explicitly tackled).

---

## Requirements

### Functional

1. Accept a single-shot video and produce a processed dual-pane video plus
   `shots.json` and `detections.csv`.
2. Lock the hoop (manual 4-point or automatic via detector weights) before
   evaluating shot start.
3. Detect the shot lifecycle and fit exactly one parabola per shot.
4. Score confidence (40% detection, 60% trajectory).
5. Render render-only story layers without ever affecting the fit, RMSE,
   confidence, or lifecycle.
6. On clips where the ball can't be seen, produce no shot and no misleading
   trajectory — an honest failure.
7. Emit make/miss as a render-only, clearly-provisional verdict.

### Non-functional

1. **CPU-only must always work.** The pipeline must run end-to-end on a device
   with no GPU. This is a hard requirement, not a fallback afterthought.
2. **GPU is an optional accelerator.** GPU inference is selectable by config
   (`backend: cuda`/`triton`/`http`) with identical outputs where the same model
   runs; it must never become a requirement to run the product.
3. **Model-agnostic.** Detector and tracker are swappable by config with no code
   change; a better detector next year is a config line, not a rewrite.
4. **Benchmark-gated.** Every change reruns the suite; CORE clips must not
   regress (`|ΔRMSE| ≤ 0.15`, wrong-verdict count must not increase).
5. **Honest by construction.** No output implies more certainty than the data
   supports; every event carries its evidence.
6. **Licence-clean commercial path.** Prefer permissively-licensed models
   (RF-DETR Apache-2.0) over AGPL where a commercial future is in view.

---

## Recommended stack (current product)

- Python 3.11+, numpy, OpenCV, plain-dataclass IR (no pydantic dep — keeps
  CPU-only imports cheap).
- Detector: RF-DETR (Apache-2.0) as the commercial-safe default; YOLO11 kept as
  a baseline plugin (AGPL — research ceiling).
- Tracker: not required for single-shot today; ByteTrack is the default when the
  ball needs multi-frame association.
- Inference backends: `cpu` (mandatory), `cuda` (optional), `triton`/`http`
  (later, for remote GPU) — all behind one interface.
- Data/training: Roboflow (annotation + custom detector training).
- Offline labelling only: Florence-2 / GroundingDINO (permissive) to draft
  labels; NVIDIA LocateAnything-3B is research-licence, so offline/research use
  only and never in a commercial build.

---

## Success metrics

- **Correctness:** CORE clips fit with `weighted_residual_rmse` within pinned
  baselines; `|ΔRMSE| ≤ 0.15` on every change.
- **Honesty:** FAILURE clips (F, J today) produce zero shots and zero misleading
  right-panel paths. 100% of "no-ball" clips fail cleanly.
- **Fit coverage:** every shot with ≥ 4 flight points has a non-null
  `parabola_fit`.
- **Verdict quality:** wrong make/miss verdict count never increases release over
  release; `unknown` is preferred to a wrong verdict.
- **Portability:** full suite runs on a CPU-only machine with no GPU present.
- **Swap cost:** changing the detector is a single config edit; proven by running
  ≥ 2 detectors through the same pipeline unchanged.

## Anti-metrics (things we refuse to optimise)

- Demo polish that isn't backed by a truthful fit.
- CORE regressions traded for STRESS gains.
- Verdict agreement bought by re-pinning baselines without a decision-log entry.

---

## Milestone alignment

This PRD is served by the existing roadmap: the platform refactor is the
enabling architecture (`feature/platform-refactor`), while product correctness
continues through the detector/verdict milestones (M2 rim retrain, M3 rim-bounce
make/miss, M7 trusted-flight fit swap). None of those change product scope; they
harden the single-shot analyzer this North Star defines.
