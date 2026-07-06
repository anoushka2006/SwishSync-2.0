# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Tests (PYTHONPATH is configured in pyproject.toml — no prefix needed)
pytest                                    # all tests
pytest tests/test_trusted_flight.py       # single file
pytest tests/test_shot_story.py -k "pickup"  # filter by name
PYTHONPATH=src pytest tests/              # explicit override if needed

# Install (editable + dev deps)
pip install -e ".[dev]"

# CLI — run the full pipeline on a video
swishsync-cv --input videos/clip.mp4 --output-dir outputs/temp/my_run
swishsync-cv --input videos/clip.mp4 --select-hoop-on-first-frame --output-dir outputs/temp/my_run
swishsync-cv --input videos/clip.mp4 --hoop-model models/hoop_ball_yolov8.pt --output-dir outputs/temp/my_run  # automatic hoop lock

# Eval tools
python scripts/evaluate_testing_clips.py --report
python scripts/run_full_eval_rerun.py
python scripts/run_full_eval_rerun.py --only C P R
python scripts/eval_trusted_flight.py outputs/temp/my_run/shots_finalized.json
python scripts/verify_core_drift.py
```

## Architecture

### Real data flow (README is outdated)

```
video frames
  → YoloObjectDetector            detection/yolo.py
  → filter_basketball_detections  detection/ball_detection_gates.py  (floor-band rejection)
  → SparseBallDetectionBuffer     tracking/sparse_detection.py       (stride + ROI search → SparseBallDetection)
  → HoopLockTracker               tracking/hoop_lock.py              (locks rim position as HoopLock)
  → ShotCandidateManager.update() tracking/shot_candidate.py         (state machine: idle → collecting → finalized)
      → finalize_shot()           tracking/shot_finalization.py      (fits one parabola per shot)
  → render_debug_panel            visualization/overlay.py           (left pane)
  → render_trajectory_panel       visualization/trajectory_panel.py  (right pane)
  → compose_dual_pane             visualization/dual_pane.py
  → VideoWriter                   io/video.py
```

### Core data contracts (`data.py`)

| Type | Role |
|------|------|
| `SparseBallDetection` | One detected ball center: `(frame_index, timestamp_ms, x, y, confidence, interpolated, source)` |
| `ShotCandidate` | Mutable accumulator during collection; holds all point buffers and finalization results |
| `ParabolaFit` | Fitted quadratic `y = ax² + bx + c` with apex, x-range, R², weighted RMSE |
| `FitDiagnostics` | Per-point residuals, weights, outlier flags for every point in the fit |
| `HoopLock` | Locked rim position; `rim_center_y = bbox_xyxy[3]` (bottom of hoop bbox) |
| `ShotStoryMetadata` | **Render-only** phase boundaries (pickup/flight/post-shot frames) |
| `TrustedFlightSelection` | Pre-fit point selection (Phase 0 observability; not yet the fit input) |
| `ConfidenceScores` | Composite score: 40% detection + 60% trajectory |

`effective_point_source(point)` resolves point provenance, handling the legacy `interpolated` flag before the `source` field existed.

### ShotCandidate point buffers

`ShotCandidate` carries four distinct point lists — understanding which is which matters:

| Field | Contents | Used by |
|-------|----------|---------|
| `candidate_points` | Measured detections during collection (no gap-predicted) | Parabola fit input (via `select_contiguous_flight_cluster` + `select_flight_fit_points`) |
| `continuity_points` | Measured + gap-predicted — full visualizable path | Visualization only |
| `pickup_points` | Measured points before release | Visualization only |
| `post_shot_points` | Measured points after flight ends | Visualization only |
| `trusted_flight_debug` | Pre-fit selection (Phase 0); see below | Observability only |

### Finalization pipeline (`shot_finalization.py`)

`finalize_shot()` runs once when collection ends:
1. `select_contiguous_flight_cluster(candidate_points, max_gap=15)` — primary flight cluster
2. `select_flight_fit_points(cluster, hoop_lock)` — remove floor bounces (y > rim + 100px)
3. `fit_weighted_parabola_robust(fit_points)` — confidence-weighted polyfit + one outlier refit
4. `_select_best_fit(unanchored, anchored)` — optionally use rim anchor if it lowers RMSE
5. `score_shot_confidence(candidate)` — 40% detection + 60% trajectory
6. `compute_arc_render_metadata(...)` — visual extension range (render-only)
7. `select_trusted_flight_points(candidate_points, ...)` → `candidate.trusted_flight_debug`
8. `classify_shot_outcome(...)` → `candidate.outcome` (render-only make/miss; `tracking/shot_outcome.py`)
9. `compute_shot_story(...)` → `candidate.story` (render-only phase boundaries)

### Trusted flight points (Phase 0)

`tracking/trusted_flight.py` computes `TrustedFlightSelection` via three stages:
1. Measured-only (drop `gap_predicted`)
2. Contiguous cluster split at `config.trusted_flight.max_flight_gap_frames` (default **10**, tighter than reacquisition's 15)
3. Floor bounce exclusion

Stored as `ShotCandidate.trusted_flight_debug`. **Not read by the fitter in this phase.** Phase 2 gates wiring it into the fit on zero divergence from `scripts/eval_trusted_flight.py`. See `docs/trusted_flight_points_spec.md`.

### Hard invariants

- **Render-only layers must not touch `candidate_points`, fit coefficients, RMSE, or confidence.** `ShotStoryMetadata`, `ArcRenderMetadata`, `TrustedFlightSelection`, pickup/post-shot points are strictly observability/render.
- `finalize_shot()` runs exactly once per shot, never during collection.
- `gap_predicted` points live in `continuity_points` for visualization; they are not in `candidate_points`.
- `effective_point_source()` must be used instead of reading `.source` directly when resolving provenance (legacy records use only `interpolated=True`).

### Config hierarchy

`PipelineConfig` owns all sub-configs:
- `ShotCandidateConfig` — shot lifecycle thresholds; nests `GapRecoveryConfig` and `TrustedFlightConfig`
- `HoopLockConfig` — rim acquisition and anchor weight
- `SparseDetectionConfig` — stride, ROI search, floor-band rejection
- `VideoOutputConfig` — codec, dual-pane, debug frames; nests `ShotStoryConfig`

### Benchmark and eval

19 test clips (A–S) in `videos/Testing/`. Three categories:
- **CORE** (A, B, C, P, R, S) — must not regress; RMSE baselines in `scripts/run_full_eval_rerun.py`
- **STRESS** (D, E, G, H, I, K, L, M, N, O, Q) — track improvements, not strict gates
- **FAILURE** (F, J) — accepted zero-shot clips; pipeline must not crash

Canonical eval artifacts live in `outputs/eval/shot_story/`. See `docs/benchmark_suite.md` for per-clip expectations and pass/fail criteria.

## Model lane (work routing)

Work autoroutes to the right model tier. Route silently; don't announce routine routing.

| Lane | Model | Use for |
|------|-------|---------|
| **Down** | Haiku 4.5 reads, Sonnet 5 builds | Codebase reading/search and routine implementation. Silent, automatic. |
| **Seat** | Opus 4.8 | Planning, judging, reviewing. The default seat. |
| **UP** | Fable 5 | Design, danger, money **only**: architecture, production debugging, security review, migrations. Never routine coding. ~20% of work max. |

- **Fallback:** if a model is down/unavailable, drop one tier (Fable → Opus → Sonnet → Haiku) and say so. For anything risky, stop before substituting.

## Production guardrails

- Everything production-bound is guarded: branch → PR → automated checks → merge → deploy → verify.
- The AI never merges itself. It presents a plain-English summary derived from the **actual code diff** and waits for an explicit "go".
- If in doubt or underspecified: ask, don't assume.

### Branch workflow

- `main` — stable/demo-ready only
- `dev` — active integration
- `feature/*` — isolated experiments (current: `feature/ball-tracking`)

`videos/`, `outputs/`, `models/`, `notebooks/` are git-ignored local workspace folders.

## Decisions & Trade-offs

Engineering decisions and their trade-offs are logged reverse-chronologically in
[docs/DECISION_LOG.md](docs/DECISION_LOG.md). Add an entry there when a choice
would confuse a future session if left unexplained. Automations: see
[docs/automations.md](docs/automations.md). Own-weights training: see
[docs/training_plan.md](docs/training_plan.md).
