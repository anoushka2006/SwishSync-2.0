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

## Platform refactor (branch `feature/platform-refactor`)

Architecture migration in progress — product scope unchanged (PRD + North Star:
[docs/PRD.md](docs/PRD.md)). `src/swishsync/` is the model-agnostic platform
(IR schemas, plugin registry, Detector/Tracker/EventDetector interfaces,
cpu|cuda backends). It **coexists** with `swishsync_cv` until every CORE clip
is byte-identical through the platform Pipeline; then the legacy package
retires module by module.

Rules that govern the migration (details: DECISION_LOG 2026-07-09, roadmap
MP-A/B/C):
- **CPU-only must always run end-to-end.** GPU is `backend: cuda` in config —
  an accelerator, never a requirement.
- **WHAT vs WHERE:** stages define what a model computes; backends define
  where it runs. Detector swaps are config lines.
- **Wholesale first:** the legacy engine (sparse buffer + ShotCandidateManager
  + finalize) wraps behind platform interfaces as one unit; stages split only
  after the CORE byte-identical gate passes, one seam at a time, re-gated each
  split.
- Adapter work never edits `swishsync_cv` internals (`git diff src/swishsync_cv`
  stays empty on the refactor branch until the port begins).

## Model lane (work routing)

Work autoroutes to the right model tier. Route silently; don't announce routine routing.

| Lane | Model | Use for |
|------|-------|---------|
| **Down** | Haiku 4.5 reads, Sonnet 5 builds | Codebase reading/search and routine implementation. Silent, automatic. |
| **Seat** | Opus 4.8 | Planning, judging, reviewing. The default seat. |
| **UP** | Fable 5 | Design, danger, money **only**: architecture, production debugging, security review, migrations. Never routine coding. ~20% of work max. |

- **Fallback:** if a model is down/unavailable, drop one tier (Fable → Opus → Sonnet → Haiku) and say so. For anything risky, stop before substituting.
- **Review seat is an agent, not a model:** `.claude/agents/code-reviewer.md`
  encodes the review framework (Fable-derived, written as an explicit
  procedure) so it runs on Opus today and Sonnet after fallback — the
  procedure is the reviewer, not the tier. It reviews diffs **cold**: never
  hand it the coder's chat context, plan, or rationale; it re-derives intent
  from the diff and repo docs so coder blindspots don't propagate into review.

## Production guardrails

- Everything production-bound is guarded: branch → PR → automated checks → merge → deploy → verify.
- Before any milestone commit or PR summary of a nontrivial change, the
  `code-reviewer` agent reviews the diff and its verdict rides with the
  summary. A BLOCK verdict stops the ship until addressed or explicitly
  overridden by the user.
- The AI never merges itself. It presents a plain-English summary derived from the **actual code diff** and waits for an explicit "go".
- If in doubt or underspecified: ask, don't assume.

### Branch workflow

- `main` — stable/demo-ready only
- `dev` — active integration
- `feature/*` — isolated experiments (current: `feature/ball-tracking`)

**When to branch off:** if a piece of work is a distinct or experimental
feature — a new subsystem (pose, court/shot-charts), a risky refactor, a model
swap, or anything that could destabilize the current branch — create a new
`feature/<name>` branch and develop there, rather than piling it onto the
active branch. Small fixes and increments to the current focus stay inline.
When you spin up a new branch, say so and note why. Merge back only through the
production guardrail (PR → checks → explicit human "go").

`videos/`, `outputs/`, `models/`, `notebooks/` are git-ignored local workspace folders.

## Operating manual

How work actually gets done here. Roadmap with milestones + success metrics:
[docs/ROADMAP.md](docs/ROADMAP.md). Ship ritual: the `/milestone` skill.
Detector evaluation: `/detector-bench`. Clip visual debugging: `/diagnose-clip`.

### Working conventions

- **Reply style:** caveman (terse, no filler) unless the user turns it off.
  Code style: ponytail — laziest correct solution, stdlib first, `ponytail:`
  comments on deliberate ceilings. Commits/PRs/docs: normal prose.
- **Milestone = commit + push** to the feature branch with a plain-English
  message ending in the Co-Authored-By line. Merges and PRs NEVER happen
  without an explicit user "go" — a standing "push at milestones" covers
  feature-branch pushes only.
- **Money, credits, new deps, destructive ops, ground-truth labels: ask.**
  Everything else with an obvious default: do it and say what you chose.
- **Long runs go to background** (`run_in_background`), full output written to
  a scratchpad file, filtered on read. Never poll with sleep chains.
- **Every clip-visible claim is verified visually** — extract the frame and
  look at it before telling the user something renders.
- **New external-library code:** check current docs via context7 first.
  Reusing an in-repo adapter needs no check.
- **New subsystem / risky experiment → new `feature/<name>` branch** (see
  Branch workflow). Small fixes stay inline.
- **Subagents: scoped gruntwork only.** User-sanctioned routing: Haiku reads,
  Sonnet builds, via tight self-contained briefs (the original ban came from an
  underscoped spawn burning a session limit). Agents deliver the smallest
  verifiable gate then STOP and report — they never babysit long local runs
  (failure mode #13). Judging, shipping, and anything long-running stays in the
  main session.

### Failure modes — named, with the rule that prevents each

| # | Failure mode | What it looks like | Preventing rule |
|---|---|---|---|
| 1 | **The Wholesale Swap** | New detector swapped for ball+hoop at once; CORE RMSE 0.76→15.4 | Change ONE variable: bench new models per-class (hoop-only, then ball-only), CORE-gate each before combining |
| 2 | **The Eager Freeze** | Hoop frozen at first confident lock (frame 3, coarse box); verdicts churned | Converge, then freeze: accumulate locked boxes, snap to median; verify freeze ≤ frame 18 on all clips |
| 3 | **Render bleeding into fit** | "Just let the trail/outcome feed the fitter" | Hard invariants section is law. New fields default render-only; fit-input changes need the CORE gate + explicit user approval (see M7 ceremony) |
| 4 | **Silent baseline re-pin** | Re-pinning `BASELINE_RMSE` to make red green | Baselines re-pin ONLY with a DECISION_LOG entry naming the approved cause |
| 5 | **Metric chasing** | Reverting a correct fix because agreement dropped 14→13 when the lost verdicts rode the bug | Investigate every flip before optimizing it; mechanism correctness beats a borderline metric |
| 6 | **The Overwrite** | Weights file replaced in place; baseline lost | `*_baseline.pt` files are immutable; new models get NEW filenames; recreate-command lives in training_plan.md |
| 7 | **Dependency clobber** | `pip install mediapipe` silently replaced opencv; 20 tests broke | Any install touching opencv/numpy → rerun `pytest` AND `check_core_drift.py` before proceeding |
| 8 | **Zombie candidate** | "Arc doesn't render" — shot never finalized mid-video | Any missing-render report → check the finalize reason in the logs FIRST (`floor_idle`/`idle`/`end_of_video`) |
| 9 | **Grep-eaten evidence** | Piping a run through grep into the task file → empty log, wasted run | Write FULL output to a scratchpad file; filter when reading |
| 10 | **Assumed ground truth** | Visually-derived make/miss labels; user corrected 3 of them | Derived labels are provisional; only user-confirmed labels enter `OUTCOME_GROUND_TRUTH` |
| 11 | **Un-gated commit** | Committing/merging without the user seeing a summary | Plain-English summary from the actual diff → wait for "go" (or a standing, scoped authorization) |
| 12 | **Stale-API confidence** | Writing against a remembered external API | context7 for current docs before new external-library code |
| 13 | **The Stalled Marathon** | Agent babysits a multi-clip sweep in its own context; watchdog kills it mid-run (happened twice: M1 sweep, MP-B gate) | Agents build + prove the smallest gate (ONE clip), then stop and report; the MAIN session runs full sweeps as background Bash and judges |
| 14 | **The Stale Path** | Redirecting output to a scratchpad path remembered from summarized context; dir no longer exists, run dies at the redirect | Use the CURRENT session's scratchpad from the system prompt; `mkdir -p` before any redirect into it |
| 15 | **The Two-Package Drift** | "Fixing" migration mismatches by editing `swishsync_cv` from the platform branch | Adapter work NEVER edits `swishsync_cv`; `git diff src/swishsync_cv` stays empty until the gated port (CLAUDE.md platform rules) |

### Quality bars — checkable, run before calling anything done

```bash
pytest -q                              # bar: 100% pass (156+ tests)
python scripts/check_core_drift.py    # bar: every CORE clip |ΔRMSE| ≤ 0.15 vs pins
python scripts/eval_shot_outcome.py   # bar: wrong-verdict count must not increase
                                      #      (unknowns tolerated; wrong verdicts not)
python scripts/run_detector_benchmark.py  # detector changes only: vs frozen baseline
```

- Arc bar: every shot with ≥ 4 flight points has `parabola_fit` non-null and
  `insufficient_points_for_fit` false in shots.json.
- Hoop bar: 19/19 clips frozen by frame ≤ 18 (benchmark prints it).
- Ship bar: DECISION_LOG entry for anything a future session would re-litigate;
  tests added for every behavior change; browser rebuilt if artifacts changed.
- Honesty bar: report failures with output verbatim; `unknown` > wrong verdict.

## Decisions & Trade-offs

Engineering decisions and their trade-offs are logged reverse-chronologically in
[docs/DECISION_LOG.md](docs/DECISION_LOG.md). Add an entry there when a choice
would confuse a future session if left unexplained. Automations: see
[docs/automations.md](docs/automations.md). Own-weights training: see
[docs/training_plan.md](docs/training_plan.md). Roadmap: see
[docs/ROADMAP.md](docs/ROADMAP.md).
