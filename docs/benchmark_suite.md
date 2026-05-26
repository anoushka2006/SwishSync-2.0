# SwishSync 2.0 Benchmark Suite

Canonical definition of the 19 Testing clips, their roles, expected behavior, and pass/fail criteria. Use this document when evaluating pipeline, lifecycle, fitting, or visualization changes so regressions are measured consistently.

**Source of truth for letter mapping:** `scripts/evaluate_testing_clips.py` → `CLIP_LABELS`  
**Canonical eval artifacts:** `outputs/eval/shot_story/`  
**Do not treat render-only layers as fit inputs:** pickup preview, post-shot paths, and gap bridges are visualization metadata only.

---

## 1. Benchmark clip map

All clips live under `videos/Testing/`. Eval outputs use a slugified directory name (e.g. `IMG_2027_2/` for clip D).

| Letter | Filename | Category | Camera / scene notes | Known issues | Intended purpose |
|--------|----------|----------|----------------------|--------------|------------------|
| **A** | `IMG_1962.MOV` | CORE | Side-ish gym angle; stable hoop lock with manual/interactive bbox | Early-session reference; sensitive to hoop bbox drift | Clean single-shot regression anchor; arc extension toward rim |
| **B** | `IMG_1963.MOV` | CORE | Similar gym setup to A; moderate side angle | None critical when hoop locked | Second clean reference; flight-only fit and floor exclusion validation |
| **C** | `IMG_1961.MOV` | CORE | Primary gather/pickup reference; ball visible pre-release | Collection starts ~3 frames after release frame (sparse gate); pickup preview covers gap | **Primary CORE clip** — RMSE drift study bbox `(371.5, 233, 122, 173)`; shot-story pickup path reference |
| **D** | `IMG_2027 2.MOV` | STRESS | Strong side angle; flight segment far from rim in x | Historically zero-shot in baseline runs; arc must not extend falsely toward rim | Side-angle stress; `compute_arc_render_metadata` no-extension case |
| **E** | `IMG_2027 3.MOV` | STRESS | 2027 series; sparse detection gaps | Baseline zero-shot candidate | Lifecycle start/end under sparse YOLO |
| **F** | `IMG_2027 4.MOV` | FAILURE | 2027 series | **Accepted detector failure** — empty `shots.json`, no ball track | Future detection recovery metric; do not fail lifecycle PRs on F alone |
| **G** | `IMG_2027 6.MOV` | STRESS | 2027 series | Baseline zero-shot candidate | Sparse acquisition + hoop lock stress |
| **H** | `IMG_2027.MOV` | STRESS | Bounce / post-rim reacquisition | Split-shot history; 64f flight→post gap suppresses `show_post_shot`; 5 `post_shot_points` captured but hidden by guard | Bounce rejection, finalize cooldown, post-shot metadata stress |
| **I** | `IMG_2028 2.mov` | STRESS | 2028 series | Baseline zero-shot candidate | Angle / detection robustness |
| **J** | `IMG_2028 3.mov` | FAILURE | 2028 series | **Accepted detector failure** — empty `shots.json` | Future detection recovery metric |
| **K** | `IMG_2028.mov` | STRESS | 2028 series | Baseline zero-shot candidate | General robustness |
| **L** | `IMG_2029 11.MOV` | STRESS | Bounce clip (2029 series) | Floor-band points; excluded debug markers expected | Bounce exclusion + lifecycle end conditions |
| **M** | `IMG_2029 3.MOV` | STRESS | Bounce clip | Same bounce family as L/N/O | Floor bounce filtering |
| **N** | `IMG_2029 4.MOV` | STRESS | Bounce clip | Post-shot capture often empty; pickup path should show when data exists | Bounce + pickup visualization |
| **O** | `IMG_2029 6.MOV` | STRESS | Bounce clip | Post-rim continuation noise | Cooldown / no second shot |
| **P** | `IMG_2029 7.MOV` | CORE | Clean make; high trajectory confidence | Stale-eval risk if not rerun after serialization changes | CORE RMSE anchor (baseline ~1.19 px) |
| **Q** | `IMG_2029 8.MOV` | STRESS | Multi-motion session | Multi-shot faded history (often 1 shot detected in eval) | Shot memory / `show_shot_history` stress |
| **R** | `IMG_2029 9.MOV` | CORE | Early release frame (~24); zero-shot-style start | Very early `start_frame`; pickup may be sparse | Early-start CORE regression |
| **S** | `IMG_2029.MOV` | CORE | Clean make; long visible arc | None critical | CORE RMSE anchor (baseline ~1.10 px) |

### Category counts

| Category | Clips | Role |
|----------|-------|------|
| **CORE** | A, B, C, P, R, S | Stable clean references — must not regress |
| **STRESS** | D, E, G, H, I, K, L, M, N, O, Q | Hard cases — improve over time, not strict regression gates |
| **FAILURE** | F, J | Known detector misses — track future recovery only |

---

## 2. Clip roles

### CORE — stable clean reference clips

These clips represent the product-quality bar for a single clean make:

- One physical shot → **exactly one** finalized shot object (no spurious Shot 2).
- Weighted RMSE and trajectory confidence remain stable run-to-run.
- Hoop lock usable (manual first-frame selection or extracted bbox in eval reruns).
- Pickup path visible when `story.show_pickup=true` and `pickup_points ≥ 2`.
- Render-only layers must **not** change `candidate_points`, fit coefficients, RMSE, or confidence.

**Authoritative CORE RMSE baselines** (from `scripts/run_full_eval_rerun.py` → `BASELINE_RMSE`, shot_story era):

| Clip | Baseline weighted RMSE (px) | Max allowed (+0.2) |
|------|----------------------------|--------------------|
| A | 0.54 | 0.74 |
| C | 0.79 | 0.99 |
| P | 1.19 | 1.39 |
| R | 0.73 | 0.93 |
| S | 1.10 | 1.30 |

Clip **B** is CORE by behavior but has no numeric baseline in the rerun script — treat as qualitative regression (single shot, clean arc, no bounce pollution).

### STRESS — lifecycle and robustness

Used to test:

- Bounce / floor-band rejection (`excluded_debug_points`, not in fit).
- Post-rim reacquisition without spawning a second shot (cooldown guard).
- Long mid-flight YOLO gaps and gap-predicted continuity (render-only bridges).
- Side angles where arc must not extend through the hoop without measured flight (`clip D` rule).
- Early or late shot segmentation, split-shot prevention, insufficient-fit paths.
- Multi-shot sessions and faded history (clip Q).

STRESS clips may fail individual heuristics today; record notes in `EVAL_SUMMARY.md` but do not block merges unless a CORE clip regresses.

### FAILURE — accepted detector failures

Clips **F** and **J** currently produce **zero shot objects** (empty or missing meaningful `shots.json` entries). These are **not** lifecycle failures — they measure future **detection** improvements.

Pass criteria for F/J today: pipeline completes without crash; failure is documented, not hidden.

---

## 3. Expected behavior per benchmark clip

Use `shots.json` + `processed_<letter>.mp4` for verification. Primary shot = highest-quality finalized shot when multiple objects exist (see `analyze_clip()` in `scripts/evaluate_testing_clips.py`).

### Global expectations (all clips with a detected shot)

| Behavior | Expectation |
|----------|-------------|
| Fit timing | Parabola fit runs **once** at finalize, never during collection |
| Measured-only collection | Interpolated sparse points do not append to `candidate_points` |
| Floor bounce after rim | Points below rim band → `excluded_debug_points`, not fit |
| Gap bridges | `gap_predicted` continuity is render-only; does not alter fit inputs |
| Pickup path | Render-only; visible when `pickup_points ≥ 2` and guards allow (idle preview + collecting + finalized) |
| Post-shot path | Render-only; visible only when `post_shot_points ≥ 2` **and** `story.show_post_shot=true` |
| Dual pane | Left = camera debug; right = trajectory / shot story |

### Per-category expectations

#### CORE (A, B, C, P, R, S)

| Check | Pass |
|-------|------|
| Shot count | Exactly **1** finalized shot |
| `insufficient_points_for_fit` | **false** |
| Weighted RMSE | ≤ baseline + **0.2 px** (B: visually clean arc) |
| Trajectory confidence | ≥ **90%** (typical CORE runs) |
| Bounce pollution | **false** — no floor-band points in fit |
| Mid-flight ball lost | **false** — max measured gap ≤ 15 frames (or gap bridged visually only) |
| Second shot after rim | **none** within cooldown window |
| Pickup path | **visible** on both panes when `show_pickup=true` |
| Arc through hoop | Extension toward rim only when flight data supports it (see clip D inverse) |

#### STRESS — bounce family (H, L, M, N, O)

| Check | Pass |
|-------|------|
| Floor points | Excluded from fit; may appear in `excluded_debug_points` |
| Single primary arc | One dominant shot object for the primary attempt |
| False second shot | Must not appear for falling/reacquired ball within cooldown |
| Post-shot viz | May be suppressed by gap guard (H) or empty capture (N) — document, do not treat as render bug alone |

#### STRESS — side angle / sparse (D, E, G, I, K)

| Check | Pass |
|-------|------|
| Clip D | **No** visual arc extension to rim when observed flight is still short of rim in x |
| Zero-shot baseline clips | May have 0 shots — improvement tracked over time |
| Shot start | Should not include long dribble carry when detection works |

#### STRESS — session / multi-motion (Q)

| Check | Pass |
|-------|------|
| Shot memory | Prior finalized arcs may render faded when `show_shot_history=true` |
| Shot count | Document actual count; goal is stable primary arc |

#### FAILURE (F, J)

| Check | Pass (current) |
|-------|----------------|
| Detection | Zero shots accepted |
| Pipeline | Exits cleanly; `EVAL_SUMMARY.md` notes failure |
| Future target | ≥ 1 shot with usable trajectory when detection improves |

### Clip-specific notes

| Clip | Additional expectation |
|------|------------------------|
| **C** | `pickup_points=3` (f59–61), `show_pickup=true`; idle preview covers f62–64 release gap |
| **D** | Unit-tested: no rim extension when flight x ≪ rim x |
| **H** | `post_shot_points` may exist with `show_post_shot=false` (64f gap guard) |
| **R** | Very early `start_frame`; pickup guards must not reject valid gather |
| **F, J** | Empty `shots.json` is expected today |

---

## 4. Pass / fail criteria

### Global gates (any PR touching pipeline, tracking, or viz)

| Criterion | Pass | Fail |
|-----------|------|------|
| CORE RMSE regression | No CORE clip exceeds baseline + **0.2 px** | Any of A, C, P, R, S over threshold |
| CORE false shots | No new second shot on CORE clips | Extra shot object on A, B, C, P, R, S |
| Fit corruption from render | Duplicate pipeline runs produce **identical** `shots.json` fit metrics | RMSE, `candidate_points`, or confidence change when only viz code changes |
| Pickup/post contamination | `candidate_points` count and frames unchanged vs pre-viz run | Story/preview layers alter fit inputs |
| Tests | `pytest tests/` all pass | Any test failure |

### Detection gates

| Criterion | Pass | Fail |
|-----------|------|------|
| F, J | Remain accepted failures OR improve with documented metrics | Silent success with garbage fit |
| Future recovery | Track: shot detected, measured point count, usable trajectory score | — |

### Lifecycle gates

| Criterion | Pass | Fail |
|-----------|------|------|
| Bounce clips | Floor-band measured points excluded from fit | Bounce cluster in `candidate_points` |
| Cooldown | No spurious Shot 2 on reacquisition (H, N, O) | Second shot from falling ball after finalize |
| Gap bridges | Present in viz when `gap_predicted_frames` set | Gap points injected into fit |
| Long-gap split | One arc when single physical motion | Two arcs from one shot (regression) |

### Visualization gates (render-only PRs)

| Criterion | Pass | Fail |
|-----------|------|------|
| Pickup visibility | CORE clips show pickup on release, collecting, and post-finalize frames when data exists | Missing on both panes after eval rerun |
| Post-shot | Drawn only when data + `show_post_shot` | Invented paths without `post_shot_points` |
| Story fallback | Stale runs without `story` still show arc + pickup via fallback | Blank trajectory panel |

### Scoring reference

`scripts/evaluate_testing_clips.py` assigns `trajectory_usable` (1–5) and flags:

- `ball_lost_mid_flight`
- `bounce_floor_pollution`
- `shot_starts_too_late`
- `shot_ends_too_early`

CORE clips should score **≥ 4** with **clean** notes after reruns.

---

## 5. Future milestones

Track separately from current pass/fail — not required for today's benchmark green state:

1. **Detection robustness** — recover F and J; reduce zero-shot rate on D, E, G, I, K baseline clips.
2. **Auto hoop lock** — reduce reliance on `--select-hoop-on-first-frame` and manual bbox overrides in eval reruns.
3. **Perspective normalization** — optional analytical panel beyond camera-space arc (prior art in repo history; not current default).
4. **Post-shot capture** — populate `post_shot_points` on bounce/make clips without weakening gap guards; render mint post path when appropriate.
5. **Make / miss** — rim crossing classification using hoop anchor and flight metadata.
6. **Pose analytics** — shooter pose, release angle, jump timing (out of current CV module scope).

---

## 6. Running future evals

### Full 19-clip benchmark rerun

```bash
# Canonical location (default)
python scripts/run_full_eval_rerun.py

# Subset
python scripts/run_full_eval_rerun.py --only C P R N S

# Isolated experiment (does not overwrite canonical eval)
python scripts/run_full_eval_rerun.py --eval-dir outputs/eval/my_experiment
```

Writes per clip:

```text
outputs/eval/shot_story/<slug>/
  processed_<letter>.mp4    # e.g. processed_c.mp4
  shots.json
  detections.jsonl
  detections.csv
```

And at eval root:

```text
outputs/eval/shot_story/EVAL_SUMMARY.md
outputs/eval/shot_story/eval_rerun_results.json
```

### Report-only (no pipeline rerun)

```bash
python scripts/evaluate_testing_clips.py --report
python scripts/evaluate_testing_clips.py --report --eval-dir outputs/eval/shot_story
```

### CORE drift check (RMSE A/B)

```bash
python scripts/verify_core_drift.py
# outputs → outputs/debug/core_drift_verify/
```

### Single-clip ad-hoc run

Prefer temp dirs — do not overwrite canonical eval accidentally:

```bash
swishsync-cv \
  --input videos/Testing/IMG_1961.MOV \
  --output-dir outputs/temp/clip_c_check \
  --output-video-name processed_c.mp4 \
  --select-hoop-on-first-frame
```

### Review checklist after a rerun

1. Open `outputs/eval/shot_story/EVAL_SUMMARY.md` — scan CORE rows for RMSE and notes.
2. Spot-check CORE `processed_<letter>.mp4` at release frame, mid-collection, and post-finalize for pickup path.
3. Confirm F and J still documented as failures (or improved if detection PR).
4. Diff `eval_rerun_results.json` against prior commit for CORE RMSE and shot counts.
5. Run `pytest tests/` before merging.

### Related docs

- [`docs/output_workflow.md`](output_workflow.md) — output folder layout and naming
- [`outputs/README.md`](../outputs/README.md) — what to keep vs delete
- `scripts/evaluate_testing_clips.py` — `CLIP_LABELS`, `analyze_clip()`, markdown report builder
- `scripts/run_full_eval_rerun.py` — `CORE_CLIPS`, `BOUNCE_CLIPS`, `BASELINE_RMSE`

---

*Last aligned with shot_story eval era (19 Testing clips, letters A–S, canonical root `outputs/eval/shot_story/`).*
