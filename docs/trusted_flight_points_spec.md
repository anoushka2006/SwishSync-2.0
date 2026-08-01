# trusted_flight_points Specification

Defines the `trusted_flight_points` data structure, its filter chain, phased
rollout, and invariants that must be preserved across phases.

---

## Problem

`candidate_points` contains every measured detection accumulated during shot
collection. `parabola_fit` receives a derived `fit_points` local variable built
by ad-hoc pipeline steps inside `finalize_shot()`. This intermediate has no
name, no first-class representation, and cannot be inspected or extended without
modifying the fitter.

## Solution

`trusted_flight_points` is a named, stored, first-class pre-fit selection. It
makes the filter criteria explicit and provides a documented home for future
geometric filters without touching `fit_weighted_parabola_robust`.

---

## Phase 0 — Observability (this PR)

### Filter chain (applied in order)

| Stage | Filter | Reason for exclusion |
|-------|--------|----------------------|
| 1 | Measured-only | `gap_predicted` are synthetic, not real detections |
| 2 | Primary contiguous cluster | Split at first gap > `config.trusted_flight.max_flight_gap_frames` (default **10**) |
| 3 | Floor bounce | `y > rim_center_y + floor_below_rim_margin_px` (default 100 px) |

**Not in Phase 0:**
- Statistical outlier removal — stays inside `fit_weighted_parabola_robust`
- Motion-validation checks (horizontal jump, vertical acceleration) — reserved for
  a separate, independently observable future stage

### Threshold: max_flight_gap_frames = 10

Tighter than `reacquisition_gap_frames = 15`. The gap thresholds serve different
purposes and must remain separately tunable so each threshold's CORE impact can
be measured in isolation.

### Storage

`ShotCandidate.trusted_flight_debug: TrustedFlightSelection | None`

Set by `finalize_shot()` before `candidate.state = "shot_finalized"`.

### Fit impact

**Zero.** The fitter still receives the existing `fit_points` local variable
(contiguous cluster via `reacquisition_gap_frames=15` + floor bounce filter).
`trusted_flight_debug` is stored for observability only.

### Serialization

Exported as `"trusted_flight"` block in `shot_candidate_to_dict()`:

```json
{
  "trusted_flight": {
    "trusted": [...],
    "excluded": [{"point": {...}, "reason": "post_cluster"}, ...],
    "max_gap_frames": 10,
    "trusted_count": 6,
    "excluded_count": 2
  }
}
```

Exclusion reasons: `"gap_predicted"`, `"post_cluster"`, `"floor_bounce"`.

---

## Phase 1 — story-window alignment (✅ done 2026-07-03, render-only)

`compute_shot_story()` sources `flight_start_frame` / `flight_end_frame` from
`candidate.trusted_flight_debug` (`flight_start_frame` / `flight_end_frame`
derived properties on `TrustedFlightSelection`) instead of
`candidate_points[0]` / the fit window. Falls back to the previous behavior
when `trusted_flight_debug` is `None` or its trusted tuple is empty.

**Fit impact: zero, verified.** On the 10-clip real eval (CORE A/C/P/R/S +
STRESS B/E/G/I/K), `parabola_fit`, `fit_diagnostics`, and `confidence` JSON
blocks are byte-identical before/after, and CORE story windows are unchanged.
The discriminating case — a measured gap of 11–15 frames, where the fit
cluster (gap ≤ 15) spans the gap but trusted (gap ≤ 10) stops at the break —
is covered by `tests/test_trusted_flight_story.py`; no clip in the current
suite has a gap in that band (real gaps are ≤ 3 or ≥ 24 frames).

---

## Phase 2 gate — wiring trusted_flight_points into the fitter

**Do not proceed to Phase 2 until all of the following are true:**

1. `scripts/eval_trusted_flight.py` reports **zero divergence** across the full
   19-clip benchmark suite (CORE clips A, B, C, P, R, S must show identical
   `trusted_count == fit_point_count`).
2. CORE RMSE baselines hold within +0.2 px after the fitter switch.
3. All `pytest tests/` pass.

When the gate is clear, replace the `fit_points` argument to
`fit_weighted_parabola_robust` with `candidate.trusted_flight_debug.trusted`.
Remove the now-redundant local `fit_points` construction.

---

## Future stages (Phase 3+)

Add each as a separately observable stage inside `select_trusted_flight_points`:

- Motion-validation: horizontal jump check (`max_horizontal_jump_px`)
- Motion-validation: vertical acceleration bound (`max_vertical_accel_px`)
- Upward-reversal filter (pre-apex only)

Each new stage must:
1. Be documented here with its exclusion reason string.
2. Prove CORE-safe (zero RMSE regression) before affecting the fit.
3. Not be bundled with another new stage in the same PR.

---

## Invariants

- `trusted_flight_debug` is always `None` during collection (`state == "collecting_shot"`).
- `gap_predicted` points are never in `trusted`. (Currently a no-op since they
  are not in `candidate_points`, but the filter is an explicit contract.)
- Stage order is fixed: measured-only → contiguous cluster → floor bounce.
  Changing order changes which points are attributed to which reason.
- `select_trusted_flight_points` has no side effects on `ShotCandidate`.
