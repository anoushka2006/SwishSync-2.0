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
