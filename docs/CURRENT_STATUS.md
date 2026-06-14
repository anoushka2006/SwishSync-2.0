# Current Status

Snapshot of active work and phase gating. Update this when phases change.

---

## trusted_flight_points

| Phase | Status | Gate |
|-------|--------|------|
| **Phase 0** — Observability | ✅ Complete | 9/9 tests pass; non-fit-affecting |
| **Phase 2** — Wire into fitter | ⏳ Blocked | Zero divergence on `scripts/eval_trusted_flight.py` across all 19 benchmark clips |
| **Phase 3+** — Motion validation | ⏳ Future | Each filter gated independently after Phase 2 |

### Phase 0 deliverables (committed)

- `src/swishsync_cv/tracking/trusted_flight.py` — `select_trusted_flight_points()`
- `src/swishsync_cv/config.py` — `TrustedFlightConfig(max_flight_gap_frames=10)`
- `src/swishsync_cv/data.py` — `TrustedFlightSelection`, `TrustedFlightExclusion`,
  `ShotCandidate.trusted_flight_debug`
- `src/swishsync_cv/tracking/shot_finalization.py` — computes `trusted_flight_debug`
  in parallel; fit path unchanged
- `src/swishsync_cv/utils/serialization.py` — `"trusted_flight"` block in JSON
- `tests/test_trusted_flight.py` — 9 tests
- `scripts/eval_trusted_flight.py` — divergence eval tool
- `docs/trusted_flight_points_spec.md` — full spec and phase gates

### To unblock Phase 2

1. Run `python scripts/eval_trusted_flight.py <shots.json>` on all 19 benchmark clips.
2. Confirm zero divergence (`trusted_count == fit_point_count` for all shots).
3. If divergence found: investigate whether the tighter gap (10 vs 15) is causing
   the split and decide whether to adjust threshold or accept the difference.

---

## Branch

`feature/ball-tracking`
