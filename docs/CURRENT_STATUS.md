# Current Status

Snapshot of active work and phase gating. Update this when phases change.

---

## trusted_flight_points

| Phase | Status | Gate |
|-------|--------|------|
| **Phase 0** — Observability | ✅ Complete + gate verified | 9/9 tests pass; non-fit-affecting; zero divergence on 10-clip real eval (2026-07-02) |
| **Phase 1** — Story-window alignment | ✅ Complete (2026-07-03) | Render-only; fit/RMSE/confidence byte-identical on 10-clip eval; CORE windows unchanged |
| **Phase 2** — Wire into fitter | ⏳ Blocked | Zero divergence on `scripts/eval_trusted_flight.py` across all 19 benchmark clips; requires explicit approval + pinned golden RMSE/confidence |
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

### Real-clip gate result (2026-07-02)

Regenerated `shots.json` via `scripts/run_full_eval_rerun.py --only A B C E G I K P R S`
and ran `scripts/eval_trusted_flight.py` per clip. Frame-level comparison
(trusted frame set vs `fit_diagnostics` `used_in_fit` frames):

| Clip | fit frames | trusted frames | post-gap drops | gap break | diverges |
|------|-----------|----------------|----------------|-----------|----------|
| A (CORE) | 21 | 21 | — | — | no |
| C (CORE) | 14 | 14 | — | — | no |
| P (CORE) | 27 | 27 | — | — | no |
| R (CORE) | 17 | 17 | — | — | no |
| S (CORE) | 17 | 17 | — | — | no |
| B | 8 | 8 | [31] | 31 | no |
| E | 5 | 5 | — | — | no |
| G | 5 | 5 | — | — | no |
| I | 6 | 6 | — | — | no |
| K | 5 | 5 | [100, 104] | 100 | no |

- **CORE: zero divergence, zero post-gap drops.** Trusted selection is
  frame-identical to the current fit input on every CORE clip.
- **Long-gap mechanism confirmed on B and K**: B drops frame 31 (24-frame gap
  after flight end at frame 7); K drops frames 100/104 (41-frame gap after
  frame 59). The current fit path also excluded these frames, so trusted and
  fit agree even where gaps exist.
- **E/G/I have no post-gap drops** — their intra-flight gaps are ≤3 frames,
  well below the threshold. These are sparse/short-flight clips, not long-gap
  clips; trusted_flight fixes membership, not perspective.
- **Accepted `max_flight_gap_frames = 10`** — no tuning needed; the tighter
  gap (10 vs reacquisition's 15) never split a real flight.

### Phase 1 result (2026-07-03)

`compute_shot_story` now sources the flight window from
`trusted_flight_debug.flight_start_frame` / `.flight_end_frame` (fallback to
prior behavior when absent). Verified render-only on the 10-clip eval:
`parabola_fit` / `fit_diagnostics` / `confidence` byte-identical before/after
on every clip; story windows unchanged on all 10 (real gaps are ≤ 3 or ≥ 24
frames — the fit path already excluded the post-gap points). The gap-11–15
discriminating case is locked in by `tests/test_trusted_flight_story.py`
(5 tests). Full suite: 121 passed.

### To unblock Phase 2

1. Extend the eval to the remaining 9 benchmark clips (D, F, H, J, L, M, N, O, Q).
2. Confirm zero divergence (`trusted_count == fit_point_count` for all shots).
3. Wire `trusted_flight` into the fitter behind the gate defined in
   `docs/trusted_flight_points_spec.md`.

---

### 2026-07-05 — auto hoop lock + make/miss (this session)

- **Automatic hoop lock** shipped behind `--hoop-model` / 
  `DetectionConfig.hoop_model_path` using `models/hoop_ball_yolov8.pt`
  (unlicensed third-party weights — local use only). 19/19 clips lock by
  frame 3; CORE RMSE improves on all five clips vs manual bboxes
  (A 1.18→0.76, C 0.64→0.56, P 1.48→1.19, R 0.92→0.73, S 1.60→1.10).
  Benchmark: `scripts/eval_auto_hoop.py`.
- **Make/miss** shipped as render-only `ShotCandidate.outcome`
  (`tracking/shot_outcome.py`), serialized in shots.json; eval browser shows
  a make/miss badge + rim margin. Ground-truth labels derived visually from
  rim-area montages, pending user confirmation.
- Eval browser (`scripts/build_eval_browser.py`) now surfaces the verdict.

## Branch

`feature/ball-tracking`
