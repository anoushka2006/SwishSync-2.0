# Platform refactor (Phase A) — branch scaffold

Branch: `feature/platform-refactor`. This is the model-agnostic platform
skeleton. **No product scope changes** — SwishSync stays the single-shot shot
trajectory analyzer. This refactor only makes detectors/trackers swappable and
makes GPU inference an option without losing CPU-only operation.

## What's here

```
src/swishsync/            NEW platform package (coexists with swishsync_cv)
  core/schemas.py         IR: Detection, Track, CourtModel, WorldState, Event, ParabolaFit
  core/registry.py        plugin registry (detector/tracker/calibrator/event)
  core/pipeline.py        stage orchestrator + IR caching
  vision/interfaces.py    Detector, Tracker, CourtCalibrator ABCs
  vision/backends.py      cpu/cuda/triton/http backends (WHERE inference runs)
  events/interfaces.py    EventDetector ABC
  events/shot.py          ShotEventDetector — reuses the legacy robust fit
configs/pipeline_cpu.yaml  CPU tier (fallback, no GPU)
configs/pipeline_gpu.yaml  GPU tier (same pipeline, backend: cuda)
scripts/smoke_platform.py  runnable end-to-end smoke (no torch/video)
tests/test_platform_smoke.py
```

## Run the smoke test

```bash
PYTHONPATH=src python scripts/smoke_platform.py   # prints SMOKE OK
pytest tests/test_platform_smoke.py -q
```

Synthetic ball track -> `ShotEventDetector` -> one SHOT event with a fit whose
`a` coefficient recovers the ground-truth parabola. Proves IR + registry +
event interface + legacy-fit adapter connect.

## The design in one rule

`WHAT a model computes` (Detector/Tracker/EventDetector) is separate from
`WHERE it runs` (backend: cpu/cuda). CPU-only stays a supported tier
forever; GPU is a config swap, never a rewrite.

## Migration plan (how the old engine moves over, safely)

1. **[done here]** Platform skeleton + `ShotEventDetector` adapter that calls the
   existing `swishsync_cv.tracking.parabola.fit_weighted_parabola_robust`.
2. Wrap the current YOLO detector as `swishsync.vision.detection.yolo` behind the
   `Detector` interface (backend supplies the device).
3. Wrap the legacy engine WHOLESALE as `legacy_shot_engine` (sparse buffer +
   ShotCandidateManager + finalize as one Tracker; no ByteTrack — a
   multi-object Kalman tracker for one ball changes association and breaks the
   gate). Rationale: ~90% of shot correctness is candidate selection
   (gates/lifecycle), not the fitter — fitting raw tracks can never match.
4. Route CORE clips through the platform via `scripts/run_platform_clip.py`.
   **Exit gate:** `weighted_residual_rmse` identical to the legacy runner to
   1e-9 per CORE clip (hard fail above 0.15). Adapter work never edits
   `swishsync_cv` (`git diff src/swishsync_cv` empty).
5. Split the wholesale wrap into true stages one seam at a time (MP-C),
   re-running the gate after each split; retire `swishsync_cv` module by module.
   Delete the old package only when every CORE clip is byte-identical.

## Guardrails carried over

- Render-only stays render-only; `Event.evidence` is mandatory (nothing
  unexplained enters an event).
- No fit-input change without the CORE gate + explicit "go".
- CPU-runnable config kept so tests and benchmarks survive the GPU move.
