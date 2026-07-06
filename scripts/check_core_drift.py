#!/usr/bin/env python3
"""CORE RMSE regression gate (auto hoop lock).

Runs the CORE clips through the full pipeline and fails if any fitted RMSE
drifts from the pinned auto-lock baseline beyond tolerance. Used by the
pre-push hook. Skips (exit 0) when Testing videos or hoop weights are absent so
it never blocks a push on a machine without the local workspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# auto-lock baselines (models/hoop_ball_yolov8.pt), 2026-07-06
BASELINE_RMSE = {"A": 0.76, "C": 0.57, "P": 1.19, "R": 0.73, "S": 1.10}
TOLERANCE = 0.15
CLIP_FILES = {
    "A": "IMG_1962.MOV",
    "C": "IMG_1961.MOV",
    "P": "IMG_2029 7.MOV",
    "R": "IMG_2029 9.MOV",
    "S": "IMG_2029.MOV",
}


def main() -> int:
    hoop_model = ROOT / "models" / "hoop_ball_yolov8.pt"
    testing = ROOT / "videos" / "Testing"
    if not hoop_model.exists() or not testing.exists():
        print("check_core_drift: local videos/weights absent — skipping")
        return 0

    import logging

    logging.disable(logging.INFO)
    from swishsync_cv.config import DetectionConfig, PipelineConfig
    from swishsync_cv.pipeline import run_pipeline

    failures = []
    for label, filename in CLIP_FILES.items():
        video = testing / filename
        if not video.exists():
            print(f"check_core_drift: {label} video missing — skipping")
            return 0
        out = ROOT / "outputs" / "temp" / "core_drift" / label
        run_pipeline(
            PipelineConfig(
                input_video=video,
                output_dir=out,
                detection=DetectionConfig(hoop_model_path=str(hoop_model)),
            )
        )
        shots = json.loads((out / "shots.json").read_text())
        if not shots:
            failures.append(f"{label}: no shot detected")
            continue
        primary = max(shots, key=lambda s: len(s.get("candidate_points", [])))
        rmse = (primary.get("fit_diagnostics") or {}).get("weighted_residual_rmse")
        base = BASELINE_RMSE[label]
        delta = abs(rmse - base) if rmse is not None else None
        status = "ok" if delta is not None and delta <= TOLERANCE else "DRIFT"
        print(f"{label}: rmse={rmse} base={base} {status}")
        if status == "DRIFT":
            failures.append(f"{label}: {rmse} vs {base}")

    if failures:
        print("\nCORE DRIFT:\n  " + "\n  ".join(failures))
        return 1
    print("\nCORE RMSE within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
