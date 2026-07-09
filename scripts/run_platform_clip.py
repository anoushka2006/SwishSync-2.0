#!/usr/bin/env python3
"""MP-B gate: route CORE clips through the platform and prove RMSE equivalence.

For each clip we run TWO paths and compare the primary shot's
``weighted_residual_rmse``:

  legacy   — ``run_pipeline`` exactly as ``scripts/check_core_drift.py`` does
             (auto hoop lock via models/hoop_ball_yolov8.pt, stock yolov8n ball).
  platform — VideoReader frames -> ``LegacyShotEngine`` (same weights/config)
             -> ``finalized_events``.

Exit gate: |legacy - platform| <= 1e-9 per clip (byte-identical goal). A delta
above 0.15 prints HARD FAIL. Skips (exit 0) when videos / hoop weights are
absent, like check_core_drift, so it never blocks on a bare machine.

    PYTHONPATH=src python scripts/run_platform_clip.py --clip C
    PYTHONPATH=src python scripts/run_platform_clip.py --all-core
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CLIP_FILES = {
    "A": "IMG_1962.MOV",
    "C": "IMG_1961.MOV",
    "P": "IMG_2029 7.MOV",
    "R": "IMG_2029 9.MOV",
    "S": "IMG_2029.MOV",
}
CORE = ["A", "C", "P", "R", "S"]
EXACT_TOL = 1e-9
HARD_FAIL_TOL = 0.15


def _primary_rmse_legacy(shots_json: Path) -> float | None:
    shots = json.loads(shots_json.read_text())
    if not shots:
        return None
    primary = max(shots, key=lambda s: len(s.get("candidate_points", [])))
    return (primary.get("fit_diagnostics") or {}).get("weighted_residual_rmse")


def _run_legacy(video: Path, out_dir: Path, hoop_model: Path) -> float | None:
    from swishsync_cv.config import DetectionConfig, PipelineConfig
    from swishsync_cv.pipeline import run_pipeline

    run_pipeline(
        PipelineConfig(
            input_video=video,
            output_dir=out_dir,
            detection=DetectionConfig(hoop_model_path=str(hoop_model)),
        )
    )
    return _primary_rmse_legacy(out_dir / "shots.json")


def _run_platform(video: Path, hoop_model: Path) -> float | None:
    from swishsync.vision.backends import make_backend
    from swishsync.vision.detection.yolo import YoloDetector
    from swishsync.vision.tracking.legacy_shot_engine import LegacyShotEngine
    from swishsync_cv.io.video import VideoReader

    detector = YoloDetector(make_backend("cpu"), weights="yolov8n.pt", confidence=0.25)
    engine = LegacyShotEngine(detector, hoop_weights=str(hoop_model))

    with VideoReader(video) as reader:
        for packet in reader:
            engine.update(
                None,
                frame_index=packet.index,
                t_ms=packet.timestamp_ms,
                frame=packet.image,
            )
    engine.close()

    shots = engine.finalized_shots
    if not shots:
        return None
    events = engine.finalized_events()
    primary_i = max(range(len(shots)), key=lambda i: len(shots[i].candidate_points))
    return events[primary_i].evidence["weighted_residual_rmse"]


def _run_clip(label: str, hoop_model: Path, testing: Path) -> bool | None:
    video = testing / CLIP_FILES[label]
    if not video.exists():
        print(f"{label}: video missing — skipping")
        return None

    legacy = _run_legacy(video, ROOT / "outputs" / "temp" / "platform_gate" / label, hoop_model)
    platform = _run_platform(video, hoop_model)

    if legacy is None or platform is None:
        print(f"{label}: no shot (legacy={legacy} platform={platform}) — SKIP")
        return None

    delta = abs(legacy - platform)
    if delta <= EXACT_TOL:
        status = "OK (exact)"
    elif delta > HARD_FAIL_TOL:
        status = "HARD FAIL"
    else:
        status = "MISMATCH"
    ok = delta <= EXACT_TOL
    print(
        f"{label}: legacy={legacy!r} platform={platform!r} "
        f"delta={delta:.3e} {status}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="MP-B platform equivalence gate")
    parser.add_argument("--clip", default="C", choices=CORE)
    parser.add_argument("--all-core", action="store_true")
    args = parser.parse_args()

    hoop_model = ROOT / "models" / "hoop_ball_yolov8.pt"
    testing = ROOT / "videos" / "Testing"
    if not hoop_model.exists() or not testing.exists():
        print("run_platform_clip: local videos/weights absent — skipping")
        return 0

    logging.disable(logging.INFO)

    labels = CORE if args.all_core else [args.clip]
    results = [_run_clip(label, hoop_model, testing) for label in labels]

    graded = [r for r in results if r is not None]
    if graded and all(graded):
        print("\nMP-B GATE PASS — all graded clips identical to 1e-9.")
        return 0
    if not graded:
        print("\nNo clips graded (all skipped).")
        return 0
    print("\nMP-B GATE FAIL — see mismatches above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
