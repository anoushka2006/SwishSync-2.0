#!/usr/bin/env python3
"""Compare two detector backends across the A-S benchmark clips.

Runs the SAME pipeline with two detector configs (baseline vs custom trained
weights) and reports, per clip and aggregated:

  - shot detection rate      (shots found on clips that should have one)
  - measured points / shot   (recall proxy — more measured points = better recall)
  - trajectory completeness  (measured points / flight span)
  - weighted RMSE            (fit quality)
  - overall confidence
  - detection count          (false-positive proxy at fixed conf; higher w/o more
                              measured points suggests spurious boxes)
  - processing speed         (frames / second, CPU)

True recall / false-positive rates need per-frame ball ground truth (not in
A-S); the count-based proxies above stand in until frames are annotated.

The detector is swapped purely by model path — same YoloObjectDetector /
`Detector` interface, same SparseBallDetection output — so downstream (sparse
detection, shot lifecycle, fitting, confidence, render) is byte-for-byte
identical logic across backends.

Example:
  python scripts/compare_detectors.py \
    --baseline-hoop models/hoop_ball_yolov8.pt \
    --candidate-hoop models/hoop_ball_yolo11n.pt \
    --candidate-ball models/hoop_ball_yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swishsync_cv.config import DetectionConfig, PipelineConfig
from swishsync_cv.pipeline import run_pipeline
from scripts.evaluate_testing_clips import CLIP_LABELS, slugify

TESTING = ROOT / "videos" / "Testing"
# clips that genuinely contain a shot (F, J are accepted zero-shot failures)
EXPECTED_SHOT = set("ABCDEGHIKLMNOPQRS")


@dataclass
class ClipMetrics:
    shot_detected: bool
    num_shots: int
    measured_points: int
    completeness: float | None
    weighted_rmse: float | None
    overall_confidence: float | None
    detection_count: int
    fps: float


def _run_one(video: Path, out_dir: Path, ball_model: str, hoop_model: str | None) -> ClipMetrics:
    detection = DetectionConfig(
        model_path=ball_model,
        hoop_model_path=hoop_model,
    )
    start = time.perf_counter()
    result = run_pipeline(PipelineConfig(input_video=video, output_dir=out_dir, detection=detection))
    elapsed = time.perf_counter() - start
    fps = result.processed_frames / elapsed if elapsed > 0 else 0.0

    shots = json.loads((out_dir / "shots.json").read_text())
    if not shots:
        return ClipMetrics(False, 0, 0, None, None, None, result.detection_count, fps)
    primary = max(shots, key=lambda s: len(s.get("candidate_points", [])))
    measured = [p for p in primary.get("candidate_points", []) if not p.get("interpolated")]
    span = None
    if primary.get("end_frame") is not None and measured:
        span = primary["end_frame"] - primary["start_frame"] + 1
    completeness = (len(measured) / span) if span else None
    diag = primary.get("fit_diagnostics") or {}
    conf = primary.get("confidence") or {}
    return ClipMetrics(
        shot_detected=not primary.get("insufficient_points_for_fit", False),
        num_shots=len(shots),
        measured_points=len(measured),
        completeness=completeness,
        weighted_rmse=diag.get("weighted_residual_rmse"),
        overall_confidence=conf.get("overall_confidence"),
        detection_count=result.detection_count,
        fps=fps,
    )


def _aggregate(rows: dict[str, ClipMetrics]) -> dict[str, float]:
    expected = [lab for lab in rows if lab in EXPECTED_SHOT]
    detected = [lab for lab in expected if rows[lab].shot_detected]

    def avg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "shot_rate": len(detected) / len(expected) if expected else 0.0,
        "measured": avg(rows[l].measured_points for l in detected),
        "completeness": avg(rows[l].completeness for l in detected),
        "rmse": avg(rows[l].weighted_rmse for l in detected),
        "confidence": avg(rows[l].overall_confidence for l in detected),
        "detections": avg(rows[l].detection_count for l in rows),
        "fps": avg(rows[l].fps for l in rows),
    }


def _run_config(label: str, ball: str, hoop: str | None, only) -> dict[str, ClipMetrics]:
    rows: dict[str, ClipMetrics] = {}
    for filename, clip in sorted(CLIP_LABELS.items(), key=lambda kv: kv[1]):
        if only and clip not in only:
            continue
        video = TESTING / filename
        if not video.exists():
            continue
        out = ROOT / "outputs" / "temp" / "detector_cmp" / label / slugify(filename)
        rows[clip] = _run_one(video, out, ball, hoop)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two detector backends")
    parser.add_argument("--baseline-ball", default="yolov8n.pt")
    parser.add_argument("--baseline-hoop", default=str(ROOT / "models" / "hoop_ball_yolov8.pt"))
    parser.add_argument("--candidate-ball", default="yolov8n.pt")
    parser.add_argument("--candidate-hoop", required=True, help="Trained weights .pt")
    parser.add_argument("--only", nargs="*", help="Clip labels")
    parser.add_argument("--out", type=Path, help="Write JSON summary here")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    only = set(args.only) if args.only else None
    base = _run_config("baseline", args.baseline_ball, args.baseline_hoop, only)
    cand = _run_config("candidate", args.candidate_ball, args.candidate_hoop, only)

    agg_b, agg_c = _aggregate(base), _aggregate(cand)
    print(f"\n{'metric':22} {'baseline':>10} {'candidate':>10} {'delta':>10}")
    for key, better in [("shot_rate", "up"), ("measured", "up"), ("completeness", "up"),
                        ("rmse", "down"), ("confidence", "up"), ("detections", "-"),
                        ("fps", "up")]:
        b, c = agg_b[key], agg_c[key]
        arrow = "" if better == "-" else (" *" if (c > b) == (better == "up") else " x")
        print(f"{key:22} {b:>10.3f} {c:>10.3f} {c - b:>+10.3f}{arrow}")

    if args.out:
        args.out.write_text(json.dumps({"baseline": agg_b, "candidate": agg_c}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
