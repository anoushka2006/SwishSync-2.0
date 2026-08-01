#!/usr/bin/env python3
"""Re-run all 19 Testing clips with matched hoop bboxes and lettered output videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swishsync_cv.config import HoopLockConfig, PipelineConfig
from swishsync_cv.pipeline import run_pipeline
from swishsync_cv.visualization.overlay import HOOP_LOCK_COLOR
from scripts.evaluate_testing_clips import (
    CLIP_LABELS,
    DEFAULT_EVAL_DIR,
    analyze_clip,
    build_markdown,
    eval_search_dirs,
    processed_video_name,
    slugify,
)

TESTING_DIR = ROOT / "videos" / "Testing"

# Authoritative CORE bbox from drift study; others extracted from baseline overlay.
BBOX_OVERRIDES: dict[str, tuple[float, float, float, float]] = {
    "C": (371.5, 233.0, 122.0, 173.0),
}

# Re-pinned 2026-07-05 (manual-lock path, post contiguity-clamp/pickup-gating
# tree, commit deb4170). Prior pins: A 0.54, C 0.79, P 1.19, R 0.73, S 1.10.
BASELINE_RMSE = {
    "A": 1.18,
    "C": 0.64,
    "P": 1.48,
    "R": 0.92,
    "S": 1.60,
}

# User-confirmed make/miss ground truth (2026-07-05). F and J contain real
# shots the pipeline currently never detects (FAILURE clips).
OUTCOME_GROUND_TRUTH = {
    "A": "make", "B": "miss", "C": "make", "D": "miss", "E": "miss",
    "F": "make", "G": "miss", "H": "miss", "I": "miss", "J": "miss",
    "K": "miss", "L": "make", "M": "miss", "N": "make", "O": "miss",
    "P": "miss", "Q": "make", "R": "miss", "S": "miss",
}
BASELINE_ZERO_SHOT = {"D", "E", "F", "G", "H", "I", "J", "K"}
BOUNCE_CLIPS = {"H", "L", "M", "N", "O", "Q"}
CORE_CLIPS = {"A", "C", "P", "R", "S"}


def extract_bbox_from_video(video_path: Path) -> tuple[float, float, float, float] | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    left = frame[:, : frame.shape[1] // 2]
    bgr = np.array(HOOP_LOCK_COLOR, dtype=np.uint8)
    diff = np.abs(left.astype(np.int16) - bgr.astype(np.int16))
    mask = np.all(diff <= 30, axis=2)
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def resolve_hoop_bbox(
    label: str,
    filename: str,
    eval_dir: Path,
) -> tuple[float, float, float, float]:
    if label in BBOX_OVERRIDES:
        return BBOX_OVERRIDES[label]
    slug = slugify(filename)
    for root in eval_search_dirs(eval_dir):
        for video_name in (processed_video_name(label), "processed.mp4"):
            candidate = root / slug / video_name
            if not candidate.exists():
                continue
            bbox = extract_bbox_from_video(candidate)
            if bbox is not None:
                return bbox
    raise RuntimeError(f"Could not resolve hoop bbox for clip {label} ({filename})")


def shot_detail(shots: list[dict]) -> dict:
    if not shots:
        return {"num_shots": 0}
    primary = max(
        shots,
        key=lambda s: (
            0 if s.get("insufficient_points_for_fit") else 1000,
            len(s.get("candidate_points", [])),
            (s.get("confidence") or {}).get("trajectory_confidence") or 0,
        ),
    )
    measured = [p for p in primary.get("candidate_points", []) if not p.get("interpolated")]
    diag = primary.get("fit_diagnostics") or {}
    conf = primary.get("confidence") or {}
    arc = primary.get("arc_render") or {}
    return {
        "num_shots": len(shots),
        "start_frame": primary.get("start_frame"),
        "end_frame": primary.get("end_frame"),
        "measured_frames": [p["frame_index"] for p in measured],
        "measured_count": len(measured),
        "weighted_rmse": diag.get("weighted_residual_rmse"),
        "overall_confidence": conf.get("overall_confidence"),
        "trajectory_confidence": conf.get("trajectory_confidence"),
        "visual_extension_used": arc.get("visual_extension_used"),
        "bounce_pollution_heuristic": len(primary.get("excluded_debug_points", [])) > 0
        or (
            len(measured) >= 5
            and measured[-1]["y"] > min(p["y"] for p in measured) + 80
        ),
    }


def run_all(
    only: set[str] | None = None,
    skip_existing: bool = False,
    eval_dir: Path = DEFAULT_EVAL_DIR,
) -> dict:
    eval_dir = eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    bbox_map: dict[str, tuple[float, float, float, float]] = {}
    run_results: dict[str, dict] = {}

    items = sorted(CLIP_LABELS.items(), key=lambda kv: kv[1])
    for filename, label in items:
        if only and label not in only:
            continue
        video = TESTING_DIR / filename
        out_dir = eval_dir / slugify(filename)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video = out_dir / processed_video_name(label)
        shots_path = out_dir / "shots.json"

        if skip_existing and output_video.exists() and shots_path.exists():
            print(f"\n>>> Clip {label}: skipped (existing {output_video.name})", flush=True)
            shots = json.loads(shots_path.read_text())
            run_results[label] = shot_detail(shots)
            continue

        bbox = resolve_hoop_bbox(label, filename, eval_dir)
        bbox_map[label] = bbox

        print(f"\n>>> Clip {label}: {filename}", flush=True)
        config = PipelineConfig(
            input_video=video,
            output_dir=out_dir,
            output_video_name=processed_video_name(label),
            hoop_lock=HoopLockConfig(manual_bbox_xywh=bbox),
        )
        run_pipeline(config)

        shots = json.loads(shots_path.read_text()) if shots_path.exists() else []
        run_results[label] = shot_detail(shots)

    # Fill skipped labels from disk when using --only partial runs
    for filename, label in items:
        if label in run_results:
            continue
        shots_path = eval_dir / slugify(filename) / "shots.json"
        if shots_path.exists():
            shots = json.loads(shots_path.read_text())
            run_results[label] = shot_detail(shots)

    rows = []
    for filename, label in items:
        out_dir = eval_dir / slugify(filename)
        rows.append(analyze_clip(out_dir, filename, label))

    rows.sort(key=lambda r: r["label"])
    report_path = eval_dir / "EVAL_SUMMARY.md"
    report_path.write_text(build_markdown(rows), encoding="utf-8")

    zero_shot = {lbl for lbl in BASELINE_ZERO_SHOT if run_results.get(lbl, {}).get("num_shots", 0) > 0}
    summary = {
        "bbox_map": {k: list(v) for k, v in bbox_map.items()},
        "run_results": run_results,
        "zero_shot_recovery": sorted(zero_shot),
        "zero_shot_count": len(zero_shot),
        "baseline_zero_shot_count": len(BASELINE_ZERO_SHOT),
        "rows": rows,
    }
    (eval_dir / "eval_rerun_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {report_path}")
    print(f"Wrote {eval_dir / 'eval_rerun_results.json'}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Run only these clip labels (e.g. A D)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips that already have processed_<letter>.mp4 and shots.json",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Evaluation output root (default: outputs/eval/shot_story)",
    )
    args = parser.parse_args()
    only = {x.upper() for x in args.only} if args.only else None
    run_all(only=only, skip_existing=args.skip_existing, eval_dir=args.eval_dir)
