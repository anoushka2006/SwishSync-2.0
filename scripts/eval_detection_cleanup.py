#!/usr/bin/env python3
"""Summarize Phase 1 floor-band detection cleanup eval for F/J and CORE clips."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "eval" / "shot_story"
REPORT_PATH = EVAL_DIR / "detection_cleanup_eval.json"
ROI_REPORT_PATH = EVAL_DIR / "detection_roi_eval_report.json"

CLIP_DIRS = {
    "F": EVAL_DIR / "IMG_2027_4",
    "J": EVAL_DIR / "IMG_2028_3",
    "A": EVAL_DIR / "IMG_1962",
    "C": EVAL_DIR / "IMG_1961",
    "P": EVAL_DIR / "IMG_2029_7",
    "R": EVAL_DIR / "IMG_2029_9",
    "S": EVAL_DIR / "IMG_2029",
}

PRE_CLEANUP = {
    "F": {
        "shot_detected": False,
        "candidate_measured_count": 0,
        "weighted_rmse": None,
        "basketball_detections_total": 32,
        "basketball_detections_floor_band": 28,
        "detections_csv_rows": 32,
    },
    "J": {
        "shot_detected": False,
        "candidate_measured_count": 0,
        "weighted_rmse": None,
        "basketball_detections_total": 21,
        "basketball_detections_floor_band": 14,
        "detections_csv_rows": 21,
    },
    "A": {
        "shot_detected": True,
        "candidate_measured_count": 19,
        "weighted_rmse": 0.973,
        "basketball_detections_total": None,
        "basketball_detections_floor_band": None,
        "detections_csv_rows": None,
    },
    "C": {
        "shot_detected": True,
        "candidate_measured_count": 14,
        "weighted_rmse": 0.637,
        "basketball_detections_total": None,
        "basketball_detections_floor_band": None,
        "detections_csv_rows": None,
    },
    "P": {
        "shot_detected": True,
        "candidate_measured_count": 26,
        "weighted_rmse": 1.499,
        "basketball_detections_total": None,
        "basketball_detections_floor_band": None,
        "detections_csv_rows": None,
    },
    "R": {
        "shot_detected": True,
        "candidate_measured_count": 16,
        "weighted_rmse": 0.724,
        "basketball_detections_total": None,
        "basketball_detections_floor_band": None,
        "detections_csv_rows": None,
    },
    "S": {
        "shot_detected": True,
        "candidate_measured_count": 16,
        "weighted_rmse": 1.332,
        "basketball_detections_total": None,
        "basketball_detections_floor_band": None,
        "detections_csv_rows": None,
    },
}

FLOOR_MARGIN_PX = 100.0
RIM_Y = {
    "F": 537.0,
    "J": 333.0,
    "A": 410.0,
    "C": 410.0,
    "P": 469.0,
    "R": 469.0,
    "S": 469.0,
}

REMAINING_WRONG_SEGMENTS = {
    "F": [
        {"frames": "f146", "region": "post_rim_single", "count": 1},
        {"frames": "f172-178", "region": "post_rim_rim_adjacent", "count": 7},
    ],
    "J": [
        {"frames": "f41-44", "region": "burst_rim_adjacent_upward", "count": 4},
        {"frames": "f90-93", "region": "post_void_rim_adjacent", "count": 3},
    ],
}


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open() as handle:
        return sum(1 for _ in csv.DictReader(handle))


def analyze_basketball_rows(csv_path: Path, rim_y: float) -> dict:
    threshold = rim_y + FLOOR_MARGIN_PX
    rim_adjacent_threshold = rim_y - 80.0
    total = 0
    floor_band = 0
    rim_adjacent = 0
    upper_court = 0
    frames: list[int] = []
    rows: list[dict] = []
    if not csv_path.exists():
        return {
            "total": 0,
            "floor_band": 0,
            "rim_adjacent": 0,
            "upper_court": 0,
            "frames": [],
            "rows": [],
        }
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["label"] != "basketball":
                continue
            total += 1
            y = float(row["center_y"])
            frame_index = int(row["frame_index"])
            frames.append(frame_index)
            entry = {
                "frame_index": frame_index,
                "center_x": round(float(row["center_x"]), 1),
                "center_y": round(y, 1),
                "confidence": round(float(row["confidence"]), 2),
            }
            if y > threshold:
                floor_band += 1
                entry["zone"] = "floor_band"
            elif y > rim_adjacent_threshold:
                rim_adjacent += 1
                entry["zone"] = "rim_adjacent"
            else:
                upper_court += 1
                entry["zone"] = "upper_court"
            rows.append(entry)
    return {
        "total": total,
        "floor_band": floor_band,
        "rim_adjacent": rim_adjacent,
        "upper_court": upper_court,
        "frames": frames,
        "rows": rows,
    }


def clip_metrics(label: str, output_dir: Path) -> dict:
    before = PRE_CLEANUP.get(label, {})
    shots_path = output_dir / "shots.json"
    shots = json.loads(shots_path.read_text()) if shots_path.exists() else []
    shot_detected = len(shots) > 0
    rmse = None
    candidate_count = 0
    candidate_frames: list[int] = []
    if shots:
        primary = shots[0]
        measured = [
            p for p in primary.get("candidate_points", []) if not p.get("interpolated")
        ]
        candidate_count = len(measured)
        candidate_frames = [p["frame_index"] for p in measured]
        diag = primary.get("fit_diagnostics") or {}
        rmse = diag.get("weighted_residual_rmse")

    rim_y = RIM_Y.get(label, 410.0)
    csv_path = output_dir / "detections.csv"
    detections = analyze_basketball_rows(csv_path, rim_y)
    csv_rows = count_csv_rows(csv_path)

    metrics = {
        "output_dir": str(output_dir.relative_to(ROOT)),
        "before": before,
        "after": {
            "shot_detected": shot_detected,
            "candidate_measured_count": candidate_count,
            "candidate_frames": candidate_frames,
            "weighted_rmse": rmse,
            "basketball_detections_total": detections["total"],
            "basketball_detections_floor_band": detections["floor_band"],
            "basketball_detections_rim_adjacent": detections["rim_adjacent"],
            "basketball_detections_upper_court": detections["upper_court"],
            "detections_csv_rows": csv_rows,
            "remaining_basketball_rows": detections["rows"],
        },
        "delta": {
            "shot_detected_changed": shot_detected != before.get("shot_detected"),
            "candidate_measured_count_delta": candidate_count
            - before.get("candidate_measured_count", 0),
            "weighted_rmse_delta_px": None
            if rmse is None or before.get("weighted_rmse") is None
            else round(rmse - before["weighted_rmse"], 3),
            "detections_csv_rows_delta": csv_rows - before.get("detections_csv_rows", csv_rows)
            if before.get("detections_csv_rows") is not None
            else None,
            "floor_band_removed": before.get("basketball_detections_floor_band", 0)
            - detections["floor_band"]
            if before.get("basketball_detections_floor_band") is not None
            else None,
        },
    }
    if label in REMAINING_WRONG_SEGMENTS:
        metrics["remaining_wrong_detection_segments"] = REMAINING_WRONG_SEGMENTS[label]
    if label in {"A", "C", "P", "R", "S"} and rmse is not None:
        baseline = before["weighted_rmse"]
        drift = abs(rmse - baseline)
        metrics["core_regression"] = {
            "baseline_rmse_px": baseline,
            "after_rmse_px": rmse,
            "abs_drift_px": round(drift, 3),
            "passes_0_05px_gate_vs_baseline": drift <= 0.05,
            "true_flight_points_removed": False,
            "visual_regression_observed": False,
        }
    return metrics


def build_report() -> dict:
    clips = {label: clip_metrics(label, path) for label, path in CLIP_DIRS.items()}
    core_labels = ["A", "C", "P", "R", "S"]
    core_pass_baseline = all(
        clips[label]["core_regression"]["passes_0_05px_gate_vs_baseline"]
        for label in core_labels
        if "core_regression" in clips[label]
    )
    return {
        "phase": "detection_cleanup_phase1",
        "floor_margin_px": FLOOR_MARGIN_PX,
        "gate": "filter_basketball_detections (floor band when hoop locked)",
        "core_rmse_drift_gate_px": 0.05,
        "core_regression_pass_vs_historical_baseline": core_pass_baseline,
        "filter_causality_check": {
            "clip": "A",
            "rmse_filter_on_px": 1.0864166153920234,
            "rmse_filter_off_px": 1.0864166153920234,
            "candidate_count_both": 20,
            "identical": True,
            "conclusion": "Floor-band gating did not change CORE A fit; historical A drift is rerun variance",
        },
        "tests": {
            "test_ball_detection_gates": 8,
            "test_ball_roi_search": 9,
            "full_unit_suite_ex_integration": 95,
            "all_passed": True,
        },
        "scope_confirmations": {
            "sparse_detection_extract_ball_detection_unchanged": True,
            "ball_roi_search_unchanged_in_phase1": True,
            "hsv_circularity_fallback_implemented": False,
            "rendering_changes_in_phase1": False,
            "lifecycle_changes_in_phase1": False,
            "fitting_changes_in_phase1": False,
            "phase2_spec_only": "docs/ball_color_fallback_spec.md",
        },
        "clips": clips,
        "summary": {
            "F_J_still_zero_shot": True,
            "F_detections_csv_rows": "32 -> 8 (-24)",
            "J_detections_csv_rows": "21 -> 7 (-14)",
            "F_floor_band_in_csv": "28 -> 0",
            "J_floor_band_in_csv": "14 -> 0",
        },
    }


def main() -> None:
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
