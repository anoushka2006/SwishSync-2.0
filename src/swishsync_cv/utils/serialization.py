"""Serialization helpers for detection and shot debugging artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from swishsync_cv.data import (
    CompletedShot,
    DetectionRecord,
    FitDiagnostics,
    FrameDetections,
    ParabolaFit,
    PointDiagnostic,
    ShotAxis,
    ShotCandidate,
    SparseBallDetection,
    TrajectoryPoint,
)


def detection_to_dict(detection: DetectionRecord) -> dict[str, object]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    center_x, center_y = detection.center
    return {
        "frame_index": detection.frame_index,
        "timestamp_ms": detection.timestamp_ms,
        "label": detection.label,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "center_x": center_x,
        "center_y": center_y,
    }


def trajectory_point_to_dict(point: TrajectoryPoint) -> dict[str, object]:
    return {
        "frame_index": point.frame_index,
        "timestamp_ms": point.timestamp_ms,
        "x": point.x,
        "y": point.y,
        "confidence": point.confidence,
    }


def write_detections_jsonl(
    path: Path,
    frame_detections: list[FrameDetections],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for frame_record in frame_detections:
            payload = {
                "frame_index": frame_record.frame_index,
                "timestamp_ms": frame_record.timestamp_ms,
                "detections": [
                    detection_to_dict(detection)
                    for detection in frame_record.detections
                ],
            }
            file.write(json.dumps(payload) + "\n")


def write_detections_csv(path: Path, detections: list[DetectionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "label",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x",
        "center_y",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detection in detections:
            writer.writerow(detection_to_dict(detection))


def write_trajectory_json(path: Path, trajectory: list[TrajectoryPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [trajectory_point_to_dict(point) for point in trajectory]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sparse_detection_to_dict(point: SparseBallDetection) -> dict[str, object]:
    return {
        "frame_index": point.frame_index,
        "timestamp_ms": point.timestamp_ms,
        "x": point.x,
        "y": point.y,
        "confidence": point.confidence,
        "interpolated": point.interpolated,
    }


def parabola_fit_to_dict(fit: ParabolaFit) -> dict[str, object]:
    a, b, c = fit.coefficients
    return {
        "a": a,
        "b": b,
        "c": c,
        "r_squared": fit.r_squared,
        "apex_x": fit.apex_x,
        "apex_y": fit.apex_y,
        "x_min": fit.x_min,
        "x_max": fit.x_max,
        "weighted_r_squared": fit.weighted_r_squared,
        "weighted_residual_rmse": fit.weighted_residual_rmse,
    }


def point_diagnostic_to_dict(point: PointDiagnostic) -> dict[str, object]:
    return {
        "frame_index": point.frame_index,
        "x": point.x,
        "y": point.y,
        "confidence": point.confidence,
        "fitting_weight": point.fitting_weight,
        "residual_px": point.residual_px,
        "is_outlier": point.is_outlier,
        "used_in_fit": point.used_in_fit,
        "excluded_from_fit": point.excluded_from_fit,
    }


def fit_diagnostics_to_dict(diagnostics: FitDiagnostics) -> dict[str, object]:
    return {
        "point_count": diagnostics.point_count,
        "average_detection_confidence": diagnostics.average_detection_confidence,
        "weighted_residual_rmse": diagnostics.weighted_residual_rmse,
        "outlier_count": diagnostics.outlier_count,
        "initial_weighted_residual_rmse": diagnostics.initial_weighted_residual_rmse,
        "fit_point_count": diagnostics.fit_point_count,
        "points": [point_diagnostic_to_dict(point) for point in diagnostics.points],
    }


def shot_candidate_to_dict(candidate: ShotCandidate) -> dict[str, object]:
    payload: dict[str, object] = {
        "start_frame": candidate.start_frame,
        "end_frame": candidate.end_frame,
        "state": candidate.state,
        "insufficient_points_for_fit": candidate.insufficient_points_for_fit,
        "candidate_points": [
            sparse_detection_to_dict(point) for point in candidate.candidate_points
        ],
        "raw_points": [
            sparse_detection_to_dict(point) for point in candidate.candidate_points
        ],
        "validated_points": [
            sparse_detection_to_dict(point) for point in candidate.validated_points
        ],
        "excluded_debug_points": [
            sparse_detection_to_dict(point) for point in candidate.excluded_debug_points
        ],
        "parabola_fit": (
            parabola_fit_to_dict(candidate.parabola_fit)
            if candidate.parabola_fit is not None
            else None
        ),
        "fit_diagnostics": (
            fit_diagnostics_to_dict(candidate.fit_diagnostics)
            if candidate.fit_diagnostics is not None
            else None
        ),
        "confidence": (
            {
                "detection_confidence": candidate.confidence.detection_confidence,
                "trajectory_confidence": candidate.confidence.trajectory_confidence,
                "overall_confidence": candidate.confidence.overall_confidence,
            }
            if candidate.confidence is not None
            else None
        ),
    }
    return payload


def shot_axis_to_dict(axis: ShotAxis) -> dict[str, object]:
    return {
        "shooter_x": axis.shooter_x,
        "shooter_y": axis.shooter_y,
        "hoop_x": axis.hoop_x,
        "hoop_y": axis.hoop_y,
        "axis_dx": axis.axis_dx,
        "axis_dy": axis.axis_dy,
        "scale": axis.scale,
    }


def completed_shot_to_dict(shot: CompletedShot) -> dict[str, object]:
    return {
        "shot_id": shot.shot_id,
        "start_frame": shot.start_frame,
        "end_frame": shot.end_frame,
        "timestamp_ms": shot.timestamp_ms,
        "arc_points": [{"x": x, "y": y} for x, y in shot.arc_points],
        "normalized_arc_points": [
            {"x": x, "y": y} for x, y in shot.normalized_arc_points
        ],
        "normalized_apex": {
            "x": shot.normalized_apex[0],
            "y": shot.normalized_apex[1],
        },
        "shot_axis": shot_axis_to_dict(shot.shot_axis),
        "confidence": {
            "detection_confidence": shot.confidence.detection_confidence,
            "trajectory_confidence": shot.confidence.trajectory_confidence,
            "overall_confidence": shot.confidence.overall_confidence,
        },
        "parabola_r_squared": shot.parabola_r_squared,
        "outcome": shot.outcome,
    }


def write_finalized_shots_json(path: Path, shots: list[ShotCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [shot_candidate_to_dict(shot) for shot in shots]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_shots_json(path: Path, shots: list[CompletedShot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [completed_shot_to_dict(shot) for shot in shots]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
