"""Serialization helpers for detection and shot debugging artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from swishsync_cv.data import (
    ArcRenderMetadata,
    CompletedShot,
    DetectionRecord,
    FitDiagnostics,
    FrameDetections,
    ParabolaFit,
    PointDiagnostic,
    ShotAxis,
    ShotCandidate,
    ShotStoryMetadata,
    SparseBallDetection,
    TrajectoryPoint,
    TrustedFlightSelection,
    effective_point_source,
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
        "source": effective_point_source(point),
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


def arc_render_to_dict(metadata: ArcRenderMetadata) -> dict[str, object]:
    return {
        "fit_x_range": {
            "min": metadata.fit_x_range[0],
            "max": metadata.fit_x_range[1],
        },
        "render_x_range": {
            "min": metadata.render_x_range[0],
            "max": metadata.render_x_range[1],
        },
        "rim_center": (
            {"x": metadata.rim_center[0], "y": metadata.rim_center[1]}
            if metadata.rim_center is not None
            else None
        ),
        "rim_anchor_used": metadata.rim_anchor_used,
        "visual_extension_used": metadata.visual_extension_used,
        "observed_segment_end": (
            {"x": metadata.observed_segment_end[0], "y": metadata.observed_segment_end[1]}
            if metadata.observed_segment_end is not None
            else None
        ),
        "extended_segment_end": (
            {"x": metadata.extended_segment_end[0], "y": metadata.extended_segment_end[1]}
            if metadata.extended_segment_end is not None
            else None
        ),
    }


def shot_story_to_dict(story: ShotStoryMetadata) -> dict[str, object]:
    return {
        "release_frame": story.release_frame,
        "release_xy": {"x": story.release_xy[0], "y": story.release_xy[1]},
        "flight_start_frame": story.flight_start_frame,
        "flight_end_frame": story.flight_end_frame,
        "post_shot_start_frame": story.post_shot_start_frame,
        "pickup_frames": list(story.pickup_frames),
        "gap_predicted_frames": list(story.gap_predicted_frames),
        "show_post_shot": story.show_post_shot,
        "show_pickup": story.show_pickup,
        "trajectory_incomplete": story.trajectory_incomplete,
        "max_measured_gap_frames": story.max_measured_gap_frames,
    }


def trusted_flight_selection_to_dict(selection: TrustedFlightSelection) -> dict[str, object]:
    return {
        "trusted": [sparse_detection_to_dict(p) for p in selection.trusted],
        "excluded": [
            {"point": sparse_detection_to_dict(e.point), "reason": e.reason}
            for e in selection.excluded
        ],
        "max_gap_frames": selection.max_gap_frames,
        "trusted_count": len(selection.trusted),
        "excluded_count": len(selection.excluded),
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
        "continuity_points": [
            sparse_detection_to_dict(point) for point in candidate.continuity_points
        ],
        "pickup_points": [
            sparse_detection_to_dict(point) for point in candidate.pickup_points
        ],
        "post_shot_points": [
            sparse_detection_to_dict(point) for point in candidate.post_shot_points
        ],
        "gap_predicted_frames": list(candidate.gap_predicted_frames),
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
                "continuity_coverage": candidate.confidence.continuity_coverage,
                "gap_predicted_count": candidate.confidence.gap_predicted_count,
            }
            if candidate.confidence is not None
            else None
        ),
        "arc_render": (
            arc_render_to_dict(candidate.arc_render)
            if candidate.arc_render is not None
            else None
        ),
        "story": (
            shot_story_to_dict(candidate.story)
            if candidate.story is not None
            else None
        ),
        "trusted_flight": (
            trusted_flight_selection_to_dict(candidate.trusted_flight_debug)
            if candidate.trusted_flight_debug is not None
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
