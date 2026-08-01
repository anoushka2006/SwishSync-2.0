"""Localized ROI re-inference when full-frame YOLO misses the ball."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from swishsync_cv.config import DetectionConfig, ShotCandidateConfig, SparseDetectionConfig
from swishsync_cv.data import DetectionRecord, SparseBallDetection
from swishsync_cv.tracking.motion_validation import validate_point
from swishsync_cv.tracking.parabola import fit_parabola


def predict_tracking_center(
    recent_points: list[SparseBallDetection],
    frame_index: int,
) -> tuple[float, float]:
    """Extrapolate ball center from the last two measured points."""

    if len(recent_points) == 1:
        return recent_points[0].x, recent_points[0].y

    previous = recent_points[-2]
    latest = recent_points[-1]
    dt = max(frame_index - latest.frame_index, 1)
    frame_delta = max(latest.frame_index - previous.frame_index, 1)
    vx = (latest.x - previous.x) / frame_delta
    vy = (latest.y - previous.y) / frame_delta
    return latest.x + vx * dt, latest.y + vy * dt


def compute_roi_bounds(
    center_x: float,
    center_y: float,
    padding_px: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Return clamped crop bounds (x1, y1, x2, y2) in full-frame coordinates."""

    half = max(padding_px, 1)
    x1 = int(max(0, round(center_x - half)))
    y1 = int(max(0, round(center_y - half)))
    x2 = int(min(frame_width, round(center_x + half)))
    y2 = int(min(frame_height, round(center_y + half)))
    if x2 <= x1:
        x2 = min(frame_width, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_height, y1 + 1)
    return x1, y1, x2, y2


def offset_detection_to_full_frame(
    detection: DetectionRecord,
    roi_x1: int,
    roi_y1: int,
) -> DetectionRecord:
    """Map a crop-space detection back to full-frame coordinates."""

    x1, y1, x2, y2 = detection.bbox_xyxy
    return replace(
        detection,
        bbox_xyxy=(
            x1 + roi_x1,
            y1 + roi_y1,
            x2 + roi_x1,
            y2 + roi_y1,
        ),
    )


def select_ball_detection(
    detections: list[DetectionRecord],
    min_confidence: float,
) -> DetectionRecord | None:
    basketball_detections = [
        detection
        for detection in detections
        if detection.label == "basketball" and detection.confidence >= min_confidence
    ]
    if not basketball_detections:
        return None
    return max(basketball_detections, key=lambda detection: detection.confidence)


def detection_to_sparse_point(
    detection: DetectionRecord,
    frame_index: int,
    timestamp_ms: float,
) -> SparseBallDetection:
    x, y = detection.center
    return SparseBallDetection(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        x=x,
        y=y,
        confidence=detection.confidence,
        interpolated=False,
    )


def validate_roi_candidate(
    candidate: SparseBallDetection,
    validation_points: list[SparseBallDetection],
    shot_config: ShotCandidateConfig,
    *,
    predicted_x: float,
    predicted_y: float,
    max_prediction_distance_px: float,
) -> bool:
    if not validation_points:
        return True

    prediction_distance = float(
        np.hypot(candidate.x - predicted_x, candidate.y - predicted_y)
    )
    if prediction_distance > max_prediction_distance_px:
        return False

    parabola_fit = None
    if len(validation_points) >= shot_config.min_validated_points_for_fit:
        parabola_fit = fit_parabola(validation_points)

    return validate_point(
        candidate_point=candidate,
        validated_points=validation_points,
        config=shot_config,
        parabola_fit=parabola_fit,
    )


def effective_roi_min_confidence(
    sparse_config: SparseDetectionConfig,
    validation_point_count: int,
) -> float:
    """Raise ROI confidence as more measured context is available."""

    if validation_point_count >= 8:
        return max(sparse_config.roi_min_confidence, sparse_config.min_confidence)
    if validation_point_count >= 4:
        return max(sparse_config.roi_min_confidence, 0.20)
    return sparse_config.roi_min_confidence


def try_roi_ball_detection(
    *,
    frame: np.ndarray,
    frame_index: int,
    timestamp_ms: float,
    recent_points: list[SparseBallDetection],
    validation_points: list[SparseBallDetection] | None,
    detect_crop,
    sparse_config: SparseDetectionConfig,
    detection_config: DetectionConfig,
    shot_config: ShotCandidateConfig,
) -> SparseBallDetection | None:
    """Run a secondary YOLO pass on a predicted ROI when full-frame detection missed."""

    del detection_config
    if not sparse_config.roi_search_enabled:
        return None
    if len(recent_points) < sparse_config.roi_min_measured_points:
        return None

    latest = recent_points[-1]
    frame_gap = frame_index - latest.frame_index
    if frame_gap <= 0 or frame_gap > sparse_config.roi_max_frames_since_last_measured:
        return None

    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = predict_tracking_center(recent_points, frame_index)
    roi_bounds = compute_roi_bounds(
        center_x,
        center_y,
        sparse_config.roi_padding_px,
        frame_width,
        frame_height,
    )

    crop_detections = detect_crop(
        frame=frame,
        crop_xyxy=roi_bounds,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        min_confidence=effective_roi_min_confidence(
            sparse_config,
            len(validation_points or recent_points),
        ),
    )
    selected = select_ball_detection(
        crop_detections,
        effective_roi_min_confidence(
            sparse_config,
            len(validation_points or recent_points),
        ),
    )
    if selected is None:
        return None

    sparse_point = detection_to_sparse_point(selected, frame_index, timestamp_ms)
    measured_for_validation = validation_points or recent_points
    if not validate_roi_candidate(
        sparse_point,
        measured_for_validation,
        shot_config,
        predicted_x=center_x,
        predicted_y=center_y,
        max_prediction_distance_px=sparse_config.roi_max_prediction_distance_px,
    ):
        return None
    return sparse_point
